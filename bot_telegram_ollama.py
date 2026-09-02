"""
Bot de Telegram que estima el precio de una vivienda.

Flujo:
    Telegram -> Mistral (Ollama) -> Random Forest -> Telegram

Mistral SOLO lee el mensaje y extrae 3 datos (sector, metros, anios).
El precio lo calcula UNICAMENTE el modelo Random Forest entrenado en Colab.

Ejecucion:
    python bot_telegram_ollama.py

El token NO esta escrito en el codigo: se pide al iniciar con getpass.
"""

import csv
import difflib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from getpass import getpass
from pathlib import Path

import joblib
import pandas as pd
import requests

# --------------------------------------------------------------------------
# 1. CONFIGURACION
# --------------------------------------------------------------------------

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_casas.pkl"
RUTA_SECTORES = CARPETA / "sectores_ames.json"
RUTA_LOG = CARPETA / "historial_predicciones.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODELO = "mistral"

TELEGRAM_API = "https://api.telegram.org/bot{token}/{metodo}"

# Limites razonables para descartar datos absurdos
MIN_METROS, MAX_METROS = 15.0, 1000.0
MIN_ANIOS, MAX_ANIOS = 0, 200

# Unicos sectores que el modelo conoce. Si el usuario no nombra uno de estos,
# el bot NO estima nada: responde que le faltan datos.
SECTORES_VALIDOS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr",
    "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR",
    "MeadowV", "Mitchel", "NAmes", "NPkVill", "NWAmes",
    "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer",
    "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]

# --------------------------------------------------------------------------
# 2. NOMBRES ALTERNATIVOS DE LOS SECTORES  (extra)
#    El dataset usa codigos como "NAmes" u "OldTown". La gente escribe
#    "North Ames" o "Old Town". Este diccionario traduce lo que escribe
#    el usuario al codigo exacto que espera el modelo.
# --------------------------------------------------------------------------

ALIAS_SECTORES = {
    "Blmngtn": ["Bloomington Heights", "Bloomington"],
    "Blueste": ["Bluestem"],
    "BrDale": ["Briardale"],
    "BrkSide": ["Brookside", "Brook Side"],
    "ClearCr": ["Clear Creek"],
    "CollgCr": ["College Creek", "College"],
    "Crawfor": ["Crawford"],
    "Edwards": ["Edwards"],
    "Gilbert": ["Gilbert"],
    "IDOTRR": ["Iowa DOT and Rail Road", "Iowa DOT", "zona del ferrocarril"],
    "MeadowV": ["Meadow Village", "Meadow"],
    "Mitchel": ["Mitchell"],
    "NAmes": ["North Ames", "Norte de Ames", "Ames Norte"],
    "NoRidge": ["Northridge", "North Ridge"],
    "NPkVill": ["Northpark Villa", "North Park Villa"],
    "NWAmes": ["Northwest Ames", "Noroeste de Ames"],
    "NridgHt": ["Northridge Heights", "North Ridge Heights"],
    "OldTown": ["Old Town", "Casco Viejo", "Centro Historico", "Ciudad Vieja"],
    "SWISU": ["South West Iowa State University", "SW Iowa State", "Universidad"],
    "Sawyer": ["Sawyer"],
    "SawyerW": ["Sawyer West", "Sawyer Oeste"],
    "Somerst": ["Somerset"],
    "StoneBr": ["Stone Brook", "Stonebrook"],
    "Timber": ["Timberland", "Timber"],
    "Veenker": ["Veenker"],
}


def normalizar(texto):
    """Pasa a minusculas, quita tildes y deja solo letras y numeros."""
    texto = str(texto)
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", texto.lower())


# --------------------------------------------------------------------------
# 3. CARGA DEL MODELO Y LOS SECTORES
# --------------------------------------------------------------------------

def cargar_recursos():
    if not RUTA_MODELO.exists():
        sys.exit(f"[ERROR] Falta {RUTA_MODELO.name}. Descargalo de Colab.")
    if not RUTA_SECTORES.exists():
        sys.exit(f"[ERROR] Falta {RUTA_SECTORES.name}. Descargalo de Colab.")

    modelo = joblib.load(RUTA_MODELO)
    with open(RUTA_SECTORES, encoding="utf-8") as archivo:
        sectores = json.load(archivo)

    # Se aceptan unicamente los 25 sectores oficiales del dataset.
    sectores = [s for s in SECTORES_VALIDOS if s in sectores]
    if len(sectores) != len(SECTORES_VALIDOS):
        ausentes = [s for s in SECTORES_VALIDOS if s not in sectores]
        print("[AVISO] El JSON no trae estos sectores:", ", ".join(ausentes))

    # Tabla de busqueda: texto normalizado -> codigo oficial del sector
    busqueda = {}
    for sector in sectores:
        busqueda[normalizar(sector)] = sector
        for alias in ALIAS_SECTORES.get(sector, []):
            busqueda[normalizar(alias)] = sector

    return modelo, sectores, busqueda


def resolver_sector(texto, busqueda):
    """Convierte lo que escribio el usuario en uno de los 25 sectores validos.

    Es estricta a proposito: si el usuario habla de Quito, Cumbaya o de
    cualquier zona que no este en la lista, devuelve None y el bot responde
    que le faltan datos. Antes que inventar un barrio parecido, prefiere
    no estimar nada.

    Orden de busqueda:
      1. El texto completo es el nombre o alias de un sector.
      2. El nombre del sector aparece dentro de la frase
         ("una casa en Old Town de 120 m2").
      3. Tolerancia a errores de tipeo, pero palabra por palabra y exigente
         ("Gilbertt" si pasa, "Quito" no se parece a nada).
    """
    if not texto:
        return None

    clave = normalizar(texto)
    if not clave:
        return None

    # 1. Coincidencia exacta.
    if clave in busqueda:
        return busqueda[clave]

    # 2. El nombre aparece dentro de la frase. Se revisan los nombres mas
    #    largos primero para que "Sawyer West" gane sobre "Sawyer".
    for candidato in sorted(busqueda, key=len, reverse=True):
        if len(candidato) >= 5 and candidato in clave:
            return busqueda[candidato]

    # 3. Errores de tipeo: se comparan grupos de 1 a 3 palabras contra la
    #    lista, con un umbral alto para no aceptar cualquier parecido.
    palabras = [normalizar(p) for p in re.split(r"\s+", str(texto)) if normalizar(p)]
    llaves = list(busqueda)
    for tamano in (3, 2, 1):
        for inicio in range(len(palabras) - tamano + 1):
            trozo = "".join(palabras[inicio:inicio + tamano])
            if len(trozo) < 5:
                continue
            if trozo in busqueda:
                return busqueda[trozo]
            parecidos = difflib.get_close_matches(trozo, llaves, n=1, cutoff=0.85)
            if parecidos:
                return busqueda[parecidos[0]]

    return None


def nombre_legible(sector):
    """Devuelve 'Old Town (OldTown)' para mostrarlo bonito en el chat."""
    alias = ALIAS_SECTORES.get(sector)
    if alias and alias[0] != sector:
        return f"{alias[0]} ({sector})"
    return sector


# --------------------------------------------------------------------------
# 4. MISTRAL: convierte el mensaje libre en datos estructurados
# --------------------------------------------------------------------------

PROMPT_BASE = """Eres un extractor de datos. Del mensaje del usuario debes obtener
exactamente tres datos sobre una vivienda y devolverlos en JSON.

Devuelve SOLO este objeto JSON, sin explicaciones:
{{"sector": "<texto>", "metros": <numero>, "anios": <numero>}}

Reglas:
- "sector": el barrio o zona que menciona el usuario, tal como lo escribio.
- "metros": el area en metros cuadrados, solo el numero.
- "anios": la antiguedad de la casa en anios, solo el numero.
- Si un dato no aparece en el mensaje, pon null. NUNCA lo inventes.
- No calcules ni menciones ningun precio.

Sectores validos de referencia: {sectores}

Mensaje del usuario: {mensaje}"""


def consultar_mistral(mensaje, sectores):
    """Envia el mensaje a Ollama y devuelve un diccionario con los 3 datos."""
    prompt = PROMPT_BASE.format(sectores=", ".join(sectores), mensaje=mensaje)

    payload = {
        "model": OLLAMA_MODELO,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }

    respuesta = requests.post(OLLAMA_URL, json=payload, timeout=120)
    respuesta.raise_for_status()
    contenido = respuesta.json()["message"]["content"]
    return json.loads(contenido)


def extraer_con_regex(mensaje):
    """Plan B (extra): si Ollama se cae o responde mal, no se rompe la demo.

    Busca los numeros del mensaje con expresiones regulares.
    """
    texto = mensaje.lower().replace(",", ".")

    metros = None
    patron_metros = r"(\d+(?:\.\d+)?)\s*(?:m2|m²|mts2|metros cuadrados|metros|m)\b"
    encontrado = re.search(patron_metros, texto)
    if encontrado:
        metros = float(encontrado.group(1))

    anios = None
    patron_anios = r"(\d+(?:\.\d+)?)\s*(?:anios|años|anos|year|anio|año)\b"
    encontrado = re.search(patron_anios, texto)
    if not encontrado:
        encontrado = re.search(r"antig[uü]edad\D{0,10}(\d+)", texto)
    if encontrado:
        anios = float(encontrado.group(1))

    return {"sector": mensaje, "metros": metros, "anios": anios}


def obtener_datos(mensaje, sectores):
    """Intenta con Mistral; si falla, usa el plan B. Devuelve (datos, origen)."""
    try:
        return consultar_mistral(mensaje, sectores), "mistral"
    except Exception as error:
        print(f"[AVISO] Ollama fallo ({error}). Usando extraccion de respaldo.")
        return extraer_con_regex(mensaje), "respaldo"


# --------------------------------------------------------------------------
# 5. VALIDACION
# --------------------------------------------------------------------------

def a_numero(valor):
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    encontrado = re.search(r"-?\d+(?:[.,]\d+)?", str(valor))
    if not encontrado:
        return None
    return float(encontrado.group(0).replace(",", "."))


ETIQUETAS_FALTANTES = {
    "sector": "el sector",
    "metros": "los metros cuadrados",
    "anios": "los años de antigüedad",
    "metros_rango": f"unos metros cuadrados creíbles (entre {MIN_METROS:.0f} y {MAX_METROS:.0f})",
    "anios_rango": f"una antigüedad creíble (entre {MIN_ANIOS} y {MAX_ANIOS} años)",
}


def validar(datos, busqueda):
    """Revisa los 3 datos. Devuelve (datos_limpios, lista_de_faltantes).

    La lista trae codigos ('sector', 'metros', 'anios', 'metros_rango',
    'anios_rango'); el texto que ve el usuario se arma en mensaje_faltantes.
    """
    faltantes = []

    sector = resolver_sector(datos.get("sector"), busqueda)
    if sector is None:
        faltantes.append("sector")

    metros = a_numero(datos.get("metros"))
    if metros is None:
        faltantes.append("metros")
    elif not (MIN_METROS <= metros <= MAX_METROS):
        faltantes.append("metros_rango")
        metros = None

    anios = a_numero(datos.get("anios"))
    if anios is None:
        faltantes.append("anios")
    elif not (MIN_ANIOS <= anios <= MAX_ANIOS):
        faltantes.append("anios_rango")
        anios = None

    return {"sector": sector, "metros": metros, "anios": anios}, faltantes


def unir(elementos):
    """['a', 'b', 'c'] -> 'a, b y c'"""
    elementos = list(elementos)
    if len(elementos) == 1:
        return elementos[0]
    return ", ".join(elementos[:-1]) + " y " + elementos[-1]


def mensaje_faltantes(faltantes, sectores=None):
    """Respuesta cuando no se puede valorar.

    Si el problema es el sector, se muestra directamente la lista de
    sectores disponibles en lugar de una explicacion larga.
    """
    if "sector" in faltantes:
        disponibles = sectores or SECTORES_VALIDOS
        lineas = [
            "SECTOR NO DISPONIBLE",
            "─────────────────────",
            "Solo tengo datos de estos sectores:",
            "",
        ]
        lineas += [nombre_legible(s) for s in disponibles]

        otros = [ETIQUETAS_FALTANTES[c] for c in faltantes if c != "sector"]
        if otros:
            verbo = "falta" if len(otros) == 1 else "faltan"
            lineas += ["", f"También me {verbo} {unir(otros)}."]
        return "\n".join(lineas)

    pendientes = [ETIQUETAS_FALTANTES[codigo] for codigo in faltantes]
    return (
        "DATOS INCOMPLETOS\n"
        "─────────────────────\n"
        f"Me {'falta' if len(pendientes) == 1 else 'faltan'} {unir(pendientes)}.\n\n"
        "Ejemplo: «Casa en Old Town de 120 metros cuadrados y 25 años»"
    )


# --------------------------------------------------------------------------
# 6. PREDICCION CON RANDOM FOREST
# --------------------------------------------------------------------------

def predecir(modelo, sector, metros, anios):
    """Devuelve el precio estimado y, si se puede, un rango de los arboles."""
    entrada = pd.DataFrame([{
        "Neighborhood": sector,
        "Area_m2": metros,
        "Antiguedad": anios,
    }])

    precio = float(modelo.predict(entrada)[0])

    # Extra: cada arbol del bosque vota un precio. Mirando la dispersion de
    # esos votos obtenemos un rango, no solo un numero suelto.
    rango = None
    try:
        preparado = modelo.named_steps["preprocesamiento"].transform(entrada)
        bosque = modelo.named_steps["random_forest"]
        votos = pd.Series([float(arbol.predict(preparado)[0]) for arbol in bosque.estimators_])
        rango = (float(votos.quantile(0.10)), float(votos.quantile(0.90)))
    except Exception:
        pass

    return precio, rango


def guardar_log(chat_id, mensaje, datos, precio, origen):
    """Extra: guarda cada prediccion en un CSV para mostrarlo en la defensa."""
    nuevo = not RUTA_LOG.exists()
    with open(RUTA_LOG, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        if nuevo:
            escritor.writerow(
                ["fecha", "chat_id", "mensaje", "sector", "metros", "anios", "precio", "extraccion"]
            )
        escritor.writerow([
            datetime.now().isoformat(timespec="seconds"),
            chat_id, mensaje, datos["sector"], datos["metros"],
            datos["anios"], round(precio, 2), origen,
        ])


# --------------------------------------------------------------------------
# 7. TELEGRAM
# --------------------------------------------------------------------------

def enviar_mensaje(token, chat_id, texto):
    url = TELEGRAM_API.format(token=token, metodo="sendMessage")
    try:
        requests.post(url, json={"chat_id": chat_id, "text": texto}, timeout=30)
    except Exception as error:
        print(f"[ERROR] No se pudo enviar el mensaje: {error}")


def recibir_actualizaciones(token, offset):
    url = TELEGRAM_API.format(token=token, metodo="getUpdates")
    parametros = {"timeout": 30}
    if offset is not None:
        parametros["offset"] = offset
    respuesta = requests.get(url, params=parametros, timeout=40)
    respuesta.raise_for_status()
    return respuesta.json().get("result", [])


def verificar_token(token):
    url = TELEGRAM_API.format(token=token, metodo="getMe")
    datos = requests.get(url, timeout=20).json()
    if not datos.get("ok"):
        sys.exit("[ERROR] El token no es valido.")
    return datos["result"]["username"]


def estado_ollama():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 8. RESPUESTAS A COMANDOS  (extra)
# --------------------------------------------------------------------------

TEXTO_AYUDA = (
    "Necesito tres datos, escritos como quieras:\n"
    "  - el sector (de mi lista)\n"
    "  - los metros cuadrados\n"
    "  - los años de antigüedad\n\n"
    "Ejemplo:\n"
    "«Tengo una casa en Old Town de 120 metros cuadrados y 25 años»\n\n"
    "Comandos:\n"
    "/sectores - los 25 sectores disponibles\n"
    "/detalle - desglose de la última valoración\n"
    "/estado - estado del modelo y de Ollama"
)


def formatear_respuesta(datos, precio, rango):
    """Ficha de valoracion que recibe el usuario en Telegram."""
    precio_m2 = precio / datos["metros"]

    lineas = [
        "INFORME DE VALORACIÓN",
        "─────────────────────",
        f"Sector: {nombre_legible(datos['sector'])}",
        f"Área: {datos['metros']:,.0f} m²",
        f"Antigüedad: {datos['anios']:.0f} años",
        "",
        f"Valor estimado: ${precio:,.0f}",
    ]

    if rango:
        bajo, alto = rango
        lineas.append(f"Rango probable: ${bajo:,.0f} — ${alto:,.0f}")

    lineas += [
        f"Precio por m²: ${precio_m2:,.0f}",
        "",
        "Escribe /detalle para ver el desglose del cálculo.",
    ]
    return "\n".join(lineas)


def responder_comando(comando, contexto):
    modelo, sectores, ultima = contexto["modelo"], contexto["sectores"], contexto["ultima"]

    if comando in ("/start", "/ayuda", "/help"):
        return (
            "Servicio de valoración de viviendas de Ames, Iowa.\n"
            "Leo tu mensaje con Mistral y el precio lo calcula un Random Forest.\n\n"
            + TEXTO_AYUDA
        )

    if comando == "/sectores":
        return (
            f"SECTORES DISPONIBLES ({len(sectores)})\n"
            "─────────────────────\n"
            + "\n".join(nombre_legible(s) for s in sectores)
            + "\n\nCualquier otra zona la tomo como dato faltante."
        )

    if comando == "/estado":
        ollama = "activo" if estado_ollama() else "apagado (se usa el respaldo)"
        arboles = modelo.named_steps["random_forest"].n_estimators
        return (
            "ESTADO DEL SISTEMA\n"
            "─────────────────────\n"
            f"Modelo Random Forest: cargado ({arboles} árboles)\n"
            f"Ollama / Mistral: {ollama}\n"
            f"Sectores habilitados: {len(sectores)}"
        )

    if comando == "/detalle":
        if not ultima:
            return (
                "Todavía no hay ninguna valoración.\n"
                "Envíame primero los datos de una vivienda."
            )
        precio_m2 = ultima["precio"] / ultima["metros"]
        lineas = [
            "DESGLOSE DE LA ÚLTIMA VALORACIÓN",
            "─────────────────────",
            f"Sector: {nombre_legible(ultima['sector'])}",
            f"Área: {ultima['metros']:,.0f} m²",
            f"Antigüedad: {ultima['anios']:.0f} años",
            f"Valor estimado: ${ultima['precio']:,.0f}",
            f"Precio por m²: ${precio_m2:,.0f}",
        ]
        if ultima["rango"]:
            bajo, alto = ultima["rango"]
            lineas.append(f"Votos de los árboles (P10–P90): ${bajo:,.0f} — ${alto:,.0f}")
        lineas += [
            f"Datos extraídos por: {ultima['origen']}",
            "",
            "Estimación educativa con datos históricos de Ames, Iowa.",
            "No es una tasación profesional ni refleja precios de Ecuador.",
        ]
        return "\n".join(lineas)

    return None


# --------------------------------------------------------------------------
# 9. PROCESAR UN MENSAJE
# --------------------------------------------------------------------------

def procesar(mensaje, chat_id, contexto):
    texto = mensaje.strip()

    if texto.startswith("/"):
        comando = texto.split()[0].split("@")[0].lower()
        respuesta = responder_comando(comando, contexto)
        return respuesta or "Ese comando no existe. Escribe /ayuda para ver los disponibles."

    datos_crudos, origen = obtener_datos(texto, contexto["sectores"])
    datos, faltantes = validar(datos_crudos, contexto["busqueda"])

    if faltantes:
        print(f"  -> sin estimacion. Faltan: {', '.join(faltantes)}")
        return mensaje_faltantes(faltantes, contexto["sectores"])

    precio, rango = predecir(
        contexto["modelo"], datos["sector"], datos["metros"], datos["anios"]
    )

    contexto["ultima"] = {**datos, "precio": precio, "rango": rango, "origen": origen}
    guardar_log(chat_id, texto, datos, precio, origen)

    print(f"  -> {datos['sector']} | {datos['metros']} m2 | {datos['anios']} anios "
          f"| ${precio:,.0f} | {origen}")

    return formatear_respuesta(datos, precio, rango)


# --------------------------------------------------------------------------
# 10. PROGRAMA PRINCIPAL
# --------------------------------------------------------------------------

def main():
    print("=" * 55)
    print(" Bot inmobiliario: Telegram -> Mistral -> Random Forest")
    print("=" * 55)

    modelo, sectores, busqueda = cargar_recursos()
    print(f"Modelo cargado. {len(sectores)} sectores disponibles.")

    if estado_ollama():
        print("Ollama respondiendo en localhost:11434.")
    else:
        print("[AVISO] Ollama no responde. Se usara la extraccion de respaldo.")

    # El token nunca se escribe en el codigo ni se sube a GitHub.
    token = os.environ.get("TELEGRAM_TOKEN") or getpass("Token del bot de Telegram: ")
    usuario = verificar_token(token.strip())
    print(f"Conectado como @{usuario}. Esperando mensajes... (Ctrl+C para salir)\n")

    contexto = {"modelo": modelo, "sectores": sectores, "busqueda": busqueda, "ultima": None}
    offset = None

    while True:
        try:
            actualizaciones = recibir_actualizaciones(token, offset)
        except Exception as error:
            print(f"[AVISO] Error de conexion con Telegram: {error}")
            continue

        for actualizacion in actualizaciones:
            offset = actualizacion["update_id"] + 1
            mensaje = actualizacion.get("message") or {}
            texto = mensaje.get("text")
            chat_id = (mensaje.get("chat") or {}).get("id")

            if not texto or chat_id is None:
                continue

            print(f"[{datetime.now():%H:%M:%S}] {chat_id}: {texto}")
            try:
                respuesta = procesar(texto, chat_id, contexto)
            except Exception as error:
                print(f"[ERROR] {error}")
                respuesta = "Hubo un problema procesando el mensaje. Inténtalo otra vez."

            enviar_mensaje(token, chat_id, respuesta)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot detenido.")