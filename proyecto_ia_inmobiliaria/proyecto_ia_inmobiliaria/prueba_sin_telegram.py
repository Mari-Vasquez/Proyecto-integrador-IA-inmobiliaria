"""
Prueba el flujo completo SIN Telegram, desde la consola.

Sirve para:
  - verificar que el modelo y Ollama funcionan antes de la demostracion
  - seguir mostrando el sistema si el wifi del aula falla

Muestra exactamente la misma respuesta que enviaria el bot por Telegram.

Uso:
    python probar_sin_telegram.py                    (modo interactivo)
    python probar_sin_telegram.py --auto             (corre ejemplos de prueba)
"""

import sys

from bot_telegram_ollama import (
    cargar_recursos,
    estado_ollama,
    formatear_respuesta,
    mensaje_faltantes,
    obtener_datos,
    predecir,
    validar,
)

EJEMPLOS = [
    "Tengo una casa en Old Town de 120 metros cuadrados y 25 anios.",
    "Cuanto vale una vivienda de 95 m2, 10 anios, ubicada en North Ames?",
    "Casa en Gilbert, 150 metros cuadrados, antiguedad 8 anios.",
    "Departamento en Somerset de 80 m2 con 5 anos",
    "Una casa en el casco viejo",                    # faltan metros y anios
    "Casa en Cumbaya de 200 m2 y 10 anios",          # sector fuera de la lista
    "Depa en Quito norte, 90 metros cuadrados, 3 anios",  # sector fuera de la lista
]


def imprimir(texto):
    """Imprime sin romperse si la consola de Windows no soporta emojis."""
    try:
        print(texto)
    except UnicodeEncodeError:
        codificacion = sys.stdout.encoding or "ascii"
        print(texto.encode(codificacion, errors="ignore").decode(codificacion))


def analizar(mensaje, contexto):
    imprimir("\n" + "=" * 55)
    imprimir(f"Mensaje: {mensaje}")

    crudos, origen = obtener_datos(mensaje, contexto["sectores"])
    imprimir(f"  [extraido por {origen}] {crudos}")

    datos, faltantes = validar(crudos, contexto["busqueda"])

    imprimir("--- RESPUESTA DEL BOT -------------------------")
    if faltantes:
        imprimir(mensaje_faltantes(faltantes, contexto["sectores"]))
        return

    precio, rango = predecir(
        contexto["modelo"], datos["sector"], datos["metros"], datos["anios"]
    )
    imprimir(formatear_respuesta(datos, precio, rango))


def main():
    modelo, sectores, busqueda = cargar_recursos()
    contexto = {"modelo": modelo, "sectores": sectores, "busqueda": busqueda}

    imprimir(f"Modelo cargado. Sectores validos: {len(sectores)}")
    imprimir("Ollama: " + ("activo" if estado_ollama() else "APAGADO (se usara el respaldo)"))

    if "--auto" in sys.argv:
        for ejemplo in EJEMPLOS:
            analizar(ejemplo, contexto)
        return

    imprimir("\nEscribe un mensaje (o 'salir'):")
    while True:
        try:
            mensaje = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if mensaje.lower() in ("salir", "exit", "quit", ""):
            break
        analizar(mensaje, contexto)


if __name__ == "__main__":
    main()