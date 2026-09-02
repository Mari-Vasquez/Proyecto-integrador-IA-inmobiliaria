# IA inmobiliaria con Random Forest, Ollama y Telegram

Bot de Telegram que estima el precio de una vivienda a partir de un mensaje escrito en
lenguaje normal, por ejemplo:

> *Tengo una casa en Old Town de 120 metros cuadrados y 25 años.*

El bot responde solo con el precio estimado:

> `$184,350`

Proyecto Integrador — Pontificia Universidad Católica del Ecuador.

---

## Objetivo

Entender cómo se conectan cuatro piezas de un sistema de IA real:

1. **Datos** — dataset público de viviendas.
2. **Machine Learning** — un Random Forest que predice el precio.
3. **LLM** — Mistral, que lee el mensaje libre y saca los datos.
4. **Interfaz** — Telegram, por donde entra y sale la conversación.

## Arquitectura

```
   Usuario (Telegram)
          |
          v
   getUpdates  ──────────────►  bot_telegram_ollama.py
                                        |
                                        v
                          Ollama + Mistral (localhost:11434)
                          extrae {"sector", "metros", "anios"}
                                        |
                                        v
                          Validación en Python
                          (sector válido, rangos razonables)
                                        |
                                        v
                          Random Forest (modelo_casas.pkl)
                          calcula el precio
                                        |
                                        v
   Usuario (Telegram)  ◄──────  sendMessage  →  "$184,350"
```

**Punto clave:** Mistral **no calcula el precio**. Solo traduce el texto libre a datos
estructurados. El precio lo produce únicamente el Random Forest entrenado con casas reales.

## Datos

Dataset público *house_prices* (Ames Housing), OpenML ID **42165** — viviendas de Ames, Iowa.

| Dato del usuario | Variable original | Variable del modelo |
|---|---|---|
| Sector | `Neighborhood` | `Neighborhood` |
| Metros cuadrados | `GrLivArea` (ft²) | `Area_m2` |
| Años | `YrSold - YearBuilt` | `Antiguedad` |
| Precio (objetivo) | `SalePrice` | `SalePrice` |

> **Nota académica:** el resultado es una estimación educativa con datos históricos de
> Ames, Iowa. No es una tasación profesional ni refleja precios actuales de Ecuador.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `01_entrenamiento_random_forest_colab.ipynb` | Entrena y evalúa el modelo en Google Colab |
| `bot_telegram_ollama.py` | El bot: Telegram + Ollama + Random Forest |
| `probar_sin_telegram.py` | Prueba el flujo desde la consola, sin Telegram |
| `sectores_ames.json` | Sectores válidos del dataset |
| `requirements_local.txt` | Dependencias de Python |
| `modelo_casas.pkl` | Modelo entrenado (se descarga de Colab) |

## Cómo ejecutarlo

### 1. Entrenar el modelo

Abrir `01_entrenamiento_random_forest_colab.ipynb` en Google Colab y ejecutar todas las
celdas en orden. Al final se descargan `modelo_casas.pkl`, `sectores_ames.json` y
`requirements_local.txt`. Copiar esos archivos a esta carpeta.

### 2. Instalar Ollama y Mistral

```bash
ollama pull mistral
ollama run mistral      # prueba rápida; salir con /bye
```

Ollama debe quedar corriendo en segundo plano (expone `http://localhost:11434`).

### 3. Crear el bot de Telegram

En Telegram, hablar con **@BotFather** → `/newbot` → guardar el token.

El token **no se escribe en el código**. El programa lo pide al iniciar con `getpass`.

### 4. Instalar dependencias y ejecutar

```bash
pip install -r requirements_local.txt
python bot_telegram_ollama.py
```

Después escribir al bot desde Telegram.

## Comandos del bot

| Comando | Qué hace |
|---|---|
| `/ayuda` | Explica cómo escribir el mensaje |
| `/sectores` | Lista los 25 sectores disponibles |
| `/detalle` | Desglosa la última estimación (precio por m², rango) |
| `/estado` | Verifica que el modelo y Ollama estén activos |

## Ejemplos de mensajes válidos

- Tengo una casa en Old Town de 120 metros cuadrados y 25 años.
- ¿Cuánto vale una vivienda de 95 m2, 10 años, ubicada en North Ames?
- Casa en Gilbert, 150 metros cuadrados, antigüedad 8 años.
- Departamento en Somerset de 80 m2 con 5 años.

Si falta alguno de los tres datos, el bot lo dice. **Nunca inventa valores.**

## Extras añadidos

Además de lo pedido en la guía:

- **Nombres alternativos de sectores.** El dataset usa códigos como `NAmes` u `OldTown`.
  El bot entiende "North Ames", "Old Town", "casco viejo" e incluso errores de tipeo
  (coincidencia difusa con `difflib`).
- **Extracción de respaldo.** Si Ollama se cae en plena demostración, un extractor con
  expresiones regulares mantiene el sistema funcionando.
- **Rango de confianza.** Cada árbol del bosque vota un precio; `/detalle` muestra la
  dispersión de esos votos (percentiles 10–90).
- **Historial en CSV.** Cada predicción queda registrada en `historial_predicciones.csv`
  (ignorado por git).
- **Modo consola.** `probar_sin_telegram.py` permite demostrar el flujo sin internet.
- **Análisis en el notebook.** Comparación contra un modelo base, validación cruzada,
  importancia de variables y gráfico de real vs estimado.

## Seguridad

- El token se pide con `getpass`, nunca se guarda en el código.
- `.gitignore` bloquea `.env`, `token.txt` y el historial de predicciones.
- Este repositorio no contiene credenciales.
