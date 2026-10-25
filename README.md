# Modelo predictivo de edad para moluscos

## Descripcion
El objetivo principal de este proyecto es lograr realizar un modelo para predecir la edad del molusco Abalon, utilizando un dataset que contiene mediciones fisicas de dicho animal y su edad estimada (anillos)
## Requisitos

- Python **3.13**
- [uv](https://docs.astral.sh/uv/) (recomendado) o pip
- Git

## Reproducir el entorno

### Opción 1: con uv (recomendada)

Recrea el entorno exacto a partir de `uv.lock`:

```bash
git clone <url-del-repositorio>
cd tp5-abalone
uv sync
```

`uv sync` instala la versión de Python indicada en `.python-version` (si no
está disponible) y crea el entorno virtual `.venv` con las versiones exactas
de las dependencias.

### Opción 2: con pip

```bash
git clone <url-del-repositorio>
cd tp5-abalone
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### Usar el entorno en VS Code / Jupyter

Seleccionar como intérprete o kernel el Python de `.venv`
(`Ctrl + Shift + P` → *Python: Select Interpreter*).

## Estructura del proyecto

```
tp5-abalone/
├── data/
│   ├── raw/                 # datos originales (sin modificar)
│   ├── processed/           # datos limpios o transformados
│   └── data_log.json        # registro de transformaciones
├── src/
│   ├── preprocess.py        # limpieza y transformaciones
│   └── train_model.py       # entrenamiento y evaluación
├── results/
│   ├── metrics_v1.json      # métricas del modelo
│   └── experiments_log.csv  # registro de experimentos
├── models/                  # modelos entrenados (serializados)
├── notebooks/
│   └── tp5.ipynb            # notebook del trabajo práctico
├── .gitignore               # archivos excluidos del control de versiones
├── .python-version          # versión de Python del proyecto
├── pyproject.toml           # dependencias declaradas
├── uv.lock                  # versiones exactas resueltas (lockfile)
├── requirements.txt         # dependencias fijadas (compatibilidad con pip)
└── README.md
```
## Uso

### 1. Preprocesamiento

Desde la carpeta principal del proyecto:

```bash
uv run src/preprocess.py
```

Lee `data/raw/abalone_raw.csv` y genera:

- `data/processed/abalone_clean.csv`: dataset limpio.
- `data/data_log.json`: registro de transformaciones (fecha, archivo de
  origen, transformaciones y parámetros aplicados).

Transformaciones aplicadas:

1. Nueva columna `age = rings + 1.5` (edad estimada en años).
2. Codificación one-hot de `sex` (`sex_F`, `sex_I`, `sex_M`).
3. Eliminación de valores atípicos de `height`: registros con `height <= 0`
   (físicamente imposibles) o `height > 0.4` (valores aislados). Se eliminan
   4 registros (4177 → 4173).

El escalado de variables **no** se aplica en este paso: se realiza en el
entrenamiento, ajustado solo con los datos de entrenamiento.

> Cada vez que se modifica `src/preprocess.py`, se debe regenerar
> `abalone_clean.csv` y commitear ambos archivos en el mismo commit.

### 2. Entrenamiento

*(Se completará en el Ejercicio 3.)*
