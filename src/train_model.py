# =====================================================
# Entrenamiento y evaluación del modelo (Abalone)
#
# Ejecutar desde la carpeta principal del proyecto (tp5-abalone):
#     uv run src/train_model.py
# =====================================================

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "v1"
SEMILLA = 42          # la misma semilla en el split y en el modelo
TEST_SIZE = 0.2       # 20% de los datos para test
HIPERPARAMETROS = {"n_estimators": 100, "max_depth": 10}


# -----------------------------------------------------
# 1. Cargar el dataset limpio (salida del preprocesamiento)
# -----------------------------------------------------
df = pd.read_csv("data/processed/abalone_clean.csv")
