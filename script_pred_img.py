import pandas as pd
import numpy as np
import os
from joblib import load
import optuna

# --- Paths ---
study_path = "work/db.sqlite3"
artifact_dir = "work/optuna_temp_artifacts"
train_csv_path = "input/petfinder-adoption-prediction/train/train.csv"
output_csv_path = "work/output/resnet_preds_completado.csv"

# --- Cargar mejor trial ---
study = optuna.load_study(
    study_name="04 ResNet_1.0.0",
    storage=f"sqlite:///{study_path}"
)
trial_number = study.best_trial.number

# --- Cargar predicciones del val ---
val_pred_path = os.path.join(artifact_dir, f"test_04 ResNet_1.0.0_{trial_number}.joblib")
df_val = load(val_pred_path)
df_val["resnet_pred"] = df_val["pred"].apply(lambda x: np.argmax(x))

# --- Cargar train completo ---
df_train = pd.read_csv(train_csv_path)

# --- Unir predicciones con train ---
# Primero unimos por PetID para los que tienen predicción
df_merged = df_train.merge(df_val[["PetID", "resnet_pred"]], on="PetID", how="left")

# Para los que NO tienen resnet_pred, usamos el valor real
df_merged["resnet_pred"] = df_merged["resnet_pred"].fillna(df_merged["AdoptionSpeed"]).astype(int)

# --- Exportar ---
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df_merged[["PetID", "resnet_pred", "AdoptionSpeed"]].to_csv(output_csv_path, index=False)
print(f"CSV completo exportado a: {output_csv_path}")
