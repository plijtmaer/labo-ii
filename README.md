# 🐶 PetFinder - Predicción de AdoptionSpeed


🎯 Objetivo

Predecir la variable multiclase AdoptionSpeed (0 a 4) combinando modelos basados en:

    • Imágenes (ResNet50)
      
    • Texto (Description)
      
    • Variables tabulares (breed, age, fee, etc.)


🔍 Estructura del Proyecto

1. Entrenamiento de modelo de imágenes (05_Resnet50_1_train.ipynb)
    • Se entrenó una ResNet50 con transfer learning sobre las imágenes de mascotas.
      
    • Se utilizó torchvision con DataLoader para cargar y transformar las imágenes.
      
    • Se guardaron las predicciones (resnet_preds_final.csv) y se agregaron al dataset principal como variable resnet_pred.

2. Modelo de texto (predictions_desc.csv)
    • Se utilizó un modelo NLP para procesar las descripciones (Description) de cada mascota.
      
    • Se agregaron las columnas Pred_text y Prob_text al dataset final.

3. Integración de datos (modelos.ipynb)
    • Se combinaron las predicciones de imagen y texto con el dataset tabular original (df_refinado.csv).
      
    • Se construyó el dataset final con las features enriquecidas.

4. Modelos tabulares + Optuna
    • Se entrenaron tres modelos con Optuna para encontrar los mejores hiperparámetros:
      
    1. RandomForestClassifier
       
    2. XGBClassifier
       
    3. LGBMClassifier
      
    • Cada modelo fue evaluado con Cohen’s Kappa sobre un conjunto de validación del 20%.

5. Probabilidades y Ensemble
    • Para cada modelo se extrajeron las probabilidades por clase (predict_proba) en validación y test.
      
    • Se construyó un ensemble por promedio de probabilidades, seleccionando la clase con mayor score.
      
    • El ensemble final mostró mejor rendimiento que los modelos individuales.
      
