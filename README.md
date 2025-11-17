# 🚗 Clasificador de Evaluación de Autos

Este proyecto utiliza un modelo de Árbol de Decisión (`DecisionTreeClassifier`) para predecir la aceptabilidad de un auto basándose en sus atributos. El script carga el dataset "Car Evaluation" desde el repositorio de UCI, lo procesa, encuentra los mejores hiperparámetros con `GridSearchCV` y **evalúa el modelo final en un conjunto de datos de prueba para medir su rendimiento real.**

El objetivo es predecir la columna `class` (clase), que tiene cuatro valores posibles:
* `unacc` (No Aceptable)
* `acc` (Aceptable)
* `good` (Bueno)
* `vgood` (Muy Bueno)

---

## 📊 Dataset

* **Fuente:** Repositorio de Machine Learning de UCI
* **URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data`
* **Características (Features):**
    * `buying`: Precio de compra (vhigh, high, med, low)
    * `maint`: Costo de mantenimiento (vhigh, high, med, low)
    * `doors`: Número de puertas (2, 3, 4, 5more)
    * `persons`: Capacidad de personas (2, 4, more)
    * `lug_boot`: Tamaño del baúl (small, med, big)
    * `safety`: Seguridad (low, med, high)
* **Objetivo (Target):**
    * `class`: Aceptabilidad del auto

---

## 🤖 Flujo de Trabajo del Script

1.  **Carga de Datos:** El script carga el dataset `car.data` usando `pandas` y asigna los nombres de columna correctos.
2.  **Preprocesamiento:** Todas las características categóricas (texto) se convierten a un formato numérico usando `pd.get_dummies()` (One-Hot Encoding) para que el modelo pueda procesarlas.
3.  **División de Datos:** El dataset se divide en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%) usando `train_test_split`. El modelo solo "aprenderá" de los datos de entrenamiento.
4.  **Búsqueda de Hiperparámetros:** Se utiliza `GridSearchCV` para probar sistemáticamente múltiples combinaciones de hiperparámetros (como `max_depth`, `min_samples_split`, etc.) sobre el conjunto de entrenamiento mediante validación cruzada (`cv=5`).
5.  **Entrenamiento del Mejor Modelo:** Una vez que `GridSearchCV` encuentra la mejor combinación de hiperparámetros, re-entrena automáticamente un modelo final con esa configuración usando **todo el conjunto de entrenamiento**.
6.  **Evaluación Final:** El rendimiento del modelo final (`best_estimator_`) se mide en el **conjunto de prueba**, que contiene datos que el modelo nunca ha visto. Se calcula el `accuracy` y se genera un `classification_report` detallado para obtener una medida imparcial de su capacidad de generalización.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**
* **Pandas:** Para la carga y manipulación de datos.
* **Scikit-learn (sklearn):** Para el modelo (`DecisionTreeClassifier`), la división de datos (`train_test_split`), la optimización (`GridSearchCV`) y las métricas de evaluación.

---

## 🚀 Cómo Ejecutar

1.  Asegúrate de tener `pandas` y `scikit-learn` instalados:
    ```bash
    pip install pandas scikit-learn
    ```
2.  Guarda el código como un archivo (ej. `car-evaluation.py`).
3.  Ejecútalo desde tu terminal:
    ```bash
    python car-evaluation.py
    ```

---

## 📈 Resultados

El script imprimirá los mejores hiperparámetros, el puntaje de la validación cruzada y, lo más importante, el rendimiento final del modelo en el conjunto de prueba.

**Salida de ejemplo:**

Cargando el dataset...
Datos listos...
Iniciando GridSearchCV para encontrar los mejores hiperparametros...       

Mejores hiperparámetros encontrados: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2}
Mejor puntaje de validación cruzada (accuracy): 0.9652697117145397

Evaluando el mejor modelo en el conjunto de prueba...
Accuracy del modelo en el conjunto de prueba (no visto): 0.9682080924855492
