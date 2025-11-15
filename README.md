# 🚗 Clasificador de Evaluación de Autos

Este proyecto utiliza un modelo de Árbol de Decisión (`DecisionTreeClassifier`) para predecir la aceptabilidad de un auto basándose en sus atributos. El script carga el dataset "Car Evaluation" directamente desde el repositorio de UCI, lo procesa y **utiliza `GridSearchCV` para encontrar y entrenar el modelo con los mejores hiperparámetros.**

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

1.  **Carga de Datos:** El script carga el dataset `car.data` directamente desde la URL de UCI usando `pandas`.
2.  **Asignación de Nombres:** Se asignan los nombres de columna correctos (ya que el archivo `.data` no los incluye).
3.  **Preprocesamiento (One-Hot Encoding):** Este es el paso clave. Dado que todas las *features* son categóricas (texto), se utiliza `pd.get_dummies()` para convertirlas en un formato numérico (0s y 1s) que el modelo pueda entender.
4.  **Búsqueda de Hiperparámetros (GridSearchCV):** En lugar de una simple división de prueba, el script utiliza `GridSearchCV` con Validación Cruzada (`cv=5`). Esto prueba sistemáticamente múltiples combinaciones de hiperparámetros (como `max_depth` y `min_samples_split`) para encontrar la mejor configuración.
5.  **Entrenamiento:** Se entrena el objeto `GridSearchCV` con *todos* los datos. Este se encarga de probar todas las combinaciones y seleccionar el mejor modelo.
6.  **Evaluación:** El script reporta los mejores parámetros encontrados (`best_params_`) y el *score* de *accuracy* promedio (`best_score_`) obtenido de la validación cruzada.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**
* **Pandas:** Para la carga y manipulación de datos (incluyendo `get_dummies`).
* **Scikit-learn (sklearn):** Para el modelo (`DecisionTreeClassifier`) y la optimización de hiperparámetros (`GridSearchCV`).

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

El script imprimirá en la consola los mejores hiperparámetros encontrados y el *score* promedio (confiable) de la validación cruzada.

**Salida de ejemplo (después del refinamiento):**
Cargando el dataset... 
Datos listos... 
Iniciando GridSearchCV para encontrar los mejores hiperparámetros...

Mejores parametros {'max_depth': 10, 'min_samples_split': 6} 
Mejor puntaje 0.7587132445338025
