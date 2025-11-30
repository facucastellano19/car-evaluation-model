import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score, classification_report
import ssl
import matplotlib.pyplot as plt


#Cargar el data set
print("Cargando el dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

# Esto crea un contexto HTTPS no verificado y lo establece como predeterminado.
ssl._create_default_https_context = ssl._create_unverified_context

# Definir los nombres de las columnas
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

try:
    df = pd.read_csv(url, names=column_names)
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    exit()
    
    
# Preparacion de los datos:
# Separar las Features de los Targets
y = df['class']
X = df.drop('class', axis=1)

# Convertimos variables categoricas a numericas
X = pd.get_dummies(X)

#print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14142135)

print("Datos listos...")
print("Iniciando GridSearchCV para encontrar los mejores hiperparametros...")

param_grid = {
    'max_depth': [3, 5, 8, 15, 20],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10,20]
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=14142135), param_grid, cv=5, scoring='accuracy')

# Entrenamos el GridSearchCV
grid.fit(X_train, y_train)

print("\nMejores hiperparámetros encontrados:", grid.best_params_)
print(f"Mejor puntaje de validación cruzada (accuracy): {grid.best_score_}")

print("\nEvaluando el mejor modelo en el conjunto de prueba...")
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo en el conjunto de prueba (no visto): {test_accuracy}")


plt.figure(figsize=(20, 10))
plot_tree(best_model, 
          feature_names=X.columns.tolist(), 
          class_names=best_model.classes_, 
          filled=True,
          rounded=True,
          max_depth=5,
          fontsize=5)
plt.title("Arbol de decision - Primeros 5 niveles")
plt.show()