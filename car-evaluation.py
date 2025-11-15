import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Cargar el data set
print("Cargando el dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

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

print("Datos listos...")
print("Iniciando GridSearchCV para encontrar los mejores hiperparametros...")

param_grid = {'min_samples_split': [6,10,12,15],'max_depth':[10,15,20,None]}

grid = GridSearchCV(DecisionTreeClassifier(random_state=14142135), param_grid, cv=5, scoring='accuracy')

grid.fit(X,y)

print("\nMejores parametros ", grid.best_params_)
print("Mejor puntaje ", grid.best_score_)






