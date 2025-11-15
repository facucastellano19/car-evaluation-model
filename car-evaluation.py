import pandas as pd
from sklearn.model_selection import train_test_split
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

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14142135)

print("Datos listos...")
print("Entrenando el modelo...")
model = DecisionTreeClassifier(random_state=14142135)
model.fit(X_train, y_train)






