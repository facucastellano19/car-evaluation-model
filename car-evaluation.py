import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#Cargar el data set
print("Cargando el dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

# Definir los nombres de las columnas
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

