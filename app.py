import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.uic import loadUi


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ventana_principal.ui", self)

        self.boton_buscar.clicked.connect(self.buscar_archivo)
        self.boton_ejecutar.clicked.connect(self.iniciar_analisis)
        self.boton_ejecutar.setEnabled(False)

    def buscar_archivo(self):
        #Abrir dialogo para seleccionar un archivo .csv o .data
        ruta_archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar Dataset", "", "Archivos de Datos (*.data *.csv)")
        if ruta_archivo:
            self.linea_ruta_archivo.setText(ruta_archivo)
            self.boton_ejecutar.setEnabled(True)
            self.texto_resultados.clear()

    def iniciar_analisis(self):
        #Ejecuta todo el proceso
        ruta_archivo = self.linea_ruta_archivo.text()
        if not ruta_archivo:
            self.texto_resultados.setText("Por favor, selecciona un archivo primero.")
            return

        # Deshabilitamos botones cuando esta trabajando
        self.boton_ejecutar.setEnabled(False)
        self.boton_buscar.setEnabled(False)
        self.texto_resultados.setText("Iniciando análisis... La ventana puede congelarse por unos segundos.")
        QApplication.processEvents() # Forzamos a que se actualice el texto antes de congelarse

        try:
            self.texto_resultados.append("\nCargando el dataset...")
            nombres_columnas = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
            df = pd.read_csv(ruta_archivo, names=nombres_columnas)
            
            y = df['class']
            X = df.drop('class', axis=1)
            X = pd.get_dummies(X)

            X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=14142135)

            self.texto_resultados.append("Datos listos.\nIniciando GridSearchCV...")
            QApplication.processEvents()

            grilla_parametros = {
                'max_depth': [3, 5, 8, 15, 20],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 5, 10, 20]
            }
            grid = GridSearchCV(DecisionTreeClassifier(random_state=14142135), grilla_parametros, cv=5, scoring='accuracy')
            grid.fit(X_entrenamiento, y_entrenamiento)

            mejor_modelo = grid.best_estimator_
            y_prediccion = mejor_modelo.predict(X_prueba)
            precision_prueba = accuracy_score(y_prueba, y_prediccion)

            # Mostrar resultados
            self.texto_resultados.append("\n--- ANÁLISIS COMPLETADO ---")
            self.texto_resultados.append(f"\nMejores hiperparámetros encontrados:\n{grid.best_params_}")
            self.texto_resultados.append(f"\nMejor puntaje de validación cruzada (accuracy): {grid.best_score_:.4f}")
            self.texto_resultados.append(f"\nAccuracy del modelo en el conjunto de prueba (no visto): {precision_prueba:.4f}")
            self.texto_resultados.append("\n\nMostrando el árbol de decisión...")

            # Mostrar el gráfico
            plt.figure(figsize=(20, 10))
            plot_tree(mejor_modelo, feature_names=X.columns.tolist(), class_names=mejor_modelo.classes_, 
                      filled=True, rounded=True, max_depth=5, fontsize=5)
            plt.title("Árbol de decisión - Primeros 5 niveles")
            plt.show()

        except Exception as e:
            self.texto_resultados.append(f"\n\nERROR: Ocurrió un error durante el análisis:\n{e}")
        finally:
            # Volvemos a habilitar los botones al finalizar
            self.boton_ejecutar.setEnabled(True)
            self.boton_buscar.setEnabled(True)
            self.texto_resultados.append("\nProceso finalizado.")


if __name__ == "__main__":
    aplicacion = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(aplicacion.exec())
