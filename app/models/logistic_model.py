from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import sys

# Importar utilidades para manejo de excepciones y logging
from app.utils.logger import app_logger
from app.utils.exceptions import (
    ModelTrainingError, 
    PredictionError, 
    DataLoadError,
    VisualizationError,
    ModelNotTrainedError
)

class IrisModel:
    """
    Clase que encapsula la lógica del modelo de clasificación de Iris
    incluyendo entrenamiento, evaluación y visualización
    """
    
    def __init__(self, random_state=42):
        """Inicializar el modelo con configuraciones por defecto"""
        try:
            app_logger.info("Inicializando modelo de Regresión Logística")
            self.random_state = random_state
            self.best_model = None
            self.feature_names = None
            self.target_names = None
            self.metrics = {}
            self.model_data = {}
        except Exception as e:
            app_logger.error(f"Error al inicializar modelo de Regresión Logística: {str(e)}", sys.exc_info())
            raise
        
    def load_data(self):
        """
        Cargar el conjunto de datos de Iris
        
        Returns:
            Tupla con (X, y)
            
        Raises:
            DataLoadError: Si ocurre un error al cargar los datos
        """
        try:
            app_logger.info("Cargando conjunto de datos Iris para modelo de Regresión Logística")
            
            # Cargar el conjunto de datos
            iris = load_iris()
            self.X = iris.data
            self.y = iris.target
            self.feature_names = iris.feature_names
            self.target_names = iris.target_names
            
            app_logger.info(f"Datos cargados: {self.X.shape[0]} muestras, {self.X.shape[1]} características")
            return self.X, self.y
            
        except Exception as e:
            app_logger.error(f"Error al cargar datos para modelo de Regresión Logística: {str(e)}", sys.exc_info())
            raise DataLoadError(f"Error al cargar conjunto de datos Iris: {str(e)}")
        
    def split_data(self, test_size=0.3):
        """
        Dividir los datos en conjuntos de entrenamiento y prueba
        
        Args:
            test_size: Proporción del conjunto de prueba
            
        Returns:
            Tupla con (X_train, X_test, y_train, y_test)
            
        Raises:
            DataLoadError: Si ocurre un error al dividir los datos
        """
        try:
            app_logger.info(f"Dividiendo datos con test_size={test_size}")
            
            # Verificar que los datos estén cargados
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                app_logger.warning("Datos no cargados, cargando automáticamente")
                self.load_data()
                
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=self.random_state
            )
            
            app_logger.info(f"Datos divididos: {self.X_train.shape[0]} muestras de entrenamiento, {self.X_test.shape[0]} muestras de prueba")
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            app_logger.error(f"Error al dividir datos para Regresión Logística: {str(e)}", sys.exc_info())
            raise DataLoadError(f"Error al dividir los datos: {str(e)}")
        
    def create_pipeline(self):
        """
        Crear un pipeline de preprocesamiento y modelo
        
        Returns:
            Pipeline de scikit-learn
        """
        try:
            app_logger.info("Creando pipeline para modelo de Regresión Logística")
            
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])
            
            app_logger.info("Pipeline creado correctamente")
            return self.pipeline
            
        except Exception as e:
            app_logger.error(f"Error al crear pipeline de Regresión Logística: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Regresión Logística", f"Error al crear pipeline: {str(e)}")
        
    def optimize_hyperparameters(self, cv=5):
        """
        Optimizar hiperparámetros usando GridSearchCV
        
        Args:
            cv: Número de folds para validación cruzada
            
        Returns:
            Mejor modelo encontrado
            
        Raises:
            ModelTrainingError: Si ocurre un error durante la optimización
        """
        try:
            app_logger.info(f"Optimizando hiperparámetros con cv={cv}")
            
            # Verificar que los datos estén cargados y divididos
            if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
                app_logger.warning("Datos no divididos, dividiendo automáticamente")
                self.split_data()
                
            # Verificar que el pipeline esté creado
            if not hasattr(self, 'pipeline'):
                app_logger.warning("Pipeline no creado, creando automáticamente")
                self.create_pipeline()
                
            param_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Parámetro de regularización
                'classifier__penalty': ['l1', 'l2'],  # Tipo de regularización
                'classifier__solver': ['liblinear']  # Solver que funciona bien con L1 y L2
            }
            
            app_logger.info(f"Iniciando búsqueda de hiperparámetros con {len(param_grid['classifier__C']) * len(param_grid['classifier__penalty'])} combinaciones")
            self.grid_search = GridSearchCV(self.pipeline, param_grid, cv=cv, scoring='accuracy')
            self.grid_search.fit(self.X_train, self.y_train)
            self.best_model = self.grid_search.best_estimator_
            
            app_logger.info(f"Mejores hiperparámetros encontrados: {self.grid_search.best_params_}")
            app_logger.info(f"Mejor puntuación de validación: {self.grid_search.best_score_:.4f}")
            
            return self.best_model
            
        except Exception as e:
            app_logger.error(f"Error en optimización de hiperparámetros de Regresión Logística: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Regresión Logística", f"Error en optimización de hiperparámetros: {str(e)}")
        
    def train_baseline_model(self):
        """
        Entrenar un modelo base simple para comparación
        
        Returns:
            Precisión del modelo baseline
        """
        try:
            app_logger.info("Entrenando modelo baseline para comparación")
            
            # Verificar que los datos estén cargados y divididos
            if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
                app_logger.warning("Datos no divididos, dividiendo automáticamente")
                self.split_data()
                
            self.dummy_clf = DummyClassifier(strategy='most_frequent')
            self.dummy_clf.fit(self.X_train, self.y_train)
            dummy_pred = self.dummy_clf.predict(self.X_test)
            self.dummy_accuracy = accuracy_score(self.y_test, dummy_pred)
            
            app_logger.info(f"Precisión del modelo baseline: {self.dummy_accuracy:.4f}")
            return self.dummy_accuracy
            
        except Exception as e:
            app_logger.error(f"Error al entrenar modelo baseline: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Baseline", f"Error al entrenar modelo baseline: {str(e)}")
        
    def evaluate_model(self):
        """
        Evaluar el rendimiento del modelo usando múltiples métricas
        
        Returns:
            Diccionario con métricas de rendimiento
            
        Raises:
            ModelNotTrainedError: Si el modelo no ha sido entrenado
        """
        try:
            app_logger.info("Evaluando rendimiento del modelo de Regresión Logística")
            
            # Verificar que el modelo esté entrenado
            if self.best_model is None:
                error_msg = "El modelo no ha sido entrenado, ejecute optimize_hyperparameters primero"
                app_logger.error(error_msg)
                raise ModelNotTrainedError("Regresión Logística", error_msg)
                
            # Verificar que los datos de prueba estén disponibles
            if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
                app_logger.warning("Datos de prueba no disponibles, dividiendo datos automáticamente")
                self.split_data()
                
            self.y_pred = self.best_model.predict(self.X_test)
            self.y_proba = self.best_model.predict_proba(self.X_test)
            
            # Calcular métricas principales
            self.metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
            self.metrics['precision'] = precision_score(self.y_test, self.y_pred, average='macro')
            self.metrics['recall'] = recall_score(self.y_test, self.y_pred, average='macro')
            self.metrics['f1'] = f1_score(self.y_test, self.y_pred, average='macro')
            self.metrics['conf_matrix'] = confusion_matrix(self.y_test, self.y_pred)
            self.metrics['classification_report'] = classification_report(
                self.y_test, self.y_pred, target_names=self.target_names, output_dict=True
            )
            
            # Validación cruzada para evaluar la estabilidad
            self.metrics['cv_results'] = cross_val_score(self.best_model, self.X, self.y, cv=5)
            
            app_logger.info(f"Métricas calculadas: Accuracy={self.metrics['accuracy']:.4f}, "
                         f"Precision={self.metrics['precision']:.4f}, "
                         f"Recall={self.metrics['recall']:.4f}, "
                         f"F1={self.metrics['f1']:.4f}")
            
            # Registrar resultados en el log
            app_logger.log_model_results("logistic", self.metrics, self.grid_search.best_params_)
            
            return self.metrics
            
        except ModelNotTrainedError:
            # Re-lanzar esta excepción específica
            raise
        except Exception as e:
            app_logger.error(f"Error al evaluar modelo de Regresión Logística: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Regresión Logística", f"Error al evaluar modelo: {str(e)}")
        
    def extract_feature_importance(self):
        """
        Extraer y escalar la importancia de las características
        
        Returns:
            Diccionario con coeficientes escalados
            
        Raises:
            ModelNotTrainedError: Si el modelo no ha sido entrenado
        """
        try:
            app_logger.info("Extrayendo importancia de características para Regresión Logística")
            
            # Verificar que el modelo esté entrenado
            if self.best_model is None:
                error_msg = "El modelo no ha sido entrenado, ejecute optimize_hyperparameters primero"
                app_logger.error(error_msg)
                raise ModelNotTrainedError("Regresión Logística", error_msg)
                
            scaler = self.best_model.named_steps['scaler']
            classifier = self.best_model.named_steps['classifier']
            coefficients = classifier.coef_
            scaled_coefficients = {}
            
            for i, class_name in enumerate(self.target_names):
                # Para cada clase, obtener los coeficientes y normalizarlos
                class_coef = coefficients[i]
                abs_coef = np.abs(class_coef)
                scaled_coef = abs_coef / np.sum(abs_coef) * 100  # Como porcentaje
                scaled_coefficients[class_name] = {
                    self.feature_names[j]: scaled_coef[j] for j in range(len(self.feature_names))
                }
                
            self.scaled_coefficients = scaled_coefficients
            
            app_logger.info("Importancia de características extraída correctamente")
            return self.scaled_coefficients
            
        except ModelNotTrainedError:
            # Re-lanzar esta excepción específica
            raise
        except Exception as e:
            app_logger.error(f"Error al extraer importancia de características: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Regresión Logística", f"Error al extraer importancia de características: {str(e)}")
        
    def get_confusion_matrix_image(self):
        """
        Generar imagen de la matriz de confusión
        
        Returns:
            String base64 con la imagen
            
        Raises:
            VisualizationError: Si ocurre un error al generar la visualización
        """
        try:
            app_logger.info("Generando visualización de matriz de confusión")
            
            # Verificar que las métricas estén calculadas
            if 'conf_matrix' not in self.metrics:
                app_logger.warning("Métricas no calculadas, evaluando modelo automáticamente")
                self.evaluate_model()
                
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                self.metrics['conf_matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.target_names, 
                yticklabels=self.target_names
            )
            plt.title('Matriz de Confusión')
            plt.ylabel('Etiqueta Real')
            plt.xlabel('Etiqueta Predicha')
            
            # Convertir el gráfico a una imagen base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            app_logger.info("Visualización de matriz de confusión generada correctamente")
            return img_str
            
        except Exception as e:
            app_logger.error(f"Error al generar matriz de confusión: {str(e)}", sys.exc_info())
            raise VisualizationError(f"Error al generar matriz de confusión: {str(e)}")
    
    def get_roc_curves_image(self):
        """
        Generar imagen de las curvas ROC
        
        Returns:
            String base64 con la imagen
            
        Raises:
            VisualizationError: Si ocurre un error al generar la visualización
        """
        try:
            app_logger.info("Generando visualización de curvas ROC")
            
            # Verificar que las predicciones estén disponibles
            if not hasattr(self, 'y_proba') or not hasattr(self, 'y_test'):
                app_logger.warning("Predicciones no disponibles, evaluando modelo automáticamente")
                self.evaluate_model()
                
            plt.figure(figsize=(8, 6))
            
            for i, class_name in enumerate(self.target_names):
                # Calcular ROC para cada clase (one-vs-rest)
                y_test_binary = (self.y_test == i).astype(int)
                y_score = self.y_proba[:, i]
                
                fpr, tpr, _ = roc_curve(y_test_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curvas ROC para Multiclase (One-vs-Rest)')
            plt.legend(loc="lower right")
            
            # Convertir el gráfico a una imagen base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            app_logger.info("Visualización de curvas ROC generada correctamente")
            return img_str
            
        except Exception as e:
            app_logger.error(f"Error al generar curvas ROC: {str(e)}", sys.exc_info())
            raise VisualizationError(f"Error al generar curvas ROC: {str(e)}")
    
    def get_coefficients_image(self):
        """
        Generar imagen de los coeficientes del modelo
        
        Returns:
            String base64 con la imagen
            
        Raises:
            VisualizationError: Si ocurre un error al generar la visualización
        """
        try:
            app_logger.info("Generando visualización de coeficientes")
            
            # Verificar que los coeficientes estén calculados
            if not hasattr(self, 'scaled_coefficients'):
                app_logger.warning("Coeficientes no calculados, extrayendo automáticamente")
                self.extract_feature_importance()
                
            coef_df = pd.DataFrame(self.scaled_coefficients).T
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(coef_df, annot=True, cmap='YlGnBu', fmt='.1f')
            plt.title('Importancia Relativa de Características (%)')
            plt.ylabel('Clase')
            plt.xlabel('Característica')
            
            # Convertir el gráfico a una imagen base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            app_logger.info("Visualización de coeficientes generada correctamente")
            return img_str
            
        except Exception as e:
            app_logger.error(f"Error al generar visualización de coeficientes: {str(e)}", sys.exc_info())
            raise VisualizationError(f"Error al generar visualización de coeficientes: {str(e)}")
    
    def get_decision_boundary_image(self):
        """
        Generar imagen de la superficie de decisión
        
        Returns:
            String base64 con la imagen
            
        Raises:
            VisualizationError: Si ocurre un error al generar la visualización
        """
        try:
            app_logger.info("Generando visualización de superficie de decisión")
            
            # Verificar que los datos estén disponibles
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                app_logger.warning("Datos no disponibles, cargando automáticamente")
                self.load_data()
                
            # Para la visualización, usaremos solo 2 características
            X_reduced = self.X[:, :2]  # Usar sepal length y sepal width
            
            # Entrenar un modelo con solo estas características
            X_train_reduced, X_test_reduced, y_train_reduced, _ = train_test_split(
                X_reduced, self.y, test_size=0.3, random_state=self.random_state
            )
            reduced_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=self.random_state))
            ])
            reduced_pipeline.fit(X_train_reduced, y_train_reduced)
            
            # Crear malla para visualizar la frontera de decisión
            h = .02  # tamaño del paso en la malla
            x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
            y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Predecir en cada punto de la malla
            Z = reduced_pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Crear el gráfico
            plt.figure(figsize=(10, 8))
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
            
            # Graficar los puntos de cada clase
            for i, color in zip(range(len(self.target_names)), ['red', 'green', 'blue']):
                idx = np.where(self.y == i)
                plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], c=color, label=self.target_names[i],
                           edgecolor='black', s=50)
                
            plt.xlabel(self.feature_names[0])
            plt.ylabel(self.feature_names[1])
            plt.title('Superficie de Decisión')
            plt.legend()
            
            # Convertir el gráfico a una imagen base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            app_logger.info("Visualización de superficie de decisión generada correctamente")
            return img_str
            
        except Exception as e:
            app_logger.error(f"Error al generar superficie de decisión: {str(e)}", sys.exc_info())
            raise VisualizationError(f"Error al generar superficie de decisión: {str(e)}")
    
    def predict(self, features):
        """
        Realizar predicción con nuevos datos
        
        Args:
            features: Array con características para predecir
            
        Returns:
            Diccionario con predicción y probabilidades
            
        Raises:
            ModelNotTrainedError: Si el modelo no ha sido entrenado
            PredictionError: Si ocurre un error durante la predicción
        """
        try:
            app_logger.info(f"Realizando predicción con modelo de Regresión Logística: {features}")
            
            # Verificar que el modelo esté entrenado
            if self.best_model is None:
                error_msg = "El modelo no ha sido entrenado, entrene el modelo primero"
                app_logger.error(error_msg)
                raise ModelNotTrainedError("Regresión Logística", error_msg)
                
            # Verificar dimensiones de entrada
            if features.shape[1] != len(self.feature_names):
                error_msg = f"Número incorrecto de características: esperado {len(self.feature_names)}, recibido {features.shape[1]}"
                app_logger.error(error_msg)
                raise PredictionError("Regresión Logística", error_msg)
                
            prediction_idx = self.best_model.predict(features)[0]
            class_name = self.target_names[prediction_idx]
            
            # Obtener probabilidades
            probabilities = self.best_model.predict_proba(features)[0]
            probs = {self.target_names[i]: round(float(prob) * 100, 2) 
                    for i, prob in enumerate(probabilities)}
            
            app_logger.info(f"Predicción: {class_name} con probabilidades: {probs}")
            
            return {
                'prediction': class_name,
                'probabilities': probs
            }
            
        except (ModelNotTrainedError, PredictionError):
            # Re-lanzar estas excepciones específicas
            raise
        except Exception as e:
            app_logger.error(f"Error al realizar predicción con Regresión Logística: {str(e)}", sys.exc_info())
            raise PredictionError("Regresión Logística", f"Error al realizar predicción: {str(e)}")
    
    def train_and_evaluate(self):
        """
        Método principal para entrenar y evaluar el modelo
        
        Returns:
            Diccionario con datos del modelo
            
        Raises:
            ModelTrainingError: Si ocurre un error durante el entrenamiento o evaluación
        """
        try:
            app_logger.info("Iniciando entrenamiento y evaluación del modelo de Regresión Logística")
            
            # Cargar y preparar datos
            self.load_data()
            self.split_data()
            
            # Crear pipeline y entrenar modelo
            self.create_pipeline()
            self.optimize_hyperparameters()
            
            # Entrenar modelo baseline para comparación
            baseline_acc = self.train_baseline_model()
            
            # Evaluar modelo y generar métricas
            self.evaluate_model()
            self.extract_feature_importance()
            
            # Generar visualizaciones
            app_logger.info("Generando visualizaciones")
            self.metrics['confusion_matrix_img'] = self.get_confusion_matrix_image()
            self.metrics['roc_curves_img'] = self.get_roc_curves_image()
            self.metrics['coefficients_img'] = self.get_coefficients_image()
            self.metrics['decision_boundary_img'] = self.get_decision_boundary_image()
            
            # Obtener los mejores parámetros
            best_params = self.grid_search.best_params_
            
            # Estructurar datos para la aplicación
            self.model_data = {
                'best_model': self.best_model,
                'target_names': self.target_names,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'best_params': best_params,
                'baseline_accuracy': baseline_acc
            }
            
            app_logger.info("Entrenamiento y evaluación del modelo de Regresión Logística completados con éxito")
            app_logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}, F1-Score: {self.metrics['f1']:.4f}")
            
            return self.model_data
            
        except Exception as e:
            app_logger.error(f"Error en train_and_evaluate de Regresión Logística: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Regresión Logística", f"Error en entrenamiento y evaluación: {str(e)}") 