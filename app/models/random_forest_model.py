from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
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

class RandomForestModel:
    """
    Clase que encapsula el modelo de Random Forest para clasificación de Iris
    """
    
    def __init__(self, random_state=42):
        """Inicializar el modelo con configuraciones por defecto"""
        try:
            app_logger.info("Inicializando modelo Random Forest")
            self.random_state = random_state
            self.best_model = None
            self.feature_names = None
            self.target_names = None
            self.metrics = {}
            self.model_data = {}
        except Exception as e:
            app_logger.error(f"Error al inicializar modelo Random Forest: {str(e)}", sys.exc_info())
            raise
        
    def create_pipeline(self):
        """
        Crear pipeline con preprocesamiento y modelo
        
        Returns:
            Pipeline de scikit-learn
            
        Raises:
            ModelTrainingError: Si ocurre un error al crear el pipeline
        """
        try:
            app_logger.info("Creando pipeline para Random Forest")
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    oob_score=True
                ))
            ])
            app_logger.info("Pipeline de Random Forest creado con éxito")
            return self.pipeline
        except Exception as e:
            app_logger.error(f"Error al crear pipeline de Random Forest: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Random Forest", f"Error al crear pipeline: {str(e)}")
    
    def optimize_hyperparameters(self, X_train, y_train, cv=5):
        """
        Optimizar hiperparámetros usando GridSearchCV
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            cv: Número de folds para validación cruzada
            
        Returns:
            Mejor modelo encontrado
            
        Raises:
            ModelTrainingError: Si ocurre un error durante la optimización
        """
        try:
            app_logger.info(f"Optimizando hiperparámetros de Random Forest con cv={cv}")
            
            # Verificar que el pipeline exista
            if not hasattr(self, 'pipeline'):
                app_logger.warning("Pipeline no creado, creando automáticamente")
                self.create_pipeline()
                
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
            
            total_combinations = (len(param_grid['classifier__n_estimators']) * 
                                len(param_grid['classifier__max_depth']) * 
                                len(param_grid['classifier__min_samples_split']) * 
                                len(param_grid['classifier__min_samples_leaf']))
            
            app_logger.info(f"Iniciando búsqueda con {total_combinations} combinaciones de hiperparámetros")
            
            self.grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=cv, 
                scoring='accuracy', n_jobs=-1
            )
            self.grid_search.fit(X_train, y_train)
            self.best_model = self.grid_search.best_estimator_
            
            app_logger.info(f"Optimización de hiperparámetros completada")
            app_logger.info(f"Mejores hiperparámetros: {self.grid_search.best_params_}")
            app_logger.info(f"Mejor puntuación de validación: {self.grid_search.best_score_:.4f}")
            
            return self.best_model
        except Exception as e:
            app_logger.error(f"Error en optimización de hiperparámetros de Random Forest: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Random Forest", f"Error en optimización de hiperparámetros: {str(e)}")
    
    def calculate_metrics(self, y_test, y_pred, y_proba, target_names):
        """
        Calcular métricas de rendimiento del modelo
        
        Args:
            y_test: Etiquetas reales
            y_pred: Predicciones
            y_proba: Probabilidades de predicción
            target_names: Nombres de las clases
            
        Returns:
            Diccionario con métricas
        """
        try:
            app_logger.info("Calculando métricas de rendimiento de Random Forest")
            metrics = {}
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='macro')
            metrics['recall'] = recall_score(y_test, y_pred, average='macro')
            metrics['f1'] = f1_score(y_test, y_pred, average='macro')
            metrics['conf_matrix'] = confusion_matrix(y_test, y_pred)
            metrics['classification_report'] = classification_report(
                y_test, y_pred, target_names=target_names, output_dict=True
            )
            
            app_logger.info(f"Métricas calculadas: Accuracy={metrics['accuracy']:.4f}, "
                          f"Precision={metrics['precision']:.4f}, "
                          f"Recall={metrics['recall']:.4f}, "
                          f"F1={metrics['f1']:.4f}")
            
            return metrics
        except Exception as e:
            app_logger.error(f"Error al calcular métricas de Random Forest: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Random Forest", f"Error al calcular métricas: {str(e)}")
    
    def get_confusion_matrix_image(self, conf_matrix, target_names):
        """
        Generar imagen de la matriz de confusión
        
        Args:
            conf_matrix: Matriz de confusión
            target_names: Nombres de las clases
            
        Returns:
            String base64 con la imagen
            
        Raises:
            VisualizationError: Si ocurre un error al generar la visualización
        """
        try:
            app_logger.info("Generando visualización de matriz de confusión para Random Forest")
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=target_names, 
                yticklabels=target_names
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
            app_logger.error(f"Error al generar matriz de confusión para Random Forest: {str(e)}", sys.exc_info())
            raise VisualizationError(f"Error al generar matriz de confusión: {str(e)}")
    
    def get_roc_curves_image(self, y_test, y_proba, target_names):
        """
        Generar imagen de las curvas ROC
        
        Args:
            y_test: Etiquetas reales
            y_proba: Probabilidades de predicción
            target_names: Nombres de las clases
            
        Returns:
            String base64 con la imagen
            
        Raises:
            VisualizationError: Si ocurre un error al generar la visualización
        """
        try:
            app_logger.info("Generando visualización de curvas ROC para Random Forest")
            plt.figure(figsize=(8, 6))
            
            for i, class_name in enumerate(target_names):
                # Calcular ROC para cada clase (one-vs-rest)
                y_test_binary = (y_test == i).astype(int)
                y_score = y_proba[:, i]
                
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
            app_logger.error(f"Error al generar curvas ROC para Random Forest: {str(e)}", sys.exc_info())
            raise VisualizationError(f"Error al generar curvas ROC: {str(e)}")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_names, target_names):
        """
        Entrenar y evaluar el modelo
        
        Args:
            X_train: Características de entrenamiento
            X_test: Características de prueba
            y_train: Etiquetas de entrenamiento
            y_test: Etiquetas de prueba
            feature_names: Nombres de las características
            target_names: Nombres de las clases
            
        Returns:
            Diccionario con datos del modelo
            
        Raises:
            ModelTrainingError: Si ocurre un error durante el entrenamiento o evaluación
        """
        try:
            app_logger.info("Iniciando entrenamiento y evaluación de Random Forest")
            
            # Guardar nombres de características y clases
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.feature_names = feature_names
            self.target_names = target_names
            
            app_logger.info(f"Datos cargados: {X_train.shape[0]} muestras de entrenamiento, "
                          f"{X_test.shape[0]} muestras de prueba, "
                          f"{X_train.shape[1]} características")
            
            # Crear y optimizar el modelo
            app_logger.info("Creando pipeline y optimizando hiperparámetros")
            self.create_pipeline()
            self.optimize_hyperparameters(X_train, y_train)
            
            # Evaluar el modelo
            app_logger.info("Evaluando modelo en conjunto de prueba")
            y_pred = self.best_model.predict(X_test)
            y_proba = self.best_model.predict_proba(X_test)
            
            # Calcular métricas
            app_logger.info("Calculando métricas de rendimiento")
            self.metrics = self.calculate_metrics(y_test, y_pred, y_proba, target_names)
            
            # Validación cruzada para evaluar la estabilidad
            app_logger.info("Realizando validación cruzada para estimar estabilidad")
            self.metrics['cv_results'] = cross_val_score(
                self.best_model, np.vstack((X_train, X_test)), 
                np.hstack((y_train, y_test)), cv=5
            )
            
            # Generar visualizaciones
            app_logger.info("Generando visualizaciones")
            self.metrics['confusion_matrix_img'] = self.get_confusion_matrix_image(
                self.metrics['conf_matrix'], target_names
            )
            self.metrics['roc_curves_img'] = self.get_roc_curves_image(
                y_test, y_proba, target_names
            )
            
            # Generar visualizaciones específicas de Random Forest
            app_logger.info("Generando visualizaciones específicas de Random Forest")
            self.generate_rf_visualizations()
            
            # Estructurar datos para la aplicación
            self.model_data = {
                'best_model': self.best_model,
                'target_names': self.target_names,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'best_params': self.grid_search.best_params_,
            }
            
            # Registrar resultados en el log
            app_logger.log_model_results("random_forest", self.metrics, self.grid_search.best_params_)
            
            app_logger.info(f"Entrenamiento y evaluación de Random Forest completados con éxito")
            app_logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}, F1-Score: {self.metrics['f1']:.4f}")
            
            return self.model_data
        except Exception as e:
            app_logger.error(f"Error en train_and_evaluate de Random Forest: {str(e)}", sys.exc_info())
            raise ModelTrainingError("Random Forest", f"Error en entrenamiento y evaluación: {str(e)}")
    
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
            app_logger.info(f"Realizando predicción con modelo Random Forest: {features}")
            
            # Verificar que el modelo esté entrenado
            if self.best_model is None:
                error_msg = "El modelo no ha sido entrenado, entrene el modelo primero"
                app_logger.error(error_msg)
                raise ModelNotTrainedError("Random Forest", error_msg)
            
            # Verificar dimensiones de entrada
            if features.shape[1] != len(self.feature_names):
                error_msg = f"Número incorrecto de características: esperado {len(self.feature_names)}, recibido {features.shape[1]}"
                app_logger.error(error_msg)
                raise PredictionError("Random Forest", error_msg)
                
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
            app_logger.error(f"Error al realizar predicción con Random Forest: {str(e)}", sys.exc_info())
            raise PredictionError("Random Forest", f"Error al realizar predicción: {str(e)}")
    
    def generate_rf_visualizations(self):
        """
        Generar visualizaciones específicas de Random Forest
        
        Raises:
            VisualizationError: Si ocurre un error al generar las visualizaciones
        """
        try:
            app_logger.info("Generando visualizaciones específicas de Random Forest")
            
            # Verificar que el modelo esté entrenado
            if self.best_model is None:
                error_msg = "El modelo no ha sido entrenado, no se pueden generar visualizaciones"
                app_logger.error(error_msg)
                raise ModelNotTrainedError("Random Forest", error_msg)
                
            rf_model = self.best_model.named_steps['classifier']
            
            # Importancia de características
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Importancia de Características')
            plt.bar(range(len(importances)), 
                    importances[indices],
                    align='center')
            plt.xticks(range(len(importances)), 
                      [self.feature_names[i] for i in indices], 
                      rotation=90)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            self.metrics['feature_importance_img'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            app_logger.info("Visualización de importancia de características generada correctamente")
            
            # Visualización del OOB error (Out-of-bag)
            if hasattr(rf_model, 'oob_score_'):
                oob_score = rf_model.oob_score_
                self.metrics['oob_score'] = oob_score
                app_logger.info(f"OOB Score: {oob_score:.4f}")
                
            # Decisión boundary con las dos características más importantes
            app_logger.info("Generando visualización de superficie de decisión")
            top_features = [indices[0], indices[1]]
            
            X_reduced = self.X_test[:, top_features]
            y_test = self.y_test
            
            # Crear malla para visualizar la frontera de decisión
            h = .02  # Tamaño del paso en la malla
            x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
            y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Crear un nuevo clasificador con solo esas características
            rf_simple = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            )
            rf_simple.fit(self.X_train[:, top_features], self.y_train)
            
            # Predecir
            Z = rf_simple.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            plt.figure(figsize=(10, 8))
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
            
            # Graficar los puntos de cada clase
            for i, color in zip(range(len(self.target_names)), ['red', 'green', 'blue']):
                idx = np.where(y_test == i)
                plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], 
                          c=color, label=self.target_names[i],
                          edgecolor='black', s=50)
            
            plt.xlabel(self.feature_names[top_features[0]])
            plt.ylabel(self.feature_names[top_features[1]])
            plt.title('Superficie de Decisión con las 2 Características más Importantes')
            plt.legend()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            self.metrics['decision_boundary_img'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            app_logger.info("Visualización de superficie de decisión generada correctamente")
            
        except ModelNotTrainedError:
            # Re-lanzar esta excepción específica
            raise
        except Exception as e:
            app_logger.error(f"Error al generar visualizaciones específicas de Random Forest: {str(e)}", sys.exc_info())
            raise VisualizationError(f"Error al generar visualizaciones específicas: {str(e)}") 