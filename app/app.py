from flask import Flask, render_template, request, jsonify
import matplotlib
# Configurar el backend para no usar GUI
matplotlib.use('Agg')
import numpy as np
import traceback
import sys

# Importar modelos
from app.models import IrisModel, RandomForestModel

# Importar utilidades
from app.utils import (
    load_iris_data, 
    compare_models, 
    app_logger,
    IrisClassifierException,
    ModelTrainingError,
    PredictionError,
    InvalidInputError
)

app = Flask(__name__)
app.static_folder = 'static'
app.template_folder = 'templates'

# Variables globales para almacenar los modelos entrenados
logistic_model = None
forest_model = None

@app.route('/')
def home():
    """Página principal con formulario de predicción para ambos modelos"""
    app_logger.info("Acceso a la página principal")
    
    # Verificar si los modelos ya han sido entrenados
    global logistic_model, forest_model
    if logistic_model is None or forest_model is None:
        app_logger.info("Modelos no inicializados, entrenando...")
        initialize_models()
        
    return render_template('home.html')

@app.route('/comparison')
def comparison():
    """Página de comparación de rendimiento de modelos"""
    try:
        app_logger.info("Acceso a la página de comparación de modelos")
        global logistic_model, forest_model
        
        # Verificar si los modelos ya han sido entrenados
        if logistic_model is None or forest_model is None:
            app_logger.info("Modelos no inicializados, entrenando...")
            # Entrenar los modelos
            initialize_models()
        
        # Comparar modelos
        comparison_data = compare_models(logistic_model, forest_model)
        
        if 'error' in comparison_data:
            app_logger.error(f"Error en la comparación: {comparison_data['error']}")
            return render_template('error.html', error=comparison_data['error'])
            
        return render_template(
            'comparison.html',
            comparison=comparison_data,
            logistic_metrics=logistic_model.model_data['metrics'],
            forest_metrics=forest_model.model_data['metrics']
        )
    except Exception as e:
        app_logger.error(f"Error al cargar página de comparación: {str(e)}", sys.exc_info())
        return render_template('error.html', error=str(e))

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    """Realizar predicción con el modelo logístico"""
    try:
        app_logger.info("Solicitud de predicción con modelo logístico recibida")
        
        # Validar datos del formulario
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in request.form:
                error_msg = f"Campo requerido ausente: {field}"
                app_logger.warning(error_msg)
                raise InvalidInputError(error_msg)
                
        # Validar que los valores sean numéricos
        try:
            features = np.array([[
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]])
        except ValueError as e:
            error_msg = f"Error en conversión de valores: {str(e)}"
            app_logger.warning(error_msg)
            raise InvalidInputError(error_msg)
            
        # Realizar predicción
        result = logistic_model.predict(features)
        
        app_logger.info(f"Predicción exitosa: {result['prediction']} con probabilidades: {result['probabilities']}")
        
        return jsonify({
            'prediction': result['prediction'],
            'probabilities': result['probabilities'],
            'success': True
        })
    except InvalidInputError as e:
        app_logger.error(f"Error de entrada inválida: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        })
    except Exception as e:
        app_logger.error(f"Error en predicción logística: {str(e)}", sys.exc_info())
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.route('/predict/forest', methods=['POST'])
def predict_forest():
    """Realizar predicción con el modelo Random Forest"""
    try:
        app_logger.info("Solicitud de predicción con modelo Random Forest recibida")
        
        # Validar datos del formulario
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in request.form:
                error_msg = f"Campo requerido ausente: {field}"
                app_logger.warning(error_msg)
                raise InvalidInputError(error_msg)
                
        # Validar que los valores sean numéricos
        try:
            features = np.array([[
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]])
        except ValueError as e:
            error_msg = f"Error en conversión de valores: {str(e)}"
            app_logger.warning(error_msg)
            raise InvalidInputError(error_msg)
            
        # Realizar predicción
        result = forest_model.predict(features)
        
        app_logger.info(f"Predicción exitosa: {result['prediction']} con probabilidades: {result['probabilities']}")
        
        return jsonify({
            'prediction': result['prediction'],
            'probabilities': result['probabilities'],
            'success': True
        })
    except InvalidInputError as e:
        app_logger.error(f"Error de entrada inválida: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        })
    except Exception as e:
        app_logger.error(f"Error en predicción Random Forest: {str(e)}", sys.exc_info())
        return jsonify({
            'error': str(e),
            'success': False
        })

def initialize_models():
    """Inicializar y entrenar los modelos"""
    global logistic_model, forest_model
    
    try:
        app_logger.info("Iniciando inicialización de modelos")
        
        # Cargar datos
        X_train, X_test, y_train, y_test, feature_names, target_names, X, y = load_iris_data()
        
        # Inicializar y entrenar modelo de Regresión Logística
        app_logger.info("Entrenando modelo de Regresión Logística")
        logistic_model = IrisModel()
        logistic_model.train_and_evaluate()
        
        # Inicializar y entrenar modelo Random Forest
        app_logger.info("Entrenando modelo Random Forest")
        forest_model = RandomForestModel()
        forest_model.train_and_evaluate(
            X_train, X_test, y_train, y_test, feature_names, target_names
        )
        
        app_logger.info("Modelos entrenados correctamente")
    except Exception as e:
        app_logger.error(f"Error al inicializar modelos: {str(e)}", sys.exc_info())
        raise ModelTrainingError("ambos", f"Error al inicializar modelos: {str(e)}")

@app.errorhandler(404)
def page_not_found(e):
    """Manejador para errores 404"""
    app_logger.warning(f"Página no encontrada: {request.path}")
    return render_template('error.html', error="Página no encontrada"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Manejador para errores 500"""
    app_logger.error(f"Error interno del servidor: {str(e)}")
    return render_template('error.html', error="Error interno del servidor"), 500

if __name__ == '__main__':
    # Entrenar modelos antes de iniciar la aplicación
    try:
        app_logger.info("Iniciando aplicación")
        initialize_models()
        
        # Iniciar aplicación
        app.run(debug=True)
    except Exception as e:
        app_logger.critical(f"Error fatal al iniciar la aplicación: {str(e)}", sys.exc_info()) 