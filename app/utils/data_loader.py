from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sys
from app.utils.logger import app_logger
from app.utils.exceptions import DataLoadError

def load_iris_data(test_size=0.3, random_state=42):
    """
    Carga el dataset de Iris y lo divide en train/test
    
    Args:
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla con (X_train, X_test, y_train, y_test, feature_names, target_names, X, y)
        
    Raises:
        DataLoadError: Si ocurre un error al cargar los datos
    """
    try:
        app_logger.info(f"Cargando conjunto de datos Iris con test_size={test_size}")
        
        # Cargar el conjunto de datos
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        app_logger.info(f"Conjunto de datos cargado: {X.shape[0]} muestras, {X.shape[1]} características")
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        app_logger.info(f"Datos divididos: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")
        
        return X_train, X_test, y_train, y_test, feature_names, target_names, X, y
        
    except Exception as e:
        app_logger.error(f"Error al cargar datos de Iris: {str(e)}", sys.exc_info())
        raise DataLoadError(f"Error al cargar conjunto de datos Iris: {str(e)}") 