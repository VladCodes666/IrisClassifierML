class IrisClassifierException(Exception):
    """Clase base para excepciones personalizadas en la aplicación"""
    pass


class ModelTrainingError(IrisClassifierException):
    """Excepción lanzada cuando ocurre un error durante el entrenamiento de un modelo"""
    
    def __init__(self, model_name, message="Error durante el entrenamiento del modelo"):
        self.model_name = model_name
        self.message = f"{message}: {model_name}"
        super().__init__(self.message)


class PredictionError(IrisClassifierException):
    """Excepción lanzada cuando ocurre un error durante la predicción"""
    
    def __init__(self, model_name, message="Error durante la predicción con el modelo"):
        self.model_name = model_name
        self.message = f"{message}: {model_name}"
        super().__init__(self.message)


class InvalidInputError(IrisClassifierException):
    """Excepción lanzada cuando los datos de entrada no son válidos"""
    
    def __init__(self, message="Datos de entrada no válidos"):
        self.message = message
        super().__init__(self.message)


class DataLoadError(IrisClassifierException):
    """Excepción lanzada cuando hay un error al cargar datos"""
    
    def __init__(self, message="Error al cargar los datos"):
        self.message = message
        super().__init__(self.message)


class VisualizationError(IrisClassifierException):
    """Excepción lanzada cuando hay un error al generar visualizaciones"""
    
    def __init__(self, message="Error al generar visualización"):
        self.message = message
        super().__init__(self.message)


class ModelNotTrainedError(IrisClassifierException):
    """Excepción lanzada cuando se intenta usar un modelo no entrenado"""
    
    def __init__(self, model_name, message="Modelo no entrenado"):
        self.model_name = model_name
        self.message = f"{message}: {model_name}"
        super().__init__(self.message) 