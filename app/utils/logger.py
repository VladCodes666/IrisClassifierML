import os
import logging
import datetime
from logging.handlers import RotatingFileHandler
import traceback
import sys
import json

class Logger:
    """
    Clase para manejar el registro de eventos en la aplicación.
    Proporciona funcionalidades para registrar mensajes, errores y resultados
    de los modelos en archivos de registro.
    """
    
    def __init__(self, name="iris_classifier", log_level=logging.INFO):
        """
        Inicializa el logger con configuración personalizada.
        
        Args:
            name: Nombre del logger
            log_level: Nivel de registro (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Limpiar handlers previos si existen
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # Crear directorio de logs si no existe
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configurar formato de log
        self.formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Añadir handler para la consola (todos los niveles)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
        
        # Añadir handler para archivo de información (solo INFO y niveles inferiores)
        self.info_handler = self._add_file_handler('app.log', min_level=logging.INFO, max_level=logging.INFO)
        
        # Añadir handler específico para errores (ERROR y CRITICAL)
        self.error_handler = self._add_file_handler('error.log', min_level=logging.ERROR)
        
    def _add_file_handler(self, filename, min_level=None, max_level=None, max_bytes=10485760, backup_count=3):
        """
        Añade un manejador de archivo rotativo al logger con filtro de nivel.
        
        Args:
            filename: Nombre del archivo de log
            min_level: Nivel mínimo de log para este handler (opcional)
            max_level: Nivel máximo de log para este handler (opcional)
            max_bytes: Tamaño máximo del archivo antes de rotar (10MB por defecto)
            backup_count: Número de archivos de respaldo a mantener
            
        Returns:
            El handler creado
        """
        file_path = os.path.join(self.log_dir, filename)
        handler = RotatingFileHandler(
            file_path, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setFormatter(self.formatter)
        
        # Agregar filtro personalizado si se especificaron niveles
        if min_level is not None or max_level is not None:
            handler.addFilter(self._level_filter(min_level, max_level))
            
        self.logger.addHandler(handler)
        return handler
    
    def _level_filter(self, min_level=None, max_level=None):
        """
        Crea un filtro para limitar los registros a un rango de niveles específico.
        
        Args:
            min_level: Nivel mínimo de log (inclusive)
            max_level: Nivel máximo de log (inclusive)
            
        Returns:
            Función de filtro
        """
        min_level = min_level or logging.NOTSET
        max_level = max_level or logging.CRITICAL
        
        def filter_function(record):
            return min_level <= record.levelno <= max_level
            
        return filter_function
        
    def info(self, message):
        """Registra un mensaje informativo"""
        self.logger.info(message)
        
    def debug(self, message):
        """Registra un mensaje de depuración"""
        self.logger.debug(message)
        
    def warning(self, message):
        """Registra un mensaje de advertencia"""
        self.logger.warning(message)
        
    def error(self, message, exc_info=None):
        """
        Registra un mensaje de error, opcionalmente con información de la excepción
        
        Args:
            message: Mensaje de error
            exc_info: Información de la excepción (sys.exc_info()) si está disponible
        """
        if exc_info:
            self.logger.error(f"{message}\n{traceback.format_exception(*exc_info)}")
        else:
            self.logger.error(message)
            
    def critical(self, message, exc_info=None):
        """
        Registra un mensaje crítico, opcionalmente con información de la excepción
        
        Args:
            message: Mensaje crítico
            exc_info: Información de la excepción (sys.exc_info()) si está disponible
        """
        if exc_info:
            self.logger.critical(f"{message}\n{traceback.format_exception(*exc_info)}")
        else:
            self.logger.critical(message)
            
    def exception(self, message):
        """
        Registra una excepción con stack trace completo.
        Se debe llamar dentro de un bloque except.
        
        Args:
            message: Mensaje descriptivo de la excepción
        """
        self.logger.exception(message)
        
    def log_model_results(self, model_name, metrics, params=None):
        """
        Registra los resultados de un modelo en un archivo JSON y en el log.
        
        Args:
            model_name: Nombre del modelo (ej. 'logistic', 'random_forest')
            metrics: Diccionario con métricas del modelo
            params: Parámetros del modelo (opcional)
        """
        # Crear diccionario con resultados
        result = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_name,
            'metrics': {}
        }
        
        # Filtrar y añadir métricas (solo valores numéricos)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                result['metrics'][key] = value
                
        # Añadir parámetros si están disponibles
        if params:
            result['parameters'] = params
            
        # Guardar en archivo JSON
        results_file = os.path.join(self.log_dir, f'{model_name}_results.json')
        
        try:
            # Cargar resultados existentes si el archivo existe
            existing_results = []
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    existing_results = json.load(f)
                    if not isinstance(existing_results, list):
                        existing_results = [existing_results]
            
            # Añadir nuevo resultado
            existing_results.append(result)
            
            # Guardar resultados actualizados
            with open(results_file, 'w') as f:
                json.dump(existing_results, f, indent=4)
                
            # Registrar en el log
            self.info(f"Resultados del modelo {model_name} guardados correctamente")
            self.info(f"Métricas: Accuracy={metrics.get('accuracy', 'N/A')}, "
                     f"Precision={metrics.get('precision', 'N/A')}, "
                     f"Recall={metrics.get('recall', 'N/A')}, "
                     f"F1={metrics.get('f1', 'N/A')}")
                
        except Exception as e:
            self.error(f"Error al guardar resultados del modelo {model_name}: {str(e)}", sys.exc_info())
            
# Instancia global del logger
app_logger = Logger()