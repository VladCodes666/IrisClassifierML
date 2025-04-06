import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
import sys
from app.utils.logger import app_logger
from app.utils.exceptions import VisualizationError

def compare_models(logistic_model, forest_model):
    """
    Compara el rendimiento de dos modelos y genera visualizaciones comparativas
    
    Args:
        logistic_model: Objeto con la implementación del modelo de regresión logística
        forest_model: Objeto con la implementación del modelo Random Forest
        
    Returns:
        Un diccionario con las métricas comparativas y visualizaciones
    """
    try:
        app_logger.info("Iniciando comparación de modelos")
        
        # Extraer métricas de ambos modelos
        logistic_metrics = logistic_model.model_data['metrics']
        forest_metrics = forest_model.model_data['metrics']
        
        # Determinar el mejor modelo basado en precisión
        log_accuracy = logistic_metrics['accuracy'] 
        rf_accuracy = forest_metrics['accuracy']
        
        best_model = 'Regresión Logística' if log_accuracy > rf_accuracy else 'Random Forest'
        app_logger.info(f"Mejor modelo basado en exactitud: {best_model} (Logística: {log_accuracy:.4f}, Random Forest: {rf_accuracy:.4f})")
        
        # Crear diccionario con métricas para comparación
        metrics_comparison = {
            'Regresión Logística': {
                'accuracy': logistic_metrics['accuracy'],
                'precision': logistic_metrics['precision'],
                'recall': logistic_metrics['recall'],
                'f1': logistic_metrics['f1']
            },
            'Random Forest': {
                'accuracy': forest_metrics['accuracy'],
                'precision': forest_metrics['precision'],
                'recall': forest_metrics['recall'],
                'f1': forest_metrics['f1']
            }
        }
        
        # Generar visualización comparativa
        app_logger.info("Generando visualización comparativa de métricas")
        metrics_comparison_img = generate_metrics_comparison(metrics_comparison)
        
        # Guardar resultados de la comparación en el log
        app_logger.log_model_results("comparison", {
            'log_accuracy': log_accuracy,
            'rf_accuracy': rf_accuracy,
            'log_precision': logistic_metrics['precision'],
            'rf_precision': forest_metrics['precision'],
            'log_recall': logistic_metrics['recall'],
            'rf_recall': forest_metrics['recall'],
            'log_f1': logistic_metrics['f1'],
            'rf_f1': forest_metrics['f1']
        })
        
        # Resultado final
        app_logger.info("Comparación de modelos completada con éxito")
        return {
            'best_model': best_model,
            'metrics_comparison': metrics_comparison,
            'metrics_comparison_img': metrics_comparison_img,
            'log_best_params': logistic_model.model_data['best_params'],
            'rf_best_params': forest_model.model_data['best_params']
        }
    
    except Exception as e:
        app_logger.error(f"Error en comparación de modelos: {str(e)}", sys.exc_info())
        # Manejar errores
        return {
            'error': str(e),
            'success': False
        }

def generate_metrics_comparison(metrics_comparison):
    """
    Genera una visualización comparativa de métricas entre modelos
    
    Args:
        metrics_comparison: Diccionario con métricas de ambos modelos
        
    Returns:
        Imagen en formato base64 con la comparación
        
    Raises:
        VisualizationError: Si ocurre un error al generar la visualización
    """
    try:
        app_logger.info("Generando visualización comparativa de métricas")
        
        # Preparar datos para la visualización
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(metrics_comparison.keys())
        
        # Crear DataFrame para facilitar la visualización
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Modelo': model,
                    'Métrica': metric.capitalize(),
                    'Valor': metrics_comparison[model][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Crear visualización
        plt.figure(figsize=(12, 8))
        
        # Crear paleta de colores personalizada
        colors = {'Regresión Logística': '#6c5ce7', 'Random Forest': '#00b894'}
        
        # Crear gráfico de barras agrupadas
        sns.barplot(
            x='Métrica', 
            y='Valor', 
            hue='Modelo', 
            data=df,
            palette=colors
        )
        
        plt.title('Comparación de Métricas entre Modelos', fontsize=16)
        plt.ylabel('Valor', fontsize=14)
        plt.xlabel('Métrica', fontsize=14)
        plt.legend(title='Modelo')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)  # Establecer límites del eje Y
        
        # Añadir valores sobre las barras
        for i, p in enumerate(plt.gca().patches):
            plt.text(
                p.get_x() + p.get_width() / 2., 
                p.get_height() + 0.01,
                f'{p.get_height():.3f}', 
                ha='center', 
                fontsize=10
            )
        
        # Convertir el gráfico a una imagen base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        app_logger.info("Visualización comparativa generada correctamente")
        return img_str
    
    except Exception as e:
        app_logger.error(f"Error en generate_metrics_comparison: {str(e)}", sys.exc_info())
        raise VisualizationError(f"Error al generar visualización comparativa: {str(e)}") 