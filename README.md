# IrisClassifierML

Flask web app comparing Logistic Regression and Random Forest models for iris species classification. Features interactive prediction interface, performance metrics, and visualization tools.

## Características

- Clasificación de especies Iris (setosa, versicolor, virginica)
- Implementación de dos modelos de ML: Regresión Logística y Random Forest
- Interfaz de usuario intuitiva con Bootstrap 5
- Tema claro/oscuro adaptable
- Visualización de probabilidades de predicción
- Métricas comparativas de rendimiento
- Sistema de logging para seguimiento de actividad

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python run.py
```

## Estructura del Proyecto

```
IrisVision/
│
├── app/                    # Directorio principal de la aplicación
│   ├── models/             # Implementación de modelos ML
│   │   ├── __init__.py     # Exporta los modelos
│   │   ├── logistic_model.py  # Modelo de regresión logística
│   │   └── random_forest_model.py  # Modelo Random Forest
│   │
│   ├── templates/          # Plantillas HTML
│   │   ├── layout.html     # Plantilla base
│   │   ├── home.html       # Página principal con formulario
│   │   ├── comparison.html # Comparación de modelos
│   │   ├── error.html      # Página de error
│   │   ├── forest.html     # Página específica Random Forest
│   │   └── logistic.html   # Página específica Reg. Logística
│   │
│   ├── utils/              # Utilidades
│   │   ├── __init__.py     # Exporta utilidades
│   │   ├── data_loader.py  # Carga de datos Iris
│   │   ├── compare.py      # Comparación de modelos
│   │   ├── logger.py       # Sistema de logging
│   │   └── exceptions.py   # Excepciones personalizadas
│   │
│   ├── logs/               # Archivos de registro
│   │
│   ├── __init__.py         # Inicializa app como paquete
│   └── app.py              # Aplicación Flask con rutas
│
├── requirements.txt        # Dependencias
└── run.py                  # Punto de entrada
```

## Tecnologías

- Python 3.x
- Flask
- scikit-learn
- numpy
- matplotlib
- Bootstrap 5

## Uso

1. Ingresar medidas de sépalos y pétalos en el formulario
2. Obtener predicciones de ambos modelos con distribución de probabilidades
3. Comparar rendimiento de los modelos en la sección de métricas
4. Cambiar entre modo claro/oscuro según preferencia

## API

La aplicación expone dos endpoints para predicciones:

- `/predict/logistic` - POST: Predicción usando modelo de Regresión Logística
- `/predict/forest` - POST: Predicción usando modelo Random Forest

Ambos esperan 4 parámetros: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`

Ejemplo de respuesta:
```json
{
  "prediction": "Iris-setosa",
  "probabilities": {
    "setosa": 97.2,
    "versicolor": 1.5,
    "virginica": 1.3
  },
  "success": true
}
```