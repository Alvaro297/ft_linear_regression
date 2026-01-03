# ft_linear_regression

Implementación de un algoritmo de regresión lineal desde cero, sin utilizar librerías de machine learning, para predecir el precio de un coche en función de su kilometraje.

## 📋 Descripción

Este proyecto implementa un modelo de regresión lineal simple utilizando el algoritmo de **gradiente descendente** para entrenar el modelo. El objetivo es predecir el precio de un vehículo basándose en su kilometraje, sin depender de librerías de machine learning como scikit-learn.

**Características principales:**
- Implementación manual del algoritmo de gradiente descendente
- Normalización de datos para mejor convergencia
- Visualización de resultados con matplotlib
- Persistencia de parámetros entrenados en formato JSON

## 🚀 Instalación

### Requisitos previos
- Python 3.x
- pip (gestor de paquetes de Python)

### Dependencias

```bash
pip install pandas matplotlib numpy
```

## 📁 Estructura del proyecto

```
ft_linear_regression/
│
├── data.csv              # Dataset con kilometraje y precio de coches
├── training.py           # Script para entrenar el modelo
├── predict.py            # Script para realizar predicciones
└── trained_data.json     # Parámetros entrenados (theta0, theta1)
```

## 💻 Uso

### 1. Entrenar el modelo

Primero, necesitas entrenar el modelo con los datos disponibles:

```bash
python training.py
```

Este script:
- Carga los datos desde `data.csv`
- Normaliza los datos para mejorar la convergencia
- Aplica gradiente descendente durante 1000 iteraciones
- Calcula el error cuadrático medio (MSE)
- Guarda los parámetros `theta0` y `theta1` en `trained_data.json`

**Salida esperada:**
```
El error cuadratico medio es: [valor]
Theta propio (gradiente descendente, desnormalizado): [theta0] y [theta1]
```

### 2. Realizar predicciones

Una vez entrenado el modelo, puedes predecir precios:

```bash
python predict.py
```

El programa te pedirá introducir el kilometraje del coche: 

```
Introduce los kms del coche: 100000
El precio del coche es:  5234.56
```

Además, mostrará una gráfica con: 
- Puntos azules: datos reales del dataset
- Línea roja: línea de regresión calculada
- Punto amarillo: tu predicción

## 🧮 Algoritmo

### Gradiente Descendente

El modelo utiliza la ecuación de regresión lineal simple: 

```
y = θ₀ + θ₁ × x
```

Donde:
- `y` = precio predicho
- `x` = kilometraje
- `θ₀` (theta0) = intercepto
- `θ₁` (theta1) = pendiente

El algoritmo ajusta los parámetros mediante: 

```
θ₀ = θ₀ - α × (1/m) × Σ(y_pred - y_real)
θ₁ = θ₁ - α × (1/m) × Σ((y_pred - y_real) × x)
```

Donde:
- `α` (alpha) = tasa de aprendizaje (learning rate = 0.1)
- `m` = número de ejemplos
- `y_pred` = valor predicho
- `y_real` = valor real

### Normalización

Los datos se normalizan para mejorar la convergencia:

```
x_norm = (x - x_min) / (x_max - x_min)
```

Tras el entrenamiento, los parámetros se desnormalizan para trabajar con valores originales.

## 📊 Dataset

El archivo `data.csv` contiene 24 ejemplos de coches con dos características: 
- **km**: kilometraje del vehículo
- **price**:  precio del vehículo

Rango de datos: 
- Kilometraje: 22,899 - 240,000 km
- Precio: 3,650 - 8,290 €

## 🔧 Configuración

Puedes ajustar los siguientes parámetros en `training.py`:

- **Learning rate (`lr`)**: Actualmente 0.1
- **Iteraciones**:  Actualmente 1000
- **Inicialización**: `theta0` y `theta1` inician en 0.0

## ⚠️ Nota importante

Asegúrate de que la ruta en los archivos apunte correctamente a `trained_data.json`. Actualmente está configurada como:

```python
"./OuterCore/ft_linear_regressin/trained_data. json"
```

Es posible que necesites cambiarla a una ruta relativa local: 

```python
"./trained_data.json"
```

## 📈 Ejemplo de visualización

Al ejecutar `predict.py`, verás una gráfica similar a:

```
        │
  Precio│    •
        │      •   •
        │        •   •
        │          •─────── Línea de regresión
        │            •  •
        │              •  ⭐ (tu predicción)
        │                •
        └──────────────────────────
                Kilometraje
```

## 🎓 Aprendizaje

Este proyecto es parte del OuterCore y está diseñado para comprender:
- Fundamentos de machine learning
- Algoritmos de optimización (gradiente descendente)
- Normalización y preprocesamiento de datos
- Métricas de error (MSE)
- Implementación desde cero sin frameworks

## 📝 Licencia

Este proyecto es de código abierto y está disponible para fines educativos. 

## 👤 Autor

**Alvaro297**
- GitHub: [@Alvaro297](https://github.com/Alvaro297)

---

*Proyecto desarrollado como parte del programa OuterCore*
