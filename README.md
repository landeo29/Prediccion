# 📊 Predicción de Estrés con LSTM  

🚀 **Este proyecto utiliza redes neuronales LSTM para predecir niveles de estrés a partir de datos almacenados en MySQL.**  

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/2/28/Artificial_neural_network.svg" width="400px">
</div>

---

## 🛠️ Tecnologías Utilizadas  

🔹 **Python** (TensorFlow, Pandas, Scikit-learn, Matplotlib)  
🔹 **MySQL** (Base de datos para almacenar datos de estrés)  
🔹 **LSTM (Long Short-Term Memory)** (Modelo de aprendizaje profundo para predicciones de series temporales)  

---

## 📌 Requisitos  

Antes de ejecutar el proyecto, asegúrate de tener instalados los siguientes paquetes:  

```sh
pip install tensorflow pandas numpy mysql-connector-python matplotlib scikit-learn
```

También, asegúrate de tener una base de datos MySQL configurada.  

---

## ⚙️ Instalación y Configuración  

### 1️⃣ Clonar el repositorio  

```sh
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

### 2️⃣ Configurar la base de datos  

Asegúrate de tener una base de datos con una tabla `registros` que contenga los siguientes campos:  

| Campo     | Tipo        | Descripción |
|-----------|------------|-------------|
| id        | INT (PK)   | Identificador único |
| fecha     | DATE       | Fecha del registro |
| valor     | FLOAT      | Nivel de estrés promedio |

📌 **Modifica las credenciales en el script si es necesario.**  

```python
conexion = mysql.connector.connect(
    host="localhost",
    user="tu_usuario",
    password="tu_contraseña",
    database="tu_base_datos"
)
```

---

## 🚀 Ejecución  

Ejecuta el script principal para entrenar el modelo y generar predicciones:  

```sh
python main.py
```

El modelo calculará la tendencia basada en los **últimos 7 días** y predecirá los siguientes **7 días**.  

📊 **Resultados:**  
✔️ JSON con la predicción  
✔️ Gráfico comparativo de datos reales y predicciones  

---

## 📊 Visualización de Resultados  

El script generará un gráfico con los valores históricos y la predicción:  

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Matplotlib_pie_chart.svg" width="400px">
</div>

🔹 **Línea azul** → Datos reales  
🔹 **Línea roja** → Predicción para los próximos 7 días  

---

## 💾 Guardado del Modelo  

El modelo entrenado se guardará en el archivo `modelo_prediccion.h5`, lo que permite reutilizarlo sin necesidad de reentrenarlo.  

---

## 🔥 JSON de Salida  

La salida del script incluye un JSON con la predicción:  

```json
{
    "historico": [
        {"dia": "2025-02-01", "promedio": 2.5},
        {"dia": "2025-02-02", "promedio": 3.1}
    ],
    "prediccion": [
        {"dia": "2025-02-08", "valor_predicho": 2.8},
        {"dia": "2025-02-09", "valor_predicho": 3.2}
    ],
    "precision": "92.5%"
}
```

---

## 📜 Licencia  

Este proyecto es de uso libre. **Personalízalo como necesites** y siéntete libre de mejorarlo. 🚀  

---

Si te gustó este proyecto, ¡no olvides darle una ⭐ en GitHub! 😎🔥
