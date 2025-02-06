import numpy as np
import mysql.connector
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import json

np.random.seed(42)
tf.random.set_seed(42)

def obtener_datos():
    try:
        conexion = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="base_datos"
        )
        consulta = """
            SELECT DATE(fecha) as dia, AVG(valor) as promedio
            FROM registros
            WHERE valor IS NOT NULL
            GROUP BY dia
            ORDER BY dia ASC;
        """
        datos = pd.read_sql(consulta, conexion)
        conexion.close()
        if datos.empty:
            raise ValueError("No hay suficientes datos en la base de datos.")
        return datos
    except Exception as e:
        print(f"Error al obtener datos de MySQL: {e}")
        return None

def preparar_datos(df):
    df['dia'] = pd.to_datetime(df['dia'])
    df['tendencia'] = df['promedio'].diff()
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def crear_modelo(forma_entrada):
    modelo = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=forma_entrada),
        Dropout(0.3),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='tanh'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(7, activation='linear')
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return modelo

def crear_secuencias(datos, dias):
    X, y = [], []
    for i in range(len(datos) - dias):
        X.append(datos[i:i + dias])
        y.append(datos[i + 1:i + dias + 1])
    return np.array(X), np.array(y)

datos = obtener_datos()
if datos is None:
    exit()

datos = preparar_datos(datos)
escalador = MinMaxScaler(feature_range=(0, 1))
valores_escalados = escalador.fit_transform(datos[['promedio']])

num_dias = len(valores_escalados)
print(f"Número de días disponibles: {num_dias}")

if num_dias < 7:
    print(f"Error: Se necesitan al menos 7 días de datos, pero solo hay {num_dias}.")
    exit()

dias_prediccion = 7
X, y = crear_secuencias(valores_escalados, dias_prediccion)

if X.shape[0] == 0:
    print("No se pueden crear secuencias con los datos disponibles.")
    exit()

tamanio_entrenamiento = int(len(X) * 0.8)
X_entrenamiento, X_prueba = X[:tamanio_entrenamiento], X[tamanio_entrenamiento:]
y_entrenamiento, y_prueba = y[:tamanio_entrenamiento], y[tamanio_entrenamiento:]

modelo = crear_modelo(forma_entrada=(X_entrenamiento.shape[1], X_entrenamiento.shape[2]))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
]

historial = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=200, batch_size=32, 
                       validation_data=(X_prueba, y_prueba), callbacks=callbacks, verbose=1)

prediccion = modelo.predict(X)
prediccion_desnormalizada = escalador.inverse_transform(prediccion.reshape(-1, 1)).flatten()

historico_real = escalador.inverse_transform(valores_escalados.reshape(-1, 1)).flatten()
print(f"Histórico completo: {historico_real.tolist()}")
print(f"Predicción para los próximos 7 días: {prediccion_desnormalizada.tolist()}")

prediccion_ultimos_7 = prediccion_desnormalizada[:7]
historico_real = historico_real[-7:]
error_mae = mean_absolute_error(historico_real, prediccion_ultimos_7)
precision = 100 - ((error_mae / (3 - 1)) * 100)

print(f"\nPrecisión del Modelo: {precision:.2f}%")

resultado = {
    "historico": [{"dia": row['dia'].strftime('%Y-%m-%d'), "promedio": row['promedio']}
                  for row in datos[['dia', 'promedio']].to_dict(orient='records')],
    "prediccion": [{"dia": (datos['dia'].iloc[-1] + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d'),
                    "valor_predicho": round(prediccion_desnormalizada[i], 2)}
                   for i in range(7)],
    "precision": f"{precision:.2f}%"
}

print(json.dumps(resultado, indent=4))

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(historico_real) + 1), historico_real, 'bo-', label='Histórico')
plt.plot(range(len(historico_real) + 1, len(historico_real) + 8), prediccion_ultimos_7, 'ro-',
         label='Predicción (7 días)')

plt.title('Predicción de Tendencia')
plt.xlabel('Días')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()

modelo.save('modelo_prediccion.h5')
