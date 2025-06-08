# cnn_sentiment_model.py

"""
Red neuronal convolucional (CNN) para clasificación de sentimientos.
Usa TensorFlow/Keras como framework principal.
Este archivo define la arquitectura, la compila y puede ejecutarse como script.
"""

# =========================
# 1. Importar librerías necesarias
# =========================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)

# =========================
# 2. Definir función para construir la red
# =========================
def build_cnn(vocab_size: int = 5000, max_length: int = 100) -> tf.keras.Model:
    """
    Construye y compila la arquitectura CNN.
    :param vocab_size: tamaño del vocabulario
    :param max_length: longitud máxima de las secuencias de entrada
    :return: modelo compilado listo para entrenar
    """
    # Crear modelo secuencial
    model = Sequential()

    # Capa de embedding: convierte tokens en vectores densos
    model.add(Embedding(
        input_dim=vocab_size,      # número de palabras únicas
        output_dim=128,            # dimensión de embedding
        input_length=max_length    # longitud de cada secuencia
    ))

    # Capa convolucional 1D: detecta patrones locales en ventanas de tamaño 5
    model.add(Conv1D(
        filters=64,                # cantidad de filtros
        kernel_size=5,             # tamaño del kernel
        activation='relu'          # función de activación
    ))

    # Capa de pooling: reduce dimensionalidad manteniendo características clave
    model.add(MaxPooling1D(
        pool_size=2                # tamaño de la ventana de pooling
    ))

    # Aplanar la salida para conectar con capas densas
    model.add(Flatten())

    # Capa densa intermedia: aprendizaje de representaciones de alto nivel
    model.add(Dense(
        units=64,                  # neuronas en la capa
        activation='relu'          # función de activación
    ))

    # Dropout para prevenir overfitting
    model.add(Dropout(
        rate=0.5                   # probabilidad de desactivar neuronas
    ))

    # Capa de salida: clasificación binaria con sigmoide
    model.add(Dense(
        units=1,                   # una neurona de salida
        activation='sigmoid'       # sigmoide para probabilidad
    ))

    # Compilar el modelo con optimizador Adam y pérdida binaria
    model.compile(
        optimizer='adam',                   # optimizador adaptativo
        loss='binary_crossentropy',         # función de pérdida
        metrics=['accuracy']                # métrica de rendimiento
    )

    return model

# =========================
# 3. Ejecución directa como script
# =========================
if __name__ == "__main__":
    # Parámetros por defecto
    DEFAULT_VOCAB_SIZE = 5000
    DEFAULT_MAX_LENGTH = 100

    print("Resumen de la arquitectura de la CNN:")
    # Construir y mostrar resumen de la red
    cnn_model = build_cnn(DEFAULT_VOCAB_SIZE, DEFAULT_MAX_LENGTH)
    cnn_model.summary()
