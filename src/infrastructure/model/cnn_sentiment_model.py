# cnn_sentiment_model.py

"""
Red neuronal convolucional (CNN) para clasificación de sentimientos.
Usa TensorFlow/Keras como framework principal.
Este script define la red, la compila, y muestra su topología con summary().
"""

# =========================
# 1. Importar librerías necesarias
# =========================
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,     # convierte tokens en vectores densos
    Conv1D,        # extrae patrones locales de la secuencia
    MaxPooling1D,  # reduce dimensionalidad reteniendo características clave
    Flatten,       # aplana la salida 2D a 1D para la capa densa
    Dense,         # capa totalmente conectada
    Dropout        # regularización para evitar overfitting
)

# =========================
# 2. Función que construye y compila la CNN
# =========================
def build_cnn(vocab_size: int = 5000, max_length: int = 100) -> tf.keras.Model:
    """
    Construye y compila la arquitectura CNN.
    Args:
        vocab_size (int): tamaño del vocabulario (número de tokens únicos).
        max_length (int): longitud fija de cada secuencia de entrada.
    Returns:
        tf.keras.Model: modelo compilado listo para entrenar.
    """
    # Iniciar modelo secuencial
    model = Sequential()

    # Capa de embedding: convierte enteros de token en vectores densos
    model.add(Embedding(
        input_dim=vocab_size,      # número máximo de tokens
        output_dim=128,            # dimensión de embedding por token
        input_shape=(max_length,)  # forma de cada entrada (batch, seq_len)
    ))

    # Capa convolucional 1D: detecta patrones locales de tamaño 5
    model.add(Conv1D(
        filters=64,                # cuántos filtros/convoluciones
        kernel_size=5,             # tamaño de la ventana de convolución
        activation='relu'          # función de activación ReLU
    ))

    # Capa de pooling: reduce la longitud de la secuencia a la mitad
    model.add(MaxPooling1D(
        pool_size=2                # tamaño de la ventana de pooling
    ))

    # Aplano la salida para pasar a la capa densa
    model.add(Flatten())

    # Capa densa oculta: aprendizaje de representaciones de alto nivel
    model.add(Dense(
        units=64,                  # neuronas en esta capa
        activation='relu'          # función ReLU
    ))

    # Dropout: apaga aleatoriamente neuronas para evitar overfitting
    model.add(Dropout(rate=0.5))

    # Capa de salida: produce probabilidad para clasificación binaria
    model.add(Dense(
        units=1,                   # una sola neurona de salida
        activation='sigmoid'       # sigmoide para probabilidad entre 0 y 1
    ))

    # Compilar el modelo: definir optimizador y función de pérdida
    model.compile(
        optimizer='adam',               # Adam adaptativo
        loss='binary_crossentropy',     # pérdida para binaria
        metrics=['accuracy']            # métrica de precisión
    )

    return model

# =========================
# 3. Ejecución directa como script
# =========================
if __name__ == "__main__":
    # Parámetros por defecto para pruebas
    DEFAULT_VOCAB_SIZE = 5000
    DEFAULT_MAX_LENGTH = 100

    print("\nResumen de la arquitectura de la CNN:")
    # Construir el modelo con la forma de entrada conocida
    cnn_model = build_cnn(DEFAULT_VOCAB_SIZE, DEFAULT_MAX_LENGTH)
    # Mostrar topología y conteo de parámetros
    cnn_model.summary()
