import numpy as np
import re
from typing import List, Dict, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dropout,
    Dense
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from domain.model.Comment import Comment
from domain.model.SentimentAnalysis import SentimentAnalysis
from domain.model.TrainingResult import TrainingResult
from domain.repository.IModelRepository import IModelRepository


class TensorFlowModelRepository(IModelRepository):
    """
    Implementación de IModelRepository usando TensorFlow/Keras.
    Entrena y predice con una arquitectura CNN + RNN para análisis de sentimientos.
    """
    def __init__(self):
        """
        Inicializa valores predeterminados y fija semillas para reproducibilidad.
        """
        self.model = None
        self.tokenizer = None
        # Parámetros de tokenización y padding
        self.max_features = 1000  # número máximo de palabras únicas
        self.max_length = 50     # longitud fija de cada secuencia

        # Establecer semillas para garantizar resultados reproducibles
        np.random.seed(42)
        tf.random.set_seed(42)

    def train_model(self, comments: List[Comment], **config) -> TrainingResult:
        """
        Entrena la red con los comentarios datos.

        Pasos:
        1. Aplicar configuración de entrenamiento (epochs, batch_size, etc.).
        2. Preprocesar textos: limpieza y normalización.
        3. Tokenizar y convertir a secuencias.
        4. Dividir en conjunto de entrenamiento y prueba.
        5. Construir la arquitectura y entrenar.
        6. Evaluar en datos de prueba.

        Args:
            comments: lista de objetos Comment con etiqueta sentiment.
            config: parámetros de entrenamiento: epochs, batch_size,
                    max_features, max_length y validation_split.

        Retorna:
            TrainingResult con precisión, pérdida, épocas completadas
            e historial de métricas.
        """
        # Leer configuración con valores por defecto
        self.max_features = config.get('max_features', self.max_features)
        self.max_length = config.get('max_length', self.max_length)
        epochs = config.get('epochs', 15)
        batch_size = config.get('batch_size', 8)
        validation_split = config.get('validation_split', 0.2)

        # 1. Preparar los datos de entrada
        texts = [self._preprocess_text(c.content) for c in comments]
        labels = [c.sentiment for c in comments]

        # 2. Crear el tokenizer y ajustarlo al texto
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)

        # 3. Convertir textos a secuencias de enteros y hacer padding
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        y = np.array(labels)

        # 4. División en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. Construir y entrenar la red
        vocab_size = min(self.max_features, len(self.tokenizer.word_index) + 1)
        self.model = self._build_model(vocab_size)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        # 6. Evaluar desempeño en el conjunto de prueba
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        return TrainingResult(
            accuracy=test_accuracy,
            loss=test_loss,
            epochs_completed=epochs,
            training_history=history.history
        )

    def predict_sentiment(self, comments: List[Comment]) -> List[SentimentAnalysis]:
        """
        Genera predicciones de sentimiento para una lista de comentarios.

        Pasos:
        1. Verificar que el modelo ya esté entrenado.
        2. Preprocesar cada texto y convertir a secuencia.
        3. Llamar a model.predict y transformar en SentimentAnalysis.

        Args:
            comments: lista de objetos Comment a analizar.

        Retorna:
            Lista de SentimentAnalysis con etiqueta y confianza.
        """
        if not self.is_model_trained():
            raise RuntimeError("El modelo debe estar entrenado para hacer predicciones")

        # Limpiar y normalizar cada comentario
        texts = [self._preprocess_text(c.content) for c in comments]
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')

        # Obtener probabilidades de sentimiento positivo
        predictions = self.model.predict(X)

        # Convertir a objetos de dominio
        results = []
        for c, prob in zip(comments, predictions):
            analysis = SentimentAnalysis.create(c, prob[0])
            results.append(analysis)

        return results

    def save_model(self, file_path: str) -> bool:
        """
        Guarda el modelo entrenado y el tokenizer en archivos.

        Args:
            file_path: ruta base para guardar (sin extensión).

        Retorna:
            True si se guardó correctamente, False en caso de error.
        """
        try:
            if self.model is None:
                return False
            # Guardar modelo principal
            self.model.save(f"{file_path}_model.h5")
            # Serializar tokenizer
            import pickle
            with open(f"{file_path}_tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)
            return True
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            return False

    def load_model(self, file_path: str) -> bool:
        """
        Carga un modelo previamente guardado junto con su tokenizer.

        Args:
            file_path: ruta base de los archivos guardados.

        Retorna:
            True si la carga fue exitosa, False en caso contrario.
        """
        try:
            self.model = tf.keras.models.load_model(f"{file_path}_model.h5")
            # Deserializar tokenizer
            import pickle
            with open(f"{file_path}_tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return False

    def is_model_trained(self) -> bool:
        """
        Verifica si el modelo y tokenizer están inicializados.

        Retorna:
            True si hay modelo y tokenizer cargados, False en otro caso.
        """
        return (self.model is not None) and (self.tokenizer is not None)

    def _build_model(self, vocab_size: int) -> Sequential:
        """
        Define la arquitectura CNN + RNN para análisis de sentimientos.

        Args:
            vocab_size: tamaño real del vocabulario a usar.

        Retorna:
            Modelo compilado de Keras.
        """
        model = Sequential([
            # Capa de embedding para vectores densos
            Embedding(vocab_size, 128, input_length=self.max_length),
            # Convolución 1D para extraer características locales
            Conv1D(64, 5, activation='relu'),
            # Pooling para reducir dimensionalidad
            MaxPooling1D(2),
            # LSTM bidireccional para captar dependencias secuenciales
            Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
            # Dropout para evitar sobreajuste
            Dropout(0.5),
            # Capa densa intermedia
            Dense(32, activation='relu'),
            # Otro dropout
            Dropout(0.3),
            # Salida con sigmoide para clasificación binaria
            Dense(1, activation='sigmoid')
        ])

        # Compilar con Adam y pérdida binaria
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _preprocess_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto de entrada.

        Pasos:
        1. Pasar a minúsculas.
        2. Eliminar caracteres no alfabéticos.
        3. Colapsar espacios múltiples.

        Args:
            text: cadena original.

        Retorna:
            Texto procesado listo para tokenizar.
        """
        text = text.lower()
        text = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', text)
        text = ' '.join(text.split())
        return text
