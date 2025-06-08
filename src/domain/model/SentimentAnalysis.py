# src/domain/model/SentimentAnalysis.py

from dataclasses import dataclass
from .Comment import Comment

@dataclass
class SentimentAnalysis:
    """
    Resultado de analizar el sentimiento de un comentario.

    Atributos:
        comment (Comment): El objeto Comment original.
        predicted_sentiment (str): "Positivo" o "Negativo".
        confidence_score (float): Probabilidad asociada (0.0–1.0).
    """
    comment: Comment
    predicted_sentiment: str
    confidence_score: float
    
    @classmethod
    def create(cls, comment: Comment, prediction: float) -> 'SentimentAnalysis':
        """
        Crea una instancia de SentimentAnalysis a partir de la puntuación
        de la red (valor entre 0 y 1).

        Args:
            comment (Comment): Comentario a analizar.
            prediction (float): Salida de la red, probabilidad de positivo.

        Retorna:
            SentimentAnalysis con etiqueta y confianza calculadas.
        """
        # Decidir etiqueta según umbral 0.5
        sentiment = "Positivo" if prediction > 0.5 else "Negativo"
        # Ajustar confianza para que siempre sea ≥ 0.5
        confidence = prediction if prediction > 0.5 else 1 - prediction
        return cls(comment, sentiment, confidence)
