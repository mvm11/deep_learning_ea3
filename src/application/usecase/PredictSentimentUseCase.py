from typing import List, Union
from domain.model.SentimentAnalysis import SentimentAnalysis
from domain.model.Comment import Comment
from domain.repository.IModelRepository import IModelRepository


class PredictSentimentUseCase:
    
    def __init__(self, model_repository: IModelRepository):
        self._model_repository = model_repository
    
    def execute(self, comments: Union[str, List[str]]) -> List[SentimentAnalysis]:
        """
        Predice el sentimiento de uno o varios comentarios
        
        Args:
            comments: Comentario único (string) o lista de comentarios
            
        Returns:
            Lista de análisis de sentimiento
        """
        # 1. Validar que el modelo esté entrenado
        if not self._model_repository.is_model_trained():
            raise RuntimeError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # 2. Normalizar entrada
        if isinstance(comments, str):
            comments = [comments]
        
        # 3. Crear objetos Comment
        comment_objects = [Comment(content=text) for text in comments]
        
        # 4. Realizar predicciones
        predictions = self._model_repository.predict_sentiment(comment_objects)
        
        return predictions
    
    def predict_single(self, comment_text: str) -> SentimentAnalysis:
        """Conveniencia para predecir un solo comentario"""
        results = self.execute(comment_text)
        return results[0] if results else None