from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..model.Comment import Comment
from ..model.TrainingResult import TrainingResult
from ..model.SentimentAnalysis import SentimentAnalysis

class IModelRepository(ABC):
    
    @abstractmethod
    def train_model(self, comments: List[Comment], **config) -> TrainingResult:
        """Entrena el modelo con los comentarios proporcionados"""
        pass
    
    @abstractmethod
    def predict_sentiment(self, comments: List[Comment]) -> List[SentimentAnalysis]:
        """Predice el sentimiento de los comentarios"""
        pass
    
    @abstractmethod
    def save_model(self, file_path: str) -> bool:
        """Guarda el modelo entrenado"""
        pass
    
    @abstractmethod
    def load_model(self, file_path: str) -> bool:
        """Carga un modelo previamente entrenado"""
        pass
    
    @abstractmethod
    def is_model_trained(self) -> bool:
        """Verifica si el modelo está entrenado"""
        pass