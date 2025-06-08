from abc import ABC, abstractmethod
from typing import List
from ..model.Comment import Comment

class ICommentRepository(ABC):
    
    @abstractmethod
    def load_comments(self, file_path: str) -> List[Comment]:
        """Carga comentarios desde una fuente de datos"""
        pass
    
    @abstractmethod
    def save_comments(self, comments: List[Comment], file_path: str) -> bool:
        """Guarda comentarios en una fuente de datos"""
        pass
    
    @abstractmethod
    def get_sample_comments(self, count: int) -> List[Comment]:
        """Obtiene comentarios de ejemplo para pruebas"""
        pass