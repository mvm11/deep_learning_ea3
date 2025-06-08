# src/domain/model/Comment.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class Comment:
    """
    Representa un comentario de usuario junto con su etiqueta de sentimiento
    y nivel de confianza (opcional).

    Atributos:
        content (str): Texto completo del comentario.
        sentiment (Optional[int]): Etiqueta de sentimiento:
            0 = Negativo, 1 = Positivo, None si aún no está etiquetado.
        confidence (Optional[float]): Grado de confianza en la etiqueta
            (entre 0.0 y 1.0), o None si no aplica.
    """
    content: str
    sentiment: Optional[int] = None
    confidence: Optional[float] = None

    def is_positive(self) -> bool:
        """
        Indica si este comentario se considera positivo.
        
        Retorna:
            True si sentiment == 1, False en cualquier otro caso.
        """
        return self.sentiment == 1 if self.sentiment is not None else False

    def is_negative(self) -> bool:
        """
        Indica si este comentario se considera negativo.
        
        Retorna:
            True si sentiment == 0, False en cualquier otro caso.
        """
        return self.sentiment == 0 if self.sentiment is not None else False
