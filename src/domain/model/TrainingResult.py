# src/domain/model/TrainingResult.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingResult:
    """
    Guarda la información clave tras entrenar el modelo.

    Atributos:
        accuracy (float): Precisión en el conjunto de validación.
        loss (float): Valor de la función de pérdida final.
        epochs_completed (int): Número de épocas completadas.
        training_history (Dict[str, Any]): Historial de métricas
            por época, p.ej. {'accuracy': [...], 'val_accuracy': [...], ...}.
    """
    accuracy: float
    loss: float
    epochs_completed: int
    training_history: Dict[str, Any]
    
    def is_successful(self) -> bool:
        """
        Indica si el entrenamiento fue satisfactorio según un umbral.

        Retorna:
            True si accuracy > 0.7, False en caso contrario.
        """
        return self.accuracy > 0.7  # Umbral mínimo de éxito
