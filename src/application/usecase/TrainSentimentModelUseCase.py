
from typing import Any, Dict
from domain.model import TrainingResult
from domain.repository import ICommentRepository, IModelRepository


class TrainSentimentModelUseCase:
    
    def __init__(self, 
                 comment_repository: ICommentRepository,
                 model_repository: IModelRepository):
        self._comment_repository = comment_repository
        self._model_repository = model_repository
    
    def execute(self, data_source: str, training_config: Dict[str, Any] = None) -> TrainingResult:
        """
        Ejecuta el entrenamiento del modelo de análisis de sentimientos
        
        Args:
            data_source: Ruta al archivo de datos o identificador de fuente
            training_config: Configuración del entrenamiento (epochs, batch_size, etc.)
        """
        if training_config is None:
            training_config = {
                'epochs': 15,
                'batch_size': 8,
                'max_features': 1000,
                'max_length': 50
            }
        
        # 1. Cargar comentarios
        comments = self._comment_repository.load_comments(data_source)
        
        if not comments:
            # Usar comentarios de ejemplo si no se encuentran datos
            comments = self._comment_repository.get_sample_comments(50)
        
        # 2. Validar que hay suficientes datos
        if len(comments) < 10:
            raise ValueError("Se necesitan al menos 10 comentarios para entrenar el modelo")
        
        # 3. Entrenar modelo
        training_result = self._model_repository.train_model(comments, **training_config)
        
        # 4. Validar resultado del entrenamiento
        if not training_result.is_successful():
            raise RuntimeError(f"El entrenamiento falló. Precisión obtenida: {training_result.accuracy}")
        
        return training_result