#!/usr/bin/env python3
"""
Flujo principal:
1. Muestra el summary de la CNN.
2. Entrena el modelo con comentarios en CSV.
3. Imprime predicciones de prueba.
4. Genera matriz de confusión y curva de exactitud en docs/.
"""

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Casos de uso y repositorios
from application.usecase.TrainSentimentModelUseCase import TrainSentimentModelUseCase
from application.usecase.PredictSentimentUseCase import PredictSentimentUseCase
from infrastructure.repository.CSVCommentRepository import CSVCommentRepository
from infrastructure.repository.TensorFlowModelRepository import TensorFlowModelRepository
# Función de construcción de la CNN (con input_shape correcto)
from infrastructure.model.cnn_sentiment_model import build_cnn


def main() -> None:
    # --------- 0. Parámetros ------------------
    VOCAB_SIZE = 1000
    MAX_LEN = 50
    DATA_PATH = "/Users/mateo/Desktop/EA3/db/comments.csv"

    # --------- 1. Mostrar summary --------------
    print("\nResumen de la arquitectura de la CNN:")
    cnn_model = build_cnn(VOCAB_SIZE, MAX_LEN)
    # Ya tiene input_shape dentro de Embedding → summary mostrará parámetros
    cnn_model.summary()

    # --------- 2. Inyección de dependencias ----
    comment_repo = CSVCommentRepository()
    model_repo = TensorFlowModelRepository()
    train_uc = TrainSentimentModelUseCase(comment_repo, model_repo)
    predict_uc = PredictSentimentUseCase(model_repo)

    # --------- 3. Entrenamiento ----------------
    print("\nIniciando entrenamiento...")
    training_cfg = {
        "epochs": 15,
        "batch_size": 8,
        "max_features": VOCAB_SIZE,
        "max_length": MAX_LEN,
    }
    # Cargar comentarios (para métricas y entrenamiento)
    all_comments = comment_repo.load_comments(DATA_PATH)
    result = train_uc.execute(DATA_PATH, training_cfg)
    print(f"Entrenamiento completado. Precisión: {result.accuracy:.4f}\n")

    # --------- 4. Predicciones de prueba -------
    test_comments = [
        "El servicio fue terrible y muy lento",
        "Excelente trabajo, muy profesional",
        "El producto llegó en perfectas condiciones",
    ]
    predictions = predict_uc.execute(test_comments)
    for p in predictions:
        icon = "🟢" if p.predicted_sentiment == "Positivo" else "🔴"
        print(f"📝 {p.comment.content}")
        print(f"{icon} {p.predicted_sentiment} (Confianza: {p.confidence_score:.4f})")
        print("-" * 50)

    # --------- 5. Diagramas en docs/ -----------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    # 5.1 Matriz de confusión
    y_true = [c.sentiment for c in all_comments]
    y_pred = [
        1 if sa.predicted_sentiment == "Positivo" else 0
        for sa in predict_uc.execute([c.content for c in all_comments])
    ]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negativo", "Positivo"])
    disp.plot()
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "confusion_matrix.png"))
    plt.close()

    # 5.2 Curva de exactitud
    plt.figure()
    plt.plot(result.training_history["accuracy"], label="Entrenamiento")
    plt.plot(result.training_history["val_accuracy"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Exactitud")
    plt.title("Exactitud por Época")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "accuracy_plot.png"))
    plt.close()

    print(f"Diagramas guardados en {docs_dir}/")


if __name__ == "__main__":
    main()
