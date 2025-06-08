#!/usr/bin/env python3
"""
Script para generar y guardar diagramas de métricas del modelo:
- Matriz de Confusión
- Evolución de Exactitud por Época
Guarda las imágenes en la carpeta docs junto a este script.
"""
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from application.usecase.TrainSentimentModelUseCase import TrainSentimentModelUseCase
from application.usecase.PredictSentimentUseCase import PredictSentimentUseCase
from infrastructure.repository.CSVCommentRepository import CSVCommentRepository
from infrastructure.repository.TensorFlowModelRepository import TensorFlowModelRepository


def main():
    # Inyección de dependencias
    comment_repo = CSVCommentRepository()
    model_repo = TensorFlowModelRepository()

    # =========================
    # 1. Mostrar resumen de la arquitectura de la red
    # =========================
    temp_vocab = 1000
    temp_length = 50
    print("Resumen de la arquitectura de la CNN:")
    model = model_repo._build_model(temp_vocab)
    model.summary()

    # =========================
    # 2. Definir casos de uso
    # =========================
    train_use_case = TrainSentimentModelUseCase(comment_repo, model_repo)
    predict_use_case = PredictSentimentUseCase(model_repo)

    # =========================
    # 3. Cargar y entrenar modelo
    # =========================
    data_path = "/Users/mateo/Desktop/EA3/db/comments.csv"
    print("Iniciando entrenamiento...")
    training_config = {
        'epochs': 15,
        'batch_size': 8,
        'max_features': temp_vocab,
        'max_length': temp_length
    }
    # 3.1. Cargar todos los comentarios para métricas posteriores
    all_comments = comment_repo.load_comments(data_path)

    result = train_use_case.execute(data_path, training_config)
    print(f"Entrenamiento completado. Precisión: {result.accuracy:.4f}")

    # =========================
    # 4. Realizar predicciones de prueba
    # =========================
    test_comments = [
        "El servicio fue terrible y muy lento",
        "Excelente trabajo, muy profesional",
        "El producto llegó en perfectas condiciones"
    ]
    predictions = predict_use_case.execute(test_comments)
    for pred in predictions:
        icon = "🟢" if pred.predicted_sentiment == "Positivo" else "🔴"
        print(f"📝 Comentario: {pred.comment.content}")
        print(f"{icon} Sentimiento: {pred.predicted_sentiment} (Confianza: {pred.confidence_score:.4f})")
        print("-" * 50)

    # =========================
    # 5. Generar diagramas de métricas en docs/
    # =========================
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Preparar directorio de docs
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    docs_dir = os.path.join(base_dir, 'docs')
    os.makedirs(docs_dir, exist_ok=True)

    # Matriz de confusión sobre todo el conjunto
    y_true = [c.sentiment for c in all_comments]
    preds_all = predict_use_case.execute([c.content for c in all_comments])
    y_pred = [1 if p.predicted_sentiment == 'Positivo' else 0 for p in preds_all]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Negativo', 'Positivo'])
    disp.plot()
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'confusion_matrix.png'))
    plt.close()

    # Gráfica de exactitud por época
    plt.figure()
    plt.plot(result.training_history['accuracy'], label='Entrenamiento')
    plt.plot(result.training_history['val_accuracy'], label='Validación')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.title('Exactitud por Época')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'accuracy_plot.png'))
    plt.close()

    print(f"Diagramas guardados en {docs_dir}/")
    # Definir rutas
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'db', 'comments.csv')
    docs_dir = os.path.dirname(__file__)

    # Inyección de dependencias
    comment_repo = CSVCommentRepository()
    model_repo = TensorFlowModelRepository()
    train_uc = TrainSentimentModelUseCase(comment_repo, model_repo)
    predict_uc = PredictSentimentUseCase(model_repo)

    # 1) Entrenar modelo y obtener métricas de entrenamiento
    result = train_uc.execute(data_path, {
        'epochs': 15,
        'batch_size': 8,
        'max_features': 1000,
        'max_length': 50
    })
    history = result.training_history

    # 2) Cargar comentarios reales y etiquetados
    comments = comment_repo.load_comments(data_path)
    y_true = [c.sentiment for c in comments]
    # 3) Obtener predicciones del modelo
    y_pred_labels = predict_uc.execute([c.content for c in comments])
    y_pred = [1 if sa.predicted_sentiment == 'Positivo' else 0 for sa in y_pred_labels]

    # 4) Generar y guardar matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo'])
    disp.plot()
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'confusion_matrix.png'))
    plt.close()

    # 5) Graficar evolución de exactitud por época
    plt.figure()
    plt.plot(history['accuracy'], label='Entrenamiento')
    plt.plot(history['val_accuracy'], label='Validación')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.title('Exactitud por Época')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'accuracy_plot.png'))
    plt.close()

    print('Diagramas generados en:', docs_dir)


if __name__ == '__main__':
    main()
