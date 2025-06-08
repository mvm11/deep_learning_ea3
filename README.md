# Evidencia de Aprendizaje 3: Implementación Básica de una Red Neuronal

## 📖 Caso de Estudio

Una agencia de marketing digital desea analizar los comentarios de los clientes para identificar opiniones positivas y negativas. Utilizando una Red Neuronal Recurrente (RNN) y herramientas de análisis de texto, mejoraremos la comprensión de los sentimientos expresados y proporcionaremos recomendaciones para incrementar las ventas.

---

## 🎯 Objetivo

Implementar una red neuronal básica (CNN + RNN) que clasifique sentimientos en comentarios de texto, siguiendo el informe de la EA2 y cumpliendo los criterios de la EA3.

---

## 📚 Instrucciones Generales

1. **Lee** el material de la Unidad 3 con atención.
2. **Selecciona** el framework: TensorFlow o PyTorch.
3. **Programa** la red neuronal considerando:

   * Importación de librerías necesarias.
   * Configuración de capas (número de filtros, tamaño de filtros, activación).
   * Capa de pooling.
   * Compilador (`model.compile`) o `forward` (PyTorch), incluyendo optimizador y función de pérdida.
4. **Imprime** la topología de la red con `summary()` (TensorFlow) o `print(model)` (PyTorch).
5. **Construye** un archivo `.py` (o `.ipynb`) con el código completo.
6. **Documenta** cada línea de código en tus propias palabras.

---

## 🗂 Estructura de Archivos

```
EA3/
├─ src/
│  ├─ application/usecase/
│  │  ├─ PredictSentimentUseCase.py
│  │  └─ TrainSentimentModelUseCase.py
│  ├─ domain/model/
│  │  ├─ Comment.py
│  │  ├─ SentimentAnalysis.py
│  │  └─ TrainingResult.py
│  ├─ infrastructure/model/
│  │  └─ cnn_sentiment_model.py  ← Código de la CNN+RNN documentado
│  └─ infrastructure/repository/
│     ├─ CSVCommentRepository.py
│     └─ TensorFlowModelRepository.py
├─ main.py                      ← Ejecución, summary, entrenamiento, métricas
├─ docs/
│  ├─ confusion_matrix.png      ← Matriz de confusión (generada)
│  ├─ accuracy_plot.png         ← Gráfica de exactitud por época
└─ README.md                    ← Este archivo
```

---

## 🚀 Cómo Ejecutar

1. Crear y activar el entorno virtual:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar el script de arquitectura:

   ```bash
   python src/infrastructure/model/cnn_sentiment_model.py
   ```
4. Correr flujo principal (summary, entrenamiento, predicción, métricas):

   ```bash
   python main.py
   ```

---
