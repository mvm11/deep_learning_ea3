

import re
import pandas as pd
from typing import List
from domain.repository.ICommentRepository import ICommentRepository
from domain.model.Comment import Comment



class CSVCommentRepository(ICommentRepository):
    
    def __init__(self):
        self._positive_words = [
            'excelente', 'bueno', 'efectivo', 'profesional', 'atractivo', 
            'perfecto', 'encantó', 'ayudaron', 'fácil', 'mejorar', 'eficiente'
        ]
        self._negative_words = [
            'nunca', 'llegó', 'demorado', 'lento', 'dañado', 'tarde', 'malo', 
            'terrible', 'pésimo', 'problema'
        ]
    
    def load_comments(self, file_path: str) -> List[Comment]:
        """Carga comentarios desde archivo CSV"""
        try:
            df = pd.read_csv(file_path)
            
            if 'comment' not in df.columns:
                print(f"Error: No se encontró la columna 'comment' en {file_path}")
                return []
            
            comments = []
            for _, row in df.iterrows():
                if pd.notna(row['comment']):
                    comment_text = str(row['comment'])
                    sentiment = self._label_sentiment(comment_text)
                    comments.append(Comment(content=comment_text, sentiment=sentiment))
            
            return comments
            
        except Exception as e:
            print(f"Error al cargar archivo CSV: {e}")
            return []
    
    def save_comments(self, comments: List[Comment], file_path: str) -> bool:
        """Guarda comentarios en archivo CSV"""
        try:
            data = [{
                'comment': comment.content,
                'sentiment': comment.sentiment,
                'confidence': comment.confidence
            } for comment in comments]
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            return True
            
        except Exception as e:
            print(f"Error al guardar archivo CSV: {e}")
            return False
    
    def get_sample_comments(self, count: int) -> List[Comment]:
        """Proporciona comentarios de ejemplo para pruebas"""
        sample_texts = [
            "El pedido nunca llegó.",
            "Demoraron demasiado en contestar mis dudas.",
            "El diseño de la campaña fue atractivo y eficiente.",
            "La campaña publicitaria fue muy efectiva.",
            "El equipo fue muy profesional y atento.",
            "El servicio al cliente fue muy lento.",
            "Me encantó el producto, llegó en perfectas condiciones.",
            "Excelente servicio y atención al cliente.",
            "El producto llegó dañado y tarde.",
            "Me ayudaron a mejorar mis ventas en poco tiempo.",
            "La página es muy fácil de navegar."
        ]
        
        # Repetir para alcanzar el count solicitado
        expanded_texts = (sample_texts * (count // len(sample_texts) + 1))[:count]
        
        comments = []
        for text in expanded_texts:
            sentiment = self._label_sentiment(text)
            comments.append(Comment(content=text, sentiment=sentiment))
        
        return comments
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocesa el texto limpiando y normalizando"""
        text = text.lower()
        text = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _label_sentiment(self, comment: str) -> int:
        """Etiqueta automáticamente los sentimientos basado en palabras clave"""
        comment_lower = comment.lower()
        positive_score = sum(1 for word in self._positive_words if word in comment_lower)
        negative_score = sum(1 for word in self._negative_words if word in comment_lower)
        
        if positive_score > negative_score:
            return 1  # Positivo
        elif negative_score > positive_score:
            return 0  # Negativo
        else:
            # En caso de empate, usar palabras más específicas
            if any(word in comment_lower for word in ['excelente', 'encantó', 'perfecto']):
                return 1
            elif any(word in comment_lower for word in ['nunca llegó', 'dañado', 'terrible']):
                return 0
            else:
                return 1  # Por defecto positivo si es neutral