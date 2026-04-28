"""相似度計算模組初始化"""
from .base import SimilarityBase
from .knn import KNNSimilarity

__all__ = ['SimilarityBase', 'KNNSimilarity']
