"""
相似度計算基類
職責：定義 similarity 介面（可替換）
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict


class SimilarityBase(ABC):
    """
    相似度計算的抽象基類
    
    所有相似度計算方法都應繼承此類
    """
    
    @abstractmethod
    def fit(self, reference_vectors: np.ndarray, labels: np.ndarray = None):
        """
        訓練/擬合相似度模型
        
        Args:
            reference_vectors: 參考特徵向量 (N, D)，N=樣本數，D=特徵維度
            labels: 可選的標籤（用於監督學習或加權）
        """
        pass
    
    @abstractmethod
    def find_similar(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        找到最相似的 k 個樣本
        
        Args:
            query_vector: 查詢特徵向量 (D,)
            k: 返回最相似的樣本數
            
        Returns:
            (indices, distances): 
                - indices: 相似樣本的索引列表
                - distances: 對應的距離/相似度列表
        """
        pass
    
    @abstractmethod
    def compute_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        計算兩個向量之間的相似度
        
        Args:
            vector1: 第一個向量
            vector2: 第二個向量
            
        Returns:
            相似度分數
        """
        pass
    
    def find_similar_batch(self, query_vectors: np.ndarray, k: int = 5) -> List[Tuple[List[int], List[float]]]:
        """
        批量找相似樣本
        
        Args:
            query_vectors: 查詢向量集合 (M, D)
            k: 每個查詢返回的樣本數
            
        Returns:
            結果列表，每個元素為 (indices, distances) 元組
        """
        results = []
        for query in query_vectors:
            indices, distances = self.find_similar(query, k)
            results.append((indices, distances))
        return results
