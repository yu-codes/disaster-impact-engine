"""
KNN 相似度計算實現
職責：使用 K-Nearest Neighbors 找相似颱風
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from .base import SimilarityBase


class KNNSimilarity(SimilarityBase):
    """
    KNN 相似度計算
    
    使用歐式距離或餘弦相似度找 K 個最近鄰
    """
    
    def __init__(self, metric: str = 'euclidean'):
        """
        初始化 KNN 相似度計算器
        
        Args:
            metric: 距離度量 ('euclidean' 或 'cosine')
        """
        if metric not in ['euclidean', 'cosine']:
            raise ValueError(f"不支援的度量: {metric}")
        
        self.metric = metric
        self.reference_vectors = None
        self.labels = None
    
    def fit(self, reference_vectors: np.ndarray, labels: np.ndarray = None):
        """
        儲存參考向量（KNN 不需要訓練）
        
        Args:
            reference_vectors: 參考特徵向量 (N, D)
            labels: 可選的標籤
        """
        if not isinstance(reference_vectors, np.ndarray):
            reference_vectors = np.array(reference_vectors)
        
        if reference_vectors.ndim != 2:
            raise ValueError(f"期望 2D 陣列，得到 {reference_vectors.ndim}D")
        
        self.reference_vectors = reference_vectors
        self.labels = labels
    
    def find_similar(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        找最相似的 k 個樣本
        
        Args:
            query_vector: 查詢向量 (D,)
            k: 返回的樣本數
            
        Returns:
            (indices, distances): 索引和距離
        """
        if self.reference_vectors is None:
            raise ValueError("尚未擬合參考向量")
        
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
        
        query_vector = query_vector.reshape(1, -1)
        
        # 計算距離
        if self.metric == 'euclidean':
            distances = euclidean_distances(query_vector, self.reference_vectors)[0]
        else:  # cosine
            distances = cosine_distances(query_vector, self.reference_vectors)[0]
        
        # 取最小的 k 個
        k = min(k, len(distances))
        indices = np.argsort(distances)[:k]
        top_distances = distances[indices]
        
        return indices.tolist(), top_distances.tolist()
    
    def compute_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        計算兩個向量的距離
        
        Args:
            vector1: 第一個向量
            vector2: 第二個向量
            
        Returns:
            距離（0 表示完全相似）
        """
        if not isinstance(vector1, np.ndarray):
            vector1 = np.array(vector1)
        if not isinstance(vector2, np.ndarray):
            vector2 = np.array(vector2)
        
        if self.metric == 'euclidean':
            distance = np.linalg.norm(vector1 - vector2)
        else:  # cosine
            # 餘弦相似度需要轉換為距離
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([vector1], [vector2])[0][0]
            distance = 1 - similarity
        
        return float(distance)
    
    def find_similar_with_names(self, query_vector: np.ndarray, k: int = 5, 
                                reference_names: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        找相似樣本並返回名稱（如 typhoon_id）
        
        Args:
            query_vector: 查詢向量
            k: 返回的樣本數
            reference_names: 參考樣本的名稱列表（如颱風ID）
            
        Returns:
            [(name, distance), ...] 列表
        """
        indices, distances = self.find_similar(query_vector, k)
        
        if reference_names is None:
            reference_names = [f"sample_{i}" for i in range(len(self.reference_vectors))]
        
        results = [(reference_names[idx], dist) for idx, dist in zip(indices, distances)]
        return results
