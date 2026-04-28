"""
數據加載模組
職責：讀取颱風 / 災害資料
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DataLoader:
    """
    加載颱風和災害數據
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        初始化數據加載器
        
        Args:
            data_dir: 原始數據目錄路徑
        """
        self.data_dir = Path(data_dir)
        self.typhoon_data = None
        self.impact_data = None
    
    def load_typhoon_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        加載颱風軌跡數據
        
        預期列：
            - typhoon_id: 颱風編號
            - date: 時間
            - lat: 緯度
            - lon: 經度
            - max_wind: 最大風速 (km/h)
            - central_pressure: 中心氣壓 (hPa)
        
        Args:
            filepath: 文件路徑（若不指定，使用預設）
            
        Returns:
            颱風數據 DataFrame
        """
        if filepath is None:
            filepath = self.data_dir / "typhoon.csv"
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"颱風數據文件不存在: {filepath}")
        
        self.typhoon_data = pd.read_csv(filepath)
        self._validate_typhoon_data()
        return self.typhoon_data
    
    def load_impact_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        加載災害影響數據
        
        預期列：
            - typhoon_id: 颱風編號
            - impact_type: 災害類型 (flooding/blackout/damage/...)
            - severity: 嚴重程度 (0-5)
            - affected_areas: 受影響地區
            - economic_loss: 經濟損失
        
        Args:
            filepath: 文件路徑（若不指定，使用預設）
            
        Returns:
            災害數據 DataFrame
        """
        if filepath is None:
            filepath = self.data_dir / "impact.csv"
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"災害數據文件不存在: {filepath}")
        
        self.impact_data = pd.read_csv(filepath)
        self._validate_impact_data()
        return self.impact_data
    
    def _validate_typhoon_data(self):
        """驗證颱風數據格式"""
        required_cols = ['typhoon_id', 'date', 'lat', 'lon', 'max_wind']
        missing = [col for col in required_cols if col not in self.typhoon_data.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")
    
    def _validate_impact_data(self):
        """驗證災害數據格式"""
        required_cols = ['typhoon_id', 'impact_type', 'severity']
        missing = [col for col in required_cols if col not in self.impact_data.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")
    
    def get_typhoon_by_id(self, typhoon_id: str) -> pd.DataFrame:
        """
        根據颱風ID取得完整軌跡
        
        Args:
            typhoon_id: 颱風編號
            
        Returns:
            該颱風的所有時間步數據
        """
        if self.typhoon_data is None:
            raise ValueError("尚未加載颱風數據")
        
        return self.typhoon_data[self.typhoon_data['typhoon_id'] == typhoon_id].copy()
    
    def get_impact_by_typhoon(self, typhoon_id: str) -> pd.DataFrame:
        """
        取得特定颱風的災害數據
        
        Args:
            typhoon_id: 颱風編號
            
        Returns:
            該颱風的災害記錄
        """
        if self.impact_data is None:
            raise ValueError("尚未加載災害數據")
        
        return self.impact_data[self.impact_data['typhoon_id'] == typhoon_id].copy()
    
    def get_all_typhoon_ids(self) -> List[str]:
        """取得所有颱風ID"""
        if self.typhoon_data is None:
            raise ValueError("尚未加載颱風數據")
        
        return sorted(self.typhoon_data['typhoon_id'].unique().tolist())
    
    def save_processed_data(self, data: pd.DataFrame, filename: str, output_dir: str = "data/processed"):
        """
        保存處理後的數據
        
        Args:
            data: 要保存的 DataFrame
            filename: 文件名
            output_dir: 輸出目錄
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        data.to_csv(filepath, index=False)
        print(f"✓ 數據已保存: {filepath}")
