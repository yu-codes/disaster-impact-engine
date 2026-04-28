"""
數據集構建腳本
職責：預處理原始數據，生成可用於模型的數據集
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse


def generate_sample_typhoon_data(output_path: str, num_typhoons: int = 30):
    """
    生成範例颱風數據（用於測試）
    
    預期格式：
    - typhoon_id: 颱風編號
    - date: 時間
    - lat: 緯度
    - lon: 經度
    - max_wind: 最大風速 (km/h)
    - central_pressure: 中心氣壓 (hPa)
    
    Args:
        output_path: 輸出文件路徑
        num_typhoons: 生成的颱風數量
    """
    data = []
    base_date = datetime(2015, 1, 1)
    
    for t_id in range(1, num_typhoons + 1):
        typhoon_id = f"TYPHOON_{t_id:03d}"
        
        # 每個颱風有 10-20 個時間步
        num_steps = np.random.randint(10, 21)
        
        # 初始位置（離台灣較遠）
        init_lat = np.random.uniform(15, 25)
        init_lon = np.random.uniform(110, 130)
        
        # 初始風速和氣壓
        init_wind = np.random.uniform(100, 200)
        init_pressure = np.random.uniform(920, 980)
        
        for step in range(num_steps):
            # 模擬颱風逼近台灣
            progress = step / num_steps
            lat = init_lat + progress * 8
            lon = init_lon + progress * 10
            
            # 颱風隨時間增強或減弱
            wind = init_wind + np.random.uniform(-5, 10) * (1 - progress)
            pressure = init_pressure - np.random.uniform(0, 20) * progress
            
            date = base_date + timedelta(hours=t_id * 100 + step * 6)
            
            data.append({
                'typhoon_id': typhoon_id,
                'date': date.strftime('%Y-%m-%d %H:%M'),
                'lat': round(lat, 2),
                'lon': round(lon, 2),
                'max_wind': round(wind, 1),
                'central_pressure': round(pressure, 1)
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ 颱風數據已生成: {output_path}")
    print(f"  - 颱風數: {num_typhoons}")
    print(f"  - 總記錄數: {len(df)}")
    
    return df


def generate_sample_impact_data(typhoon_ids: list, output_path: str):
    """
    生成範例災害數據
    
    預期格式：
    - typhoon_id: 颱風編號
    - impact_type: 災害類型 (flooding/blackout/damage/windfall)
    - severity: 嚴重程度 (0-4)
    
    Args:
        typhoon_ids: 颱風 ID 列表
        output_path: 輸出文件路徑
    """
    impact_types = ['flooding', 'blackout', 'damage', 'windfall']
    data = []
    
    for typhoon_id in typhoon_ids:
        # 每個颱風可能有多種災害
        num_impacts = np.random.randint(0, 4)
        
        if num_impacts == 0:
            # 沒有災害
            data.append({
                'typhoon_id': typhoon_id,
                'impact_type': 'none',
                'severity': 0
            })
        else:
            # 隨機選擇災害類型
            selected_impacts = np.random.choice(impact_types, num_impacts, replace=False)
            
            for impact_type in selected_impacts:
                severity = np.random.randint(1, 5)  # 1-4 (0 = 無)
                
                data.append({
                    'typhoon_id': typhoon_id,
                    'impact_type': impact_type,
                    'severity': severity
                })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ 災害數據已生成: {output_path}")
    print(f"  - 颱風數: {len(df[df['impact_type'] != 'none']['typhoon_id'].unique())}")
    print(f"  - 總記錄數: {len(df)}")
    
    return df


def clean_typhoon_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理颱風數據
    - 移除重複
    - 填補缺失值
    - 驗證範圍
    """
    # 移除重複
    df = df.drop_duplicates(subset=['typhoon_id', 'date'])
    
    # 按日期排序
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['typhoon_id', 'date'])
    
    # 填補缺失的風速（使用前一個值）
    df['max_wind'] = df.groupby('typhoon_id')['max_wind'].fillna(method='ffill')
    df['central_pressure'] = df.groupby('typhoon_id')['central_pressure'].fillna(method='ffill')
    
    # 移除無效行
    df = df.dropna(subset=['max_wind', 'central_pressure'])
    
    return df


def validate_data(typhoon_df: pd.DataFrame, impact_df: pd.DataFrame) -> bool:
    """驗證數據完整性"""
    print("\n📊 數據驗證:")
    
    # 檢查必要列
    required_cols_typhoon = ['typhoon_id', 'date', 'lat', 'lon', 'max_wind']
    if not all(col in typhoon_df.columns for col in required_cols_typhoon):
        print("  ✗ 颱風數據缺少必要列")
        return False
    
    required_cols_impact = ['typhoon_id', 'impact_type', 'severity']
    if not all(col in impact_df.columns for col in required_cols_impact):
        print("  ✗ 災害數據缺少必要列")
        return False
    
    # 檢查数据範圍
    if not (typhoon_df['max_wind'] > 0).all():
        print("  ✗ 風速數據異常")
        return False
    
    if not (impact_df['severity'].between(0, 4)).all():
        print("  ✗ 嚴重程度超出範圍")
        return False
    
    print(f"  ✓ 颱風記錄: {len(typhoon_df)}")
    print(f"  ✓ 災害記錄: {len(impact_df)}")
    print(f"  ✓ 所有驗證通過")
    
    return True


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='構建颱風災害預測數據集')
    parser.add_argument('--num-typhoons', type=int, default=30, help='生成的颱風數')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='輸出目錄')
    parser.add_argument('--clean', action='store_true', help='執行數據清理')
    
    args = parser.parse_args()
    
    # 建立輸出目錄
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成數據
    print("🔨 開始構建數據集...\n")
    
    typhoon_path = f"{args.output_dir}/typhoon.csv"
    impact_path = f"{args.output_dir}/impact.csv"
    
    typhoon_df = generate_sample_typhoon_data(typhoon_path, args.num_typhoons)
    typhoon_ids = typhoon_df['typhoon_id'].unique().tolist()
    impact_df = generate_sample_impact_data(typhoon_ids, impact_path)
    
    # 數據清理
    if args.clean:
        print("\n🧹 清理數據...")
        typhoon_df = clean_typhoon_data(typhoon_df)
        typhoon_df.to_csv(typhoon_path, index=False)
    
    # 驗證
    validate_data(typhoon_df, impact_df)
    
    print("\n✅ 數據集構建完成！")


if __name__ == '__main__':
    main()
