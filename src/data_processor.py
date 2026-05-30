"""데이터 전처리 모듈"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import signal
from scipy.interpolate import interp1d


class DataProcessor:
    """센서 데이터 전처리 클래스"""
    
    def __init__(self, config: Dict):
        """
        데이터 처리기 초기화
        
        Args:
            config: 전처리 설정
        """
        self.config = config
    
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        데이터 정규화
        
        Args:
            data: 입력 데이터
            method: 정규화 방법 ('minmax' 또는 'zscore')
        
        Returns:
            정규화된 데이터
        """
        if method == 'minmax':
            return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        elif method == 'zscore':
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def remove_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        이상치 제거 (Z-score 기반)
        
        Args:
            data: 입력 데이터
            threshold: Z-score 임계값
        
        Returns:
            이상치 제거된 데이터
        """
        z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-8))
        return data[z_scores < threshold]
    
    def smooth_data(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        이동 평균을 이용한 데이터 평활
        
        Args:
            data: 입력 데이터
            window_size: 윈도우 크기
        
        Returns:
            평활된 데이터
        """
        if window_size < 3:
            return data
        
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode='same')
        return smoothed
    
    def fill_missing_values(self, data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """
        결측치 채우기
        
        Args:
            data: 입력 데이터
            method: 채우기 방법 ('interpolate' 또는 'forward_fill')
        
        Returns:
            결측치가 채워진 데이터
        """
        df = pd.DataFrame({'value': data})
        
        if method == 'interpolate':
            df['value'] = df['value'].interpolate(method='linear')
        elif method == 'forward_fill':
            df['value'] = df['value'].fillna(method='ffill')
        
        return df['value'].values
    
    def apply_filter(self, data: np.ndarray, filter_type: str = 'lowpass', 
                     cutoff_freq: float = 0.1, order: int = 4) -> np.ndarray:
        """
        디지털 필터 적용
        
        Args:
            data: 입력 데이터
            filter_type: 필터 유형 ('lowpass', 'highpass', 'bandpass')
            cutoff_freq: 컷오프 주파수
            order: 필터 차수
        
        Returns:
            필터링된 데이터
        """
        b, a = signal.butter(order, cutoff_freq, btype=filter_type)
        return signal.filtfilt(b, a, data)
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임 전처리 (전체 파이프라인)
        
        Args:
            df: 원본 데이터프레임
        
        Returns:
            전처리된 데이터프레임
        """
        df_processed = df.copy()
        
        # 센서 데이터 열 선택
        sensor_columns = ['gas', 'temperature', 'humidity']
        
        for col in sensor_columns:
            if col not in df_processed.columns:
                continue
            
            data = df_processed[col].values
            
            # 1. 결측치 채우기
            if self.config.get('fill_missing', True):
                data = self.fill_missing_values(
                    data, 
                    method=self.config.get('missing_method', 'interpolate')
                )
            
            # 2. 이상치 제거
            if self.config.get('remove_outliers', True):
                # 이상치를 NaN으로 표시한 후 보간
                threshold = self.config.get('outlier_threshold', 3.0)
                z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-8))
                data[z_scores > threshold] = np.nan
                data = self.fill_missing_values(data, method='interpolate')
            
            # 3. 평활
            if self.config.get('smooth_data', True):
                window_size = self.config.get('smoothing_window', 5)
                data = self.smooth_data(data, window_size)
            
            # 4. 정규화
            if self.config.get('normalize', True):
                method = self.config.get('normalization_method', 'minmax')
                data = self.normalize(data, method)
            
            df_processed[col] = data
        
        return df_processed
    
    def create_sliding_windows(self, data: np.ndarray, window_size: int, 
                               step_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        슬라이딩 윈도우 생성
        
        Args:
            data: 입력 데이터
            window_size: 윈도우 크기
            step_size: 슬라이딩 스텝 크기
        
        Returns:
            (윈도우 데이터, 인덱스)
        """
        windows = []
        indices = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            windows.append(data[i:i+window_size])
            indices.append(i)
        
        return np.array(windows), np.array(indices)
    
    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        학습/테스트 데이터 분할
        
        Args:
            df: 입력 데이터프레임
            test_size: 테스트 데이터 비율
            random_state: 난수 시드
        
        Returns:
            (학습 데이터, 테스트 데이터)
        """
        np.random.seed(random_state)
        indices = np.random.permutation(len(df))
        split_idx = int(len(df) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        return df.iloc[train_indices], df.iloc[test_indices]
