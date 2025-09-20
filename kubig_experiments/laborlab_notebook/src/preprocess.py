"""
DoWhy 데이터 전처리 모듈
- Basic 전처리: 기본적인 데이터 정제 및 변환
- NLP 전처리: 텍스트 데이터 처리 및 특성 추출
"""

import pandas as pd
import numpy as np

def load_and_preprocess_data(data_file, sheet_name=None, preprocessing_type='basic'):
    """
    데이터를 로드하고 전처리하는 함수
    
    Args:
        data_file (str): 데이터 파일 경로
        sheet_name (str): 엑셀 시트명 (선택사항)
        preprocessing_type (str): 전처리 타입 ('basic', 'nlp', 'both')
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    # 데이터 로드
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_file, sheet_name=sheet_name)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 사용하세요.")
    
    # 전처리 타입에 따라 처리
    if preprocessing_type == 'basic':
        df_processed = basic_preprocessing(df)
    elif preprocessing_type == 'nlp':
        df_processed = nlp_preprocessing(df)
    elif preprocessing_type == 'both':
        df_basic = basic_preprocessing(df)
        df_nlp = nlp_preprocessing(df)
        df_processed = pd.concat([df_basic, df_nlp], axis=1)
    else:
        raise ValueError("preprocessing_type은 'basic', 'nlp', 'both' 중 하나여야 합니다.")
    
    return df_processed

def basic_preprocessing(df):
    """
    기본적인 데이터 전처리를 수행하는 함수
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
    
    Returns:
        pd.DataFrame: 기본 전처리된 데이터프레임
    """
    df_processed = df.copy()
    
    # ACCR_CD를 숫자로 인코딩
    if 'ACCR_CD' in df_processed.columns:
        accr_mapping = {
            '고등학교': 1,
            '(2/3년제) 대학': 2,
            '4년제 대학': 3,
            '대학원': 4,
            '중퇴': 5
        }
        df_processed['ACCR_CD'] = df_processed['ACCR_CD'].map(accr_mapping)
    
    # 결측값 처리
    df_processed = df_processed.dropna(subset=['ACCR_CD', 'ACQ_180_YN'])
    
    # 숫자형 컬럼 정규화
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_processed[col].std() > 0:  # 표준편차가 0이 아닌 경우만
            df_processed[f'{col}_normalized'] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
    
    return df_processed

def nlp_preprocessing(df):
    """
    NLP 기반 데이터 전처리를 수행하는 함수
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
    
    Returns:
        pd.DataFrame: NLP 전처리된 데이터프레임
    """
    df_processed = df.copy()
    
    # 텍스트 컬럼 찾기
    text_columns = df_processed.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        if col in ['ACCR_CD']:  # 이미 처리된 컬럼 제외
            continue
            
        # 텍스트 길이 특성 추출
        df_processed[f'{col}_length'] = df_processed[col].astype(str).str.len()
        
        # 단어 수 특성 추출
        df_processed[f'{col}_word_count'] = df_processed[col].astype(str).str.split().str.len()
        
        # 대문자 비율 특성 추출
        df_processed[f'{col}_upper_ratio'] = df_processed[col].astype(str).apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # 숫자 포함 여부
        df_processed[f'{col}_has_number'] = df_processed[col].astype(str).str.contains(r'\d').astype(int)
        
        # 특수문자 포함 여부
        df_processed[f'{col}_has_special'] = df_processed[col].astype(str).str.contains(r'[!@#$%^&*(),.?":{}|<>]').astype(int)
    
    return df_processed

def validate_data(df):
    """데이터 유효성을 검증하는 함수"""
    return True

def get_data_info(df):
    """데이터 정보를 반환하는 함수"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    return info
