"""
DoWhy 데이터 전처리 모듈
- Basic 전처리: 기본적인 데이터 정제 및 변환
- NLP 전처리: 텍스트 데이터 처리 및 특성 추출

사용 예시:
    # JSON 파일들에 대해 각각 다른 전처리 적용
    preprocessor = Preprocessor([])
    
    # 방법 1: 개별 파일 처리
    resume_data = preprocessor.load_and_preprocess_data('resume.json', json_name='이력서')
    cover_letter_data = preprocessor.load_and_preprocess_data('cover_letter.json', json_name='자기소개서')
    
    # Excel 파일 처리
    excel_data = preprocessor.load_and_preprocess_data('data.xlsx', sheet_name='Sheet1')
    
    # 방법 2: 여러 파일을 한번에 처리
    file_list = ['resume.json', 'cover_letter.json', 'training.json', 'certification.json']
    json_names = ['이력서', '자기소개서', '직업훈련', '자격증']
    merged_df = preprocessor.get_merged_df(file_list, json_names=json_names)
"""

import pandas as pd
import numpy as np
import json



class Preprocessor:
    def __init__(self, df_list):
        self.json_names = ['이력서', '자기소개서', '직업훈련', '자격증']
        self.df_list = []
        self.variable_mapping = self.load_variable_mapping()

    def load_variable_mapping(self):
        with open('../data/variable_mapping.json', encoding='utf-8') as f:
            variable_mapping = json.load(f)
        return variable_mapping

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

    def basic_preprocessing(self, df):
        """
        기본적인 데이터 전처리를 수행하는 함수
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
        
        Returns:
            pd.DataFrame: 기본 전처리된 데이터프레임
        """

        df_processed = df
        return df_processed

    def nlp_preprocessing(self, data, json_name=None):
        """
        NLP 기반 데이터 전처리를 수행하는 함수
        
        Args:
            data: json 파일 (자기소개서, 이력서, 직업훈련, 자격증)
            json_name (str): JSON 데이터 타입에 따라 다른 전처리 적용
        Returns:
            pd.DataFrame: NLP 전처리된 데이터프레임
        """
        
        # JSON 데이터 타입에 따른 특화된 전처리
        if json_name == '이력서':
            df_processed = self._preprocess_resume(data)
        elif json_name == '자기소개서':
            df_processed = self._preprocess_cover_letter(data)
        elif json_name == '직업훈련':
            df_processed = self._preprocess_training(data)
        elif json_name == '자격증':
            df_processed = self._preprocess_certification(data)
        else:
            raise ValueError(f"지원하지 않는 json 파일입니다. {json_name}")
        
        return df_processed


    def _preprocess_resume(self, data):
        """이력서 특화 전처리"""
        df_processed = data
        return df_processed

    def _preprocess_cover_letter(self, data):
        """자기소개서 특화 전처리"""
        df_processed = data
        return df_processed

    def _preprocess_training(self, data):
        """직업훈련 특화 전처리""" 
        df_processed = data
        return df_processed

    def _preprocess_certification(self, data):
        """자격증 특화 전처리"""
        df_processed = data
        return df_processed


    def load_and_preprocess_data(self, data_file, sheet_name=None, json_name=None):
        """
        데이터를 로드하고 전처리하는 함수
        
        Args:
            data_file (str): 데이터 파일 경로
            sheet_name (str): 엑셀 시트명 (Excel 파일용)
            json_name (str): JSON 데이터 타입 ('이력서', '자기소개서', '직업훈련', '자격증')
        
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        # 데이터 로드
        if data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
            data_processed = self.basic_preprocessing(data)
        elif data_file.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(data_file, sheet_name=sheet_name)
            data_processed = self.basic_preprocessing(data)
        elif data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON 파일의 경우 json_name을 데이터 타입으로 사용
                data_processed = self.nlp_preprocessing(data, json_name=json_name)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. CSV, Excel 또는 JSON 파일을 사용하세요.")
        
        return data_processed


    def get_merged_df(self, file_list, sheet_name=None):
        """
        파일명 리스트를 받아 각 파일을 load_and_preprocess_data로 읽고 self.df_list에 append,
        이후 SEEK_CUST_NO 컬럼 기준으로 순차적으로 조인하여 데이터프레임 반환

        Args:
            file_list (list): 파일명(str) 리스트
            sheet_names (list): 각 파일에 대응하는 시트명 리스트 (Excel 파일용, 선택사항)
            json_names (list): 각 파일에 대응하는 JSON 데이터 타입 리스트 (JSON 파일용, 선택사항)
                              - ['이력서', '자기소개서', '직업훈련', '자격증'] 등

        Returns:
            pd.DataFrame: SEEK_CUST_NO 기준으로 조인된 데이터프레임 -> repeat 처리 필요
        """
        self.df_list = []
        result = None
        
        for idx, file in enumerate(file_list):
            # 각 파일에 대응하는 시트명과 JSON명 사용       
            current_json_name = self.json_names[idx]
            df = self.load_and_preprocess_data(file, sheet_name=sheet_name, json_name=current_json_name)
            self.df_list.append(df)
            
            if idx == 0:
                result = df
            else:
                result = result.merge(df, on="SEEK_CUST_NO", how="outer", suffixes=('', f'_df{idx+1}'))
        
        return result