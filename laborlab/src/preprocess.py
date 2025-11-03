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
import re
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import Counter

from llm_reference import (
    JSON_NAMES, RESUME_SECTIONS, SUPPORTED_SECTIONS, 
    DEFAULT_MAX_COVER_LEN, DEFAULT_COVER_EXCEED_RATIO, DEFAULT_DATE_FORMAT
)
from llm_scorer import LLMScorer



class Preprocessor:
    def __init__(self, df_list, api_key=None):
        self.json_names = JSON_NAMES
        self.sheet_name = '구직인증 관련 데이터'
        self.df_list = []
        self.variable_mapping = self.load_variable_mapping()
        self.llm_scorer = LLMScorer(api_key)
        self.hope_jscd1_map = {}  # SEEK_CUST_NO -> HOPE_JSCD1 매핑 저장
        self.job_code_to_name = self.load_job_mapping()  # 소분류코드 -> 소분류명 매핑

    def load_variable_mapping(self):
        with open('../data/variable_mapping.json', encoding='utf-8') as f:
            variable_mapping = json.load(f)
        return variable_mapping
    
    def load_job_mapping(self):
        """job_subcategories.csv를 로드하여 소분류코드 -> 소분류명 매핑 생성"""
        try:
            job_df = pd.read_csv('../data/fixed_data/job_subcategories.csv', encoding='utf-8')
            # 소분류코드를 문자열로 변환하여 딕셔너리 생성
            job_mapping = dict(zip(job_df['소분류코드'].astype(str).str.zfill(3), job_df['소분류명']))
            return job_mapping
        except Exception as e:
            print(f"job_subcategories.csv 로드 실패: {e}")
            return {}
    
    def get_job_name_from_code(self, code):
        """HOPE_JSCD1 코드를 직종명으로 변환"""
        if not code:
            return "미상"
        # 코드를 문자열로 변환하고 앞에 0을 채워서 3자리로 만들기
        code_str = str(code).zfill(3)
        return self.job_code_to_name.get(code_str, f"직종코드 {code}")

    @staticmethod
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
        # variable_mapping.json의 structured_data 키만 사용
        structured_keys = set(self.variable_mapping.get("structured_data", {}).keys())
        
        # 원본 데이터에서 해당 변수들만 필터링 (존재하는 변수만)
        available_vars = list(structured_keys & set(df.columns))
        missing_vars = list(structured_keys - set(df.columns))
        
        if missing_vars:
            print(f"다음 변수들이 데이터에 없습니다: {missing_vars}")
        
        df = df[available_vars]

        # BFR_OCTR_YN 제거, BFR_OCTR_CT만 유지
        if "BFR_OCTR_YN" in df.columns and "BFR_OCTR_CT" in df.columns:
            df = df.drop(columns=["BFR_OCTR_YN"])

        # 8개 예/아니오 변수 → 합쳐서 새로운 순서형 범주 변수 생성
        agree_vars = [
            "EMAIL_RCYN", "SAEIL_CNTC_AGRE_YN", "SHRS_IDIF_AOFR_YN", "SULC_IDIF_AOFR_YN",
            "IDIF_IQRY_AGRE_YN", "SMS_RCYN", "EMAIL_OTPB_YN", "MPNO_OTPB_YN"
        ]

        # 존재하는 경우만 사용
        agree_vars = [col for col in agree_vars if col in df.columns]

        if agree_vars:
            agree_count = (df[agree_vars] == "예").sum(axis=1)
            df["AGREE_LEVEL"] = agree_count.apply(lambda x: "하" if x <= 2 else ("중" if x <= 5 else "상"))
            df_processed = df.drop(columns=agree_vars)

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
        seek_id = data.get("SEEK_CUST_NO", "")
        
        # BASIC_RESUME_YN == "Y"인 resume 찾기
        resumes = data.get("RESUMES", [])
        basic_resume = None
        for resume in resumes:
            if str(resume.get("BASIC_RESUME_YN", "")).upper() == "Y":
                basic_resume = resume
                break
        
        # 기본 이력서가 없으면 빈 결과 반환
        if basic_resume is None:
            return pd.DataFrame([{
                "SEEK_CUST_NO": seek_id,
                "resume_score": None,
                "items_num": 0
            }])
        
        # ITEMS 가져오기
        items = basic_resume.get("ITEMS", [])
        items_num = len(items)
        
        # variable_mapping에서 resume 섹션 가져오기
        resume_mapping = self.variable_mapping.get("resume", {})
        
        # ITEMS를 포매팅
        formatting_sentence = ""
        for item in items:
            for key, value in item.items():
                # variable_mapping에서 한글 변수명 찾기
                if key in resume_mapping:
                    korean_key = resume_mapping[key].get("변수명", key)
                else:
                    korean_key = key
                
                # value가 None이면 빈 문자열로 처리
                value_str = str(value) if value is not None else ""
                formatting_sentence += f"{korean_key}: {value_str}\n"
            formatting_sentence += "\n"
        
        # 포매팅된 텍스트가 비어있으면 기본값 설정
        if not formatting_sentence.strip():
            formatting_sentence = "정보 없음"
        
        # HOPE_JSCD1 정보 가져와서 직종명으로 변환
        hope_jscd1 = self.hope_jscd1_map.get(seek_id, "")
        job_name = self.get_job_name_from_code(hope_jscd1)
        job_examples = []  # 필요시 HOPE_JSCD1로부터 직종 예시 리스트 생성 가능
        
        # LLM scorer에 전달하여 점수 계산
        score, _ = self.llm_scorer.score(selection="이력서", job_name=job_name, job_examples=job_examples, text=formatting_sentence)
        
        return pd.DataFrame([{
            "SEEK_CUST_NO": seek_id,
            "resume_score": score,
            "items_num": items_num
        }])


    def _preprocess_cover_letter(self, data):
        """자기소개서 특화 전처리"""
        if not isinstance(data, list):
            data = [data]
        
        rows = []
        for item in data:
            seek_id = item.get("SEEK_CUST_NO", "")
            if not seek_id:
                continue
                
            # 자기소개서 데이터 추출 (BASS_SFID_YN == "Y"인 항목만)
            texts = []
            items = []
            for c in item.get("COVERLETTERS", []):
                if str(c.get("BASS_SFID_YN", "")).upper() == "Y":
                    items = c.get("ITEMS", []) or []
                    for it in items:
                        t = it.get("SELF_INTRO_CONT", "")
                        if t:
                            texts.append(t.strip())
                    break
            
            full_text = "\n\n".join(texts) if texts else "정보 없음"
            
            # HOPE_JSCD1 정보 가져와서 직종명으로 변환
            hope_jscd1 = self.hope_jscd1_map.get(seek_id, "")
            job_name = self.get_job_name_from_code(hope_jscd1)
            job_examples = []  # 필요시 HOPE_JSCD1로부터 직종 예시 리스트 생성 가능
            
            # 점수 계산
            score, _ = self.llm_scorer.score("자기소개서", job_name, job_examples, full_text)
            
            # 오탈자 수 계산 (TYPO_CHECK 프롬프트 사용)
            typo_count = self.llm_scorer.count_typos(full_text)
            
            # score와 오탈자 수만 반환
            rows.append({
                "SEEK_CUST_NO": seek_id,
                "cover_letter_score": score,
                "오탈자 수": typo_count
            })
        
        return pd.DataFrame(rows)


    def _preprocess_training(self, data):
        """직업훈련 특화 전처리"""
        if not isinstance(data, list):
            data = [data]
        
        rows = []
        for item in data:
            seek_id = item.get("SEEK_CUST_NO", "")
            if not seek_id:
                continue
            
            # 구직인증 일자 가져오기
            jhcr_de = item.get("JHCR_DE", "")  # 구직인증 일자
            
            # TRAININGS에서 훈련 데이터 추출
            trainings = item.get("TRAININGS", [])
            
            # TRAININGS에서 모든 TRNG_ENDE 가져와서 datetime 객체 리스트로 변환
            training_end_dates = []
            for tr in trainings:
                trng_ende = tr.get("TRNG_ENDE", "").strip()
                if trng_ende:
                    try:
                        # 날짜 문자열을 datetime 객체로 변환
                        date_obj = datetime.strptime(trng_ende, DEFAULT_DATE_FORMAT)
                        training_end_dates.append(date_obj)
                    except:
                        pass
            
            # 경과일 계산: JHCR_DE - 최근 TRNG_ENDE (일수 차이)
            elapsed_days = None
            if jhcr_de and training_end_dates:
                try:
                    # 구직인증 일자를 datetime 객체로 변환
                    jhcr_date = datetime.strptime(jhcr_de, DEFAULT_DATE_FORMAT)
                    # 가장 최근 훈련 종료일 (최대값)
                    latest_end_date = max(training_end_dates)
                    # 일수 차이 계산
                    elapsed_days = (jhcr_date - latest_end_date).days
                    elapsed_days = elapsed_days if elapsed_days >= 0 else None
                except:
                    elapsed_days = None
            
            # 텍스트 포맷팅: {TRNG_CRSN}: ({TRNG_BGDE} ~ {TRNG_ENDE})
            training_texts = []
            for tr in trainings:
                trng_crsn = tr.get("TRNG_CRSN", "").strip()  # 훈련 과정명
                trng_bgde = tr.get("TRNG_BGDE", "").strip()  # 훈련 시작일
                trng_ende = tr.get("TRNG_ENDE", "").strip()  # 훈련 종료일
                if trng_crsn and trng_bgde and trng_ende:
                    training_texts.append(f"{trng_crsn}: ({trng_bgde} ~ {trng_ende})")
            
            text = "\n".join(training_texts) if training_texts else "정보 없음"
            
            # HOPE_JSCD1 정보 가져와서 직종명으로 변환
            hope_jscd1 = self.hope_jscd1_map.get(seek_id, "")
            job_name = self.get_job_name_from_code(hope_jscd1)
            job_examples = []  # 필요시 HOPE_JSCD1로부터 직종 예시 리스트 생성 가능
            
            # 점수 계산
            score, why = self.llm_scorer.score("직업훈련", job_name, job_examples, text)
            
            # JHNT_CTN 가져오기
            jhnt_ctn = item.get("JHNT_CTN", "")
            
            rows.append({
                "SEEK_CUST_NO": seek_id,
                "JHNT_CTN": jhnt_ctn,
                "training_score": score,
                "elapsed_days": elapsed_days if elapsed_days is not None else None
            })
        
        return pd.DataFrame(rows)


    def _preprocess_certification(self, data):
        """자격증 특화 전처리"""
        if not isinstance(data, list):
            data = [data]
        
        rows = []
        for item in data:
            seek_id = item.get("SEEK_CUST_NO", "")
            if not seek_id:
                continue
            
            # JSON에서 자격증 데이터 추출
            licenses = item.get("LICENSES", [])
            
            # 자격증 포맷팅: 자격증1: 전기기능사/국가기술자격 형식
            formatted_texts = []
            for idx, lic in enumerate(licenses, start=1):
                qulf_itnm = lic.get("QULF_ITNM", "").strip()  # 자격증명
                qulf_lcns_lcfn = lic.get("QULF_LCNS_LCFN", "").strip()  # 자격증 분류
                
                if qulf_itnm and qulf_lcns_lcfn:
                    formatted_texts.append(f"자격증{idx}: {qulf_itnm}/{qulf_lcns_lcfn}")
                elif qulf_itnm:
                    formatted_texts.append(f"자격증{idx}: {qulf_itnm}")
            
            # 텍스트 생성
            text = "\n".join(formatted_texts) if formatted_texts else "정보 없음"
            
            # HOPE_JSCD1 정보 가져와서 직종명으로 변환
            hope_jscd1 = self.hope_jscd1_map.get(seek_id, "")
            job_name = self.get_job_name_from_code(hope_jscd1)
            job_examples = []  # 필요시 HOPE_JSCD1로부터 직종 예시 리스트 생성 가능
            
            # 점수 계산
            score, _ = self.llm_scorer.score("자격증", job_name, job_examples, text)
            
            # JHNT_CTN 가져오기
            jhnt_ctn = item.get("JHNT_CTN", "")
            
            # score만 반환
            rows.append({
                "SEEK_CUST_NO": seek_id,
                "JHNT_CTN": jhnt_ctn,
                "certification_score": score
            })
        
        return pd.DataFrame(rows)


    def load_and_preprocess_data(self, data_file, json_name=None):
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
            data = pd.read_excel(data_file, sheet_name=self.sheet_name)
            data_processed = self.basic_preprocessing(data)
        elif data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON 파일의 경우 json_name을 데이터 타입으로 사용
                data_processed = self.nlp_preprocessing(data, json_name=json_name)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. CSV, Excel 또는 JSON 파일을 사용하세요.")
        
        return data_processed


    def get_merged_df(self, file_list):
        """
        파일명 리스트를 받아 각 파일을 load_and_preprocess_data로 읽고 self.df_list에 append,
        이후 SEEK_CUST_NO 컬럼 기준으로 순차적으로 조인하여 데이터프레임 반환

        Args:
            file_list (list): 파일명(str) 리스트
 
        Returns:
            pd.DataFrame: SEEK_CUST_NO 또는 JHNT_CTN 기준으로 조인된 데이터프레임 -> repeat 처리 필요
        """
        self.df_list = []
        result = None
        
        # 첫 번째 파일(엑셀) 먼저 처리 - HOPE_JSCD1(희망 직종 코드) 정보 저장
        if file_list:
            current_json_name = self.json_names[0]
            df = self.load_and_preprocess_data(file_list[0], json_name=current_json_name)
            self.df_list.append(df)
            result = df
            
            # HOPE_JSCD1 정보를 SEEK_CUST_NO 기준으로 매핑하여 저장
            if 'HOPE_JSCD1' in df.columns and 'SEEK_CUST_NO' in df.columns:
                self.hope_jscd1_map = df.set_index('SEEK_CUST_NO')['HOPE_JSCD1'].to_dict()
        
        # 나머지 4개 파일 반복문으로 처리
        for idx, file in enumerate(file_list[1:], start=1):
            # 각 파일에 대응하는 시트명과 JSON명 사용       
            current_json_name = self.json_names[idx]
            df = self.load_and_preprocess_data(file, json_name=current_json_name)
            self.df_list.append(df)
            
            # 직업훈련과 자격증은 JHNT_CTN 기준으로 merge
            if current_json_name in ['직업훈련', '자격증']:
                merge_key = "JHNT_CTN"
            else:
                merge_key = "SEEK_CUST_NO"
            
            result = result.merge(df, on=merge_key, how="outer", suffixes=('', f'_df{idx+1}'))
        
        return result