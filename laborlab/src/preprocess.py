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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time

from .llm_reference import (
    JSON_NAMES, RESUME_SECTIONS, SUPPORTED_SECTIONS, 
    DEFAULT_MAX_COVER_LEN, DEFAULT_COVER_EXCEED_RATIO, DEFAULT_DATE_FORMAT
)
from .llm_scorer import LLMScorer



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
        # variable_mapping.json은 variant_data 폴더에 있음
        # __file__ 기준으로 경로 계산: src/preprocess.py -> laborlab/ -> data/
        preprocess_file = Path(__file__)  # src/preprocess.py
        laborlab_dir = preprocess_file.parent.parent  # laborlab/
        variable_mapping_path = laborlab_dir / "data" / "variant_data" / "variable_mapping.json"
        
        with open(variable_mapping_path, encoding='utf-8') as f:
            variable_mapping = json.load(f)
        return variable_mapping
    
    def load_job_mapping(self):
        """job_subcategories.csv를 로드하여 소분류코드 -> 소분류명 매핑 생성"""
        try:
            # __file__ 기준으로 경로 계산: src/preprocess.py -> laborlab/ -> data/
            preprocess_file = Path(__file__)  # src/preprocess.py
            laborlab_dir = preprocess_file.parent.parent  # laborlab/
            job_mapping_path = laborlab_dir / "data" / "fixed_data" / "job_subcategories.csv"
            
            job_df = pd.read_csv(job_mapping_path, encoding='utf-8')
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
        # 디버깅: 원본 데이터 컬럼 확인
        print(f"[DEBUG] basic_preprocessing 시작 - 원본 데이터 컬럼 수: {len(df.columns)}")
        print(f"[DEBUG] 원본 데이터에 SEEK_CUST_NO 존재: {'SEEK_CUST_NO' in df.columns}")
        print(f"[DEBUG] 원본 데이터에 JHNT_CTN 존재: {'JHNT_CTN' in df.columns}")
        print(f"[DEBUG] 원본 데이터에 JHNT_MBN 존재: {'JHNT_MBN' in df.columns}")
        

        # 병합에 필요한 키 컬럼은 항상 유지
        merge_keys = ["SEEK_CUST_NO", "JHNT_CTN", "JHNT_MBN"]
        existing_merge_keys = [key for key in merge_keys if key in df.columns]
        print(f"[DEBUG] 발견된 병합 키: {existing_merge_keys}")
        
        # variable_mapping.json의 structured_data 키만 사용
        structured_keys = set(self.variable_mapping.get("structured_data", {}).keys())
        
        # 원본 데이터에서 해당 변수들만 필터링 (존재하는 변수만)
        available_vars = list(structured_keys & set(df.columns))
        missing_vars = list(structured_keys - set(df.columns))
        
        if missing_vars:
            print(f"다음 변수들이 데이터에 없습니다: {missing_vars}")
        
        # 병합 키와 필터링된 변수들을 합침 (중복 제거)
        final_vars = list(set(available_vars + existing_merge_keys))
        print(f"[DEBUG] 최종 컬럼 수: {len(final_vars)}, SEEK_CUST_NO 포함 여부: {'SEEK_CUST_NO' in final_vars}")
        df = df[final_vars]

        # BFR_OCTR_YN 제거, BFR_OCTR_CT만 유지
        if "BFR_OCTR_YN" in df.columns and "BFR_OCTR_CT" in df.columns:
            df = df.drop(columns=["BFR_OCTR_YN"])
            print(f"[DEBUG] BFR_OCTR_YN 제거 후 SEEK_CUST_NO 존재: {'SEEK_CUST_NO' in df.columns}")

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
            df = df.drop(columns=agree_vars)
            print(f"[DEBUG] agree_vars 제거 후 SEEK_CUST_NO 존재: {'SEEK_CUST_NO' in df.columns}")

        print(f"[DEBUG] basic_preprocessing 완료 - 최종 컬럼 수: {len(df.columns)}, SEEK_CUST_NO 존재: {'SEEK_CUST_NO' in df.columns}")
        if 'SEEK_CUST_NO' in df.columns:
            print(f"[DEBUG] SEEK_CUST_NO 샘플 값: {df['SEEK_CUST_NO'].head(3).tolist()}")
        
        return df

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


    def _process_single_resume(self, item):
        """단일 이력서 레코드 처리 (병렬 처리용)"""
        seek_id = item.get("SEEK_CUST_NO", "")
        if not seek_id:
            return None
        
        # BASIC_RESUME_YN == "Y"인 resume 찾기
        resumes = item.get("RESUMES", [])
        basic_resume = None
        for resume in resumes:
            if str(resume.get("BASIC_RESUME_YN", "")).upper() == "Y":
                basic_resume = resume
                break
        
        # 기본 이력서가 없으면 빈 결과 반환
        if basic_resume is None:
            return {
                "SEEK_CUST_NO": seek_id,
                "resume_score": None,
                "items_num": 0
            }
        
        # ITEMS 가져오기
        items = basic_resume.get("ITEMS", [])
        items_num = len(items)
        
        # variable_mapping에서 resume 섹션 가져오기
        resume_mapping = self.variable_mapping.get("resume", {})
        
        # ITEMS를 포매팅
        formatting_sentence = ""
        for item_data in items:
            for key, value in item_data.items():
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
        score, _ = self.llm_scorer.score("이력서", job_name, job_examples, formatting_sentence)
        
        return {
            "SEEK_CUST_NO": seek_id,
            "resume_score": score,
            "items_num": items_num
        }
    
    def _preprocess_resume(self, data):
        """이력서 특화 전처리 (병렬 처리)"""
        # 리스트인 경우 처리 (JSON 파일이 리스트 형태일 수 있음)
        if not isinstance(data, list):
            data = [data]
        
        # 병렬 처리로 각 레코드 처리
        max_workers = min(len(data), 10)  # 최대 10개 스레드
        rows = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single_resume, item): item for item in data}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        rows.append(result)
                except Exception as e:
                    item = futures[future]
                    seek_id = item.get("SEEK_CUST_NO", "unknown")
                    print(f"⚠️ 이력서 처리 오류 (SEEK_CUST_NO: {seek_id}): {e}")
                    rows.append({
                        "SEEK_CUST_NO": seek_id,
                        "resume_score": None,
                        "items_num": 0
                    })
        
        # DataFrame 생성 전에 Logger 객체 확인 및 제거
        import logging
        cleaned_rows = []
        for row_idx, row in enumerate(rows):
            cleaned_row = {}
            for key, value in row.items():
                # Logger 객체인지 확인
                if isinstance(value, logging.Logger) or 'Logger' in str(type(value)):
                    print(f"⚠️ [이력서 전처리] {row_idx}번째 행의 딕셔너리 키 '{key}'에 Logger 객체 발견! (타입: {type(value).__name__})")
                    cleaned_row[key] = np.nan
                else:
                    cleaned_row[key] = value
            cleaned_rows.append(cleaned_row)
        
        return pd.DataFrame(cleaned_rows)


    def _process_single_cover_letter(self, item):
        """단일 자기소개서 레코드 처리 (병렬 처리용)"""
        seek_id = item.get("SEEK_CUST_NO", "")
        if not seek_id:
            return None
                
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
        
        # 점수 계산과 오탈자 수 계산을 병렬로 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            score_future = executor.submit(self.llm_scorer.score, "자기소개서", job_name, job_examples, full_text)
            typo_future = executor.submit(self.llm_scorer.count_typos, full_text)
            
            score, _ = score_future.result()
            typo_count = typo_future.result()
        
        # score와 오탈자 수만 반환 (그래프 변수명과 일치)
        return {
            "SEEK_CUST_NO": seek_id,
            "cover_score": score,  # 그래프: cover_score
            "cover_typo_count": typo_count  # 그래프: cover_typo_count
        }
    
    def _preprocess_cover_letter(self, data):
        """자기소개서 특화 전처리 (병렬 처리)"""
        if not isinstance(data, list):
            data = [data]
        
        # 병렬 처리로 각 레코드 처리
        max_workers = min(len(data), 10)  # 최대 10개 스레드
        rows = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single_cover_letter, item): item for item in data}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        rows.append(result)
                except Exception as e:
                    item = futures[future]
                    seek_id = item.get("SEEK_CUST_NO", "unknown")
                    print(f"⚠️ 자기소개서 처리 오류 (SEEK_CUST_NO: {seek_id}): {e}")
                    rows.append({
                        "SEEK_CUST_NO": seek_id,
                        "cover_score": None,
                        "cover_typo_count": 0
                    })
        
        # DataFrame 생성 전에 Logger 객체 확인 및 제거
        import logging
        cleaned_rows = []
        for row_idx, row in enumerate(rows):
            cleaned_row = {}
            for key, value in row.items():
                # Logger 객체인지 확인
                if isinstance(value, logging.Logger) or 'Logger' in str(type(value)):
                    print(f"⚠️ [자기소개서 전처리] {row_idx}번째 행의 딕셔너리 키 '{key}'에 Logger 객체 발견! (타입: {type(value).__name__})")
                    cleaned_row[key] = np.nan
                else:
                    cleaned_row[key] = value
            cleaned_rows.append(cleaned_row)
        
        return pd.DataFrame(cleaned_rows)


    def _process_single_training(self, item):
        """단일 직업훈련 레코드 처리 (병렬 처리용)"""
        seek_id = item.get("SEEK_CUST_NO", "")
        if not seek_id:
            return None
        
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
        
        return {
            "SEEK_CUST_NO": seek_id,
            "JHNT_CTN": jhnt_ctn,
            "training_score": score,
            "days_last_training_to_jobseek": elapsed_days if elapsed_days is not None else None  # 그래프: days_last_training_to_jobseek
        }
    
    def _preprocess_training(self, data):
        """직업훈련 특화 전처리 (병렬 처리)"""
        if not isinstance(data, list):
            data = [data]
        
        # 병렬 처리로 각 레코드 처리
        max_workers = min(len(data), 10)  # 최대 10개 스레드
        rows = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single_training, item): item for item in data}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        rows.append(result)
                except Exception as e:
                    item = futures[future]
                    seek_id = item.get("SEEK_CUST_NO", "unknown")
                    print(f"⚠️ 직업훈련 처리 오류 (SEEK_CUST_NO: {seek_id}): {e}")
                    rows.append({
                        "SEEK_CUST_NO": seek_id,
                        "JHNT_CTN": item.get("JHNT_CTN", ""),
                        "training_score": None,
                        "days_last_training_to_jobseek": None
                    })
        
        # DataFrame 생성 전에 Logger 객체 확인 및 제거
        import logging
        cleaned_rows = []
        for row_idx, row in enumerate(rows):
            cleaned_row = {}
            for key, value in row.items():
                # Logger 객체인지 확인
                if isinstance(value, logging.Logger) or 'Logger' in str(type(value)):
                    print(f"⚠️ [직업훈련 전처리] {row_idx}번째 행의 딕셔너리 키 '{key}'에 Logger 객체 발견! (타입: {type(value).__name__})")
                    cleaned_row[key] = np.nan
                else:
                    cleaned_row[key] = value
            cleaned_rows.append(cleaned_row)
        
        return pd.DataFrame(cleaned_rows)


    def _process_single_certification(self, item):
        """단일 자격증 레코드 처리 (병렬 처리용)"""
        seek_id = item.get("SEEK_CUST_NO", "")
        if not seek_id:
            return None
        
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
        
        # score만 반환 (그래프 변수명과 일치)
        return {
            "SEEK_CUST_NO": seek_id,
            "JHNT_CTN": jhnt_ctn,
            "license_score": score  # 그래프: license_score
        }
    
    def _preprocess_certification(self, data):
        """자격증 특화 전처리 (병렬 처리)"""
        if not isinstance(data, list):
            data = [data]
        
        # 병렬 처리로 각 레코드 처리
        max_workers = min(len(data), 10)  # 최대 10개 스레드
        rows = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single_certification, item): item for item in data}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        rows.append(result)
                except Exception as e:
                    item = futures[future]
                    seek_id = item.get("SEEK_CUST_NO", "unknown")
                    print(f"⚠️ 자격증 처리 오류 (SEEK_CUST_NO: {seek_id}): {e}")
                    rows.append({
                        "SEEK_CUST_NO": seek_id,
                        "JHNT_CTN": item.get("JHNT_CTN", ""),
                        "license_score": None
                    })
        
        # DataFrame 생성 전에 Logger 객체 확인 및 제거
        import logging
        cleaned_rows = []
        for row_idx, row in enumerate(rows):
            cleaned_row = {}
            for key, value in row.items():
                # Logger 객체인지 확인
                if isinstance(value, logging.Logger) or 'Logger' in str(type(value)):
                    print(f"⚠️ [자격증 전처리] {row_idx}번째 행의 딕셔너리 키 '{key}'에 Logger 객체 발견! (타입: {type(value).__name__})")
                    cleaned_row[key] = np.nan
                else:
                    cleaned_row[key] = value
            cleaned_rows.append(cleaned_row)
        
        return pd.DataFrame(cleaned_rows)


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
        
        첫 번째 파일(CSV)은 순차 처리하고, 나머지 4개 JSON 파일은 병렬로 처리합니다.

        Args:
            file_list (list): 파일명(str) 리스트
 
        Returns:
            pd.DataFrame: SEEK_CUST_NO 또는 JHNT_CTN 기준으로 조인된 데이터프레임 -> repeat 처리 필요
        """
        self.df_list = []
        result = None
        
        # 첫 번째 파일(정형 데이터 CSV) 먼저 처리 - HOPE_JSCD1(희망 직종 코드) 정보 저장
        if file_list:
            # 첫 번째 파일은 정형 데이터이므로 json_name=None
            csv_start_time = time.time()
            print(f"[DEBUG] 첫 번째 파일 처리 시작: {file_list[0]}, 타입: 정형 데이터 (CSV)")
            df = self.load_and_preprocess_data(file_list[0], json_name=None)
            csv_elapsed = time.time() - csv_start_time
            print(f"⏱️ 정형 데이터(CSV) 처리 소요 시간: {csv_elapsed:.2f}초")
            self.df_list.append(df)
            result = df
            
            print(f"[DEBUG] 첫 번째 데이터프레임 크기: {result.shape}")
            print(f"[DEBUG] 첫 번째 데이터프레임 컬럼: {list(result.columns)}")
            print(f"[DEBUG] 첫 번째 데이터프레임에 SEEK_CUST_NO 존재: {'SEEK_CUST_NO' in result.columns}")
            print(f"[DEBUG] 첫 번째 데이터프레임에 JHNT_CTN 존재: {'JHNT_CTN' in result.columns}")
            
            # HOPE_JSCD1 정보를 SEEK_CUST_NO 기준으로 매핑하여 저장
            if 'HOPE_JSCD1' in df.columns and 'SEEK_CUST_NO' in df.columns:
                self.hope_jscd1_map = df.set_index('SEEK_CUST_NO')['HOPE_JSCD1'].to_dict()
                print(f"[DEBUG] HOPE_JSCD1 매핑 생성 완료: {len(self.hope_jscd1_map)}개")
            else:
                print(f"[DEBUG] 경고: HOPE_JSCD1 또는 SEEK_CUST_NO가 없어 매핑을 생성할 수 없습니다.")
        
        # 나머지 4개 파일을 병렬로 처리
        json_files = []
        for idx, file in enumerate(file_list[1:], start=0):
            if idx >= len(self.json_names):
                raise IndexError(f"JSON 파일 수({len(file_list)-1})가 json_names 길이({len(self.json_names)})를 초과합니다. file: {file}")
            current_json_name = self.json_names[idx]
            json_files.append((file, current_json_name, idx))
        
        # 병렬 처리로 4개 파일 동시 처리
        processed_dfs = {}
        max_workers = min(len(json_files), 4)  # 최대 4개 스레드 (4개 파일)
        
        def process_json_file(file_info):
            """단일 JSON 파일 처리 함수 (병렬 처리용)"""
            file, json_name, idx = file_info
            try:
                file_start_time = time.time()
                print(f"[DEBUG] {idx+1}번째 파일 처리 시작: {file}, 타입: {json_name}")
                df = self.load_and_preprocess_data(file, json_name=json_name)
                file_elapsed = time.time() - file_start_time
                print(f"[DEBUG] {json_name} 데이터프레임 크기: {df.shape}")
                print(f"[DEBUG] {json_name} 데이터프레임 컬럼: {list(df.columns)}")
                print(f"⏱️ {json_name} 처리 소요 시간: {file_elapsed:.2f}초")
                return (json_name, df, idx, file_elapsed)
            except Exception as e:
                print(f"⚠️ {json_name} 파일 처리 오류: {e}")
                raise
        
        json_file_times = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_json_file, file_info): file_info for file_info in json_files}
            
            for future in as_completed(futures):
                try:
                    json_name, df, idx, file_elapsed = future.result()
                    processed_dfs[idx] = (json_name, df)
                    json_file_times[json_name] = file_elapsed
                except Exception as e:
                    file_info = futures[future]
                    print(f"⚠️ 파일 처리 실패: {file_info[0]}, 오류: {e}")
                    raise
        
        # JSON 파일 처리 시간 요약 출력
        if json_file_times:
            print("\n" + "="*60)
            print("⏱️ JSON 파일별 처리 시간 요약")
            print("="*60)
            total_json_time = sum(json_file_times.values())
            for json_name, elapsed in sorted(json_file_times.items(), key=lambda x: x[1], reverse=True):
                percentage = (elapsed / total_json_time * 100) if total_json_time > 0 else 0
                print(f"  {json_name:15s}: {elapsed:7.2f}초 ({percentage:5.1f}%)")
            print(f"  {'전체':15s}: {total_json_time:7.2f}초 (100.0%)")
            print("="*60)
        
        # 처리된 데이터프레임들을 순서대로 병합
        merge_start_time = time.time()
        for idx in sorted(processed_dfs.keys()):
            json_name, df = processed_dfs[idx]
            self.df_list.append(df)
            
            # 직업훈련과 자격증은 JHNT_CTN 기준으로 merge
            if json_name in ['직업훈련', '자격증']:
                merge_key = "JHNT_CTN"
            else:
                merge_key = "SEEK_CUST_NO"
            
            print(f"[DEBUG] 병합 키: {merge_key}")
            print(f"[DEBUG] result에 {merge_key} 존재: {merge_key in result.columns}")
            print(f"[DEBUG] {json_name}에 {merge_key} 존재: {merge_key in df.columns}")
            
            # 병합 키 컬럼 존재 여부 확인
            if merge_key not in result.columns:
                print(f"[DEBUG] ERROR: result 컬럼 목록: {list(result.columns)}")
                raise KeyError(f"병합 키 '{merge_key}'가 첫 번째 데이터프레임에 없습니다. 사용 가능한 컬럼: {list(result.columns)}")
            if merge_key not in df.columns:
                print(f"[DEBUG] ERROR: {json_name} 컬럼 목록: {list(df.columns)}")
                raise KeyError(f"병합 키 '{merge_key}'가 {json_name} 데이터프레임에 없습니다. 파일: {file_list[idx+1]}, 사용 가능한 컬럼: {list(df.columns)}")
            
            print(f"[DEBUG] 병합 전 result 크기: {result.shape}, {json_name} 크기: {df.shape}")
            
            # 병합 전에 Logger 객체가 있는지 확인
            import logging
            for col in df.columns:
                if df[col].dtype == 'object' and len(df) > 0:
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        first_val = non_null_values.iloc[0]
                        if isinstance(first_val, logging.Logger) or 'Logger' in str(type(first_val)):
                            print(f"⚠️ [병합 전] {json_name}의 컬럼 '{col}'에 Logger 객체 발견! (타입: {type(first_val).__name__})")
                            # Logger 객체를 NaN으로 대체
                            df[col] = df[col].apply(lambda x: np.nan if (isinstance(x, logging.Logger) or 'Logger' in str(type(x))) else x)
            
            result = result.merge(df, on=merge_key, how="outer", suffixes=('', f'_df{idx+1}'))
            print(f"[DEBUG] 병합 후 result 크기: {result.shape}")
            
            # 병합 후에 Logger 객체가 있는지 확인
            for col in result.columns:
                if result[col].dtype == 'object' and len(result) > 0:
                    non_null_values = result[col].dropna()
                    if len(non_null_values) > 0:
                        first_val = non_null_values.iloc[0]
                        if isinstance(first_val, logging.Logger) or 'Logger' in str(type(first_val)):
                            print(f"⚠️ [병합 후] result의 컬럼 '{col}'에 Logger 객체 발견! (타입: {type(first_val).__name__})")
        
        merge_elapsed = time.time() - merge_start_time
        print(f"⏱️ 데이터 병합 소요 시간: {merge_elapsed:.2f}초")
        
        # Logger 객체가 데이터프레임에 포함되어 있는지 검사
        import logging
        logger_columns = []
        for col in result.columns:
            if result[col].dtype == 'object' and len(result) > 0:
                non_null_values = result[col].dropna()
                if len(non_null_values) > 0:
                    first_val = non_null_values.iloc[0]
                    # Logger 객체인지 확인
                    if isinstance(first_val, logging.Logger) or 'Logger' in str(type(first_val)):
                        logger_columns.append((col, type(first_val).__name__))
                        print(f"⚠️ [전처리] 경고: 컬럼 '{col}'에 Logger 객체가 포함되어 있습니다! (타입: {type(first_val).__name__})")
        
        if logger_columns:
            print(f"\n❌ [전처리] 오류: 다음 컬럼에 Logger 객체가 발견되었습니다:")
            for col, col_type in logger_columns:
                print(f"   - {col} (타입: {col_type})")
            print(f"이 컬럼들은 데이터 정리 과정에서 제거됩니다.")
        
        return result