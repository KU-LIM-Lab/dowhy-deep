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
import asyncio
import aiohttp
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from .llm_reference import (
    JSON_NAMES, RESUME_SECTIONS, SUPPORTED_SECTIONS, 
    DEFAULT_MAX_COVER_LEN, DEFAULT_COVER_EXCEED_RATIO, DEFAULT_DATE_FORMAT
)
from .llm_scorer import LLMScorer



class Preprocessor:
    def __init__(self, df_list, job_category_file="KSIC", max_concurrent_requests=None, top_job_categories=5):
        self.json_names = JSON_NAMES
        self.sheet_name = '구직인증 관련 데이터'
        self.df_list = []
        self.variable_mapping = self.load_variable_mapping()
        self.llm_scorer = LLMScorer()
        self.hope_jscd1_map = {}  # JHNT_MBN -> HOPE_JSCD1 매핑 저장
        self.job_category_file = job_category_file  # 직종 소분류 파일명 (KECO, KSCO, KSIC)
        self.job_code_to_name = self.load_job_mapping()  # 소분류코드 -> 소분류명 매핑
        self.top_job_categories = top_job_categories  # 상위 직종 소분류 개수 (-1이면 전체 사용)
        
        # 동시 요청 수 제한 설정 (OLLAMA_NUM_PARALLEL 환경변수 또는 기본값 사용)
        if max_concurrent_requests is None:
            max_concurrent_requests = int(os.getenv("OLLAMA_NUM_PARALLEL", "32"))
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        print(f"🔧 Ollama 동시 요청 수 제한: {max_concurrent_requests}개")

    def load_variable_mapping(self):
        # variable_mapping.json은 data 폴더에 있음
        # __file__ 기준으로 경로 계산: src/preprocess.py -> laborlab_2/ -> data/
        preprocess_file = Path(__file__)  # src/preprocess.py
        laborlab_dir = preprocess_file.parent.parent  # laborlab_2/
        variable_mapping_path = laborlab_dir / "data" / "variable_mapping.json"
        
        with open(variable_mapping_path, encoding='utf-8') as f:
            variable_mapping = json.load(f)
        return variable_mapping
    
    def load_job_mapping(self):
        """job_subcategories_XXXX.csv를 로드하여 소분류코드 -> 소분류명 매핑 생성"""
        try:
            # __file__ 기준으로 경로 계산: src/preprocess.py -> laborlab_2/ -> data/
            preprocess_file = Path(__file__)  # src/preprocess.py
            laborlab_dir = preprocess_file.parent.parent  # laborlab_2/
            
            # job_category_file에 따라 파일명 결정 (KECO, KSCO, KSIC)
            job_category_file = self.job_category_file.upper()
            if job_category_file not in ["KECO", "KSCO", "KSIC"]:
                print(f"⚠️ 잘못된 직종 소분류 파일명: {job_category_file}. 기본값 KSIC 사용")
                job_category_file = "KSIC"
            
            job_mapping_path = laborlab_dir / "data" / f"job_subcategories_{job_category_file}.csv"
            
            if not job_mapping_path.exists():
                print(f"⚠️ 직종 소분류 파일을 찾을 수 없습니다: {job_mapping_path}")
                print(f"   기본값 job_subcategories_KSIC.csv 사용 시도")
                job_mapping_path = laborlab_dir / "data" / "job_subcategories_KSIC.csv"
            
            job_df = pd.read_csv(job_mapping_path, encoding='utf-8')
            print(f"✅ 직종 소분류 파일 로드 완료: {job_mapping_path.name} ({len(job_df)}개 직종)")
            
            # 소분류코드를 문자열로 변환하여 딕셔너리 생성
            job_mapping = dict(zip(job_df['소분류코드'].astype(str).str.zfill(3), job_df['소분류명']))
            return job_mapping
        except Exception as e:
            print(f"❌ 직종 소분류 파일 로드 실패: {e}")
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
        print(f"[DEBUG] 원본 데이터에 JHNT_CTN 존재: {'JHNT_CTN' in df.columns}")
        print(f"[DEBUG] 원본 데이터에 JHNT_MBN 존재: {'JHNT_MBN' in df.columns}")
        

        # 병합에 필요한 키 컬럼은 항상 유지
        merge_keys = ["JHNT_CTN", "JHNT_MBN"]
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
        print(f"[DEBUG] 최종 컬럼 수: {len(final_vars)}, JHNT_MBN 포함 여부: {'JHNT_MBN' in final_vars}")
        df = df[final_vars]

        # BFR_OCTR_YN 제거, BFR_OCTR_CT만 유지
        if "BFR_OCTR_YN" in df.columns and "BFR_OCTR_CT" in df.columns:
            df = df.drop(columns=["BFR_OCTR_YN"])
            print(f"[DEBUG] BFR_OCTR_YN 제거 후 JHNT_MBN 존재: {'JHNT_MBN' in df.columns}")

        # 9개 예/아니오 변수 → 합쳐서 새로운 순서형 범주 변수 생성
        agree_vars = [
            "EMAIL_RCYN", "SAEIL_CNTC_AGRE_YN", "SHRS_IDIF_AOFR_YN", "SULC_IDIF_AOFR_YN",
            "IDIF_IQRY_AGRE_YN", "SMS_RCYN", "EMAIL_OTPB_YN", "MPNO_OTPB_YN", "EMAIL_RCYN"
        ]

        # 존재하는 경우만 사용
        agree_vars = [col for col in agree_vars if col in df.columns]

        if agree_vars:
            agree_count = (df[agree_vars] == "예").sum(axis=1)
            df["AGREE_LEVEL"] = agree_count.apply(lambda x: "하" if x <= 3 else ("중" if x <= 6 else "상"))
            df = df.drop(columns=agree_vars)
            print(f"[DEBUG] agree_vars 제거 후 JHNT_MBN 존재: {'JHNT_MBN' in df.columns}")

        # HOPE_JSCD1_NAME 변수 추가 (HOPE_JSCD1 코드를 소분류명으로 변환)
        if "HOPE_JSCD1" in df.columns:
            df["HOPE_JSCD1_NAME"] = df["HOPE_JSCD1"].apply(lambda code: self.get_job_name_from_code(code))
            print(f"[DEBUG] HOPE_JSCD1_NAME 변수 추가 완료: {df['HOPE_JSCD1_NAME'].nunique()}개 고유값")

        print(f"[DEBUG] basic_preprocessing 완료 - 최종 컬럼 수: {len(df.columns)}, JHNT_MBN 존재: {'JHNT_MBN' in df.columns}")
        
        # JHNT_MBN과 JHNT_CTN을 문자열로 통일 (13자리 0패딩)
        if 'JHNT_MBN' in df.columns:
            df['JHNT_MBN'] = df['JHNT_MBN'].astype(str).str.zfill(13)
        if 'JHNT_CTN' in df.columns:
            df['JHNT_CTN'] = df['JHNT_CTN'].astype(str).str.zfill(13)
        
        return df

    async def nlp_preprocessing(self, data, json_name=None, limit_data=False, limit_size=5000):
        """
        NLP 기반 데이터 전처리를 수행하는 함수 (비동기)
        
        Args:
            data: json 파일 (자기소개서, 이력서, 직업훈련, 자격증)
            json_name (str): JSON 데이터 타입에 따라 다른 전처리 적용
            limit_data (bool): 테스트 모드로 데이터 제한 여부
            limit_size (int): 제한할 데이터 크기
        Returns:
            pd.DataFrame: NLP 전처리된 데이터프레임
        """
        # JSON 파일은 배열 형태로 저장되어 있으므로 리스트로 로드됨
        if isinstance(data, list):
            if limit_data and len(data) > limit_size:
                original_count = len(data)
                data = data[:limit_size]
                print(f"📊 {json_name} 데이터 제한: {len(data)}개 레코드 사용 (전체 {original_count}개 중 앞 {limit_size}개)")
            else:
                # limit_data가 False이면 모든 데이터 처리
                print(f"📊 {json_name} 전체 데이터 처리: {len(data)}개 레코드")
        else:
            # 리스트가 아닌 경우 (단일 객체)는 그대로 사용 (나중에 _preprocess_* 함수에서 리스트로 변환됨)
            print(f"⚠️ {json_name} 데이터가 리스트 형태가 아닙니다. 단일 객체로 처리됩니다.")
        
        # JSON 데이터 타입에 따른 특화된 전처리 (비동기)
        if json_name == '이력서':
            df_processed = await self._preprocess_resume(data)
        elif json_name == '자기소개서':
            df_processed = await self._preprocess_cover_letter(data)
        elif json_name == '직업훈련':
            df_processed = await self._preprocess_training(data)
        elif json_name == '자격증':
            df_processed = await self._preprocess_certification(data)
        else:
            raise ValueError(f"지원하지 않는 json 파일입니다. {json_name}")
        
        return df_processed


    async def _process_single_resume(self, item, session: aiohttp.ClientSession):
        """단일 이력서 레코드 처리 (비동기)"""
        # SEEK_CUST_NO를 JHNT_MBN으로 변환
        seek_id = item.get("JHNT_MBN", "") or item.get("SEEK_CUST_NO", "")
        if not seek_id:
            return None
        
        # BASIC_RESUME_YN == "Y"인 resume 찾기 (CONTENTS 배열에서)
        contents = item.get("CONTENTS", [])
        basic_resume = None
        for content in contents:
            if str(content.get("BASIC_RESUME_YN", "")).upper() == "Y":
                basic_resume = content
                break
        
        # 기본 이력서가 없으면 빈 결과 반환
        if basic_resume is None:
            return {
                "JHNT_MBN": seek_id,
                "resume_score": None,
                "items_num": 0
            }
        
        # RESUME_CONTENTS 가져오기
        items = basic_resume.get("RESUME_CONTENTS", [])
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
        
        # LLM scorer에 전달하여 점수 계산 (비동기)
        score, _ = await self.llm_scorer.score_async("이력서", job_name, job_examples, formatting_sentence, session)
        
        return {
            "JHNT_MBN": str(seek_id),  # 문자열로 변환
            "resume_score": score,
            "items_num": items_num
        }
    
    async def _preprocess_resume(self, data):
        """이력서 특화 전처리 (비동기 병렬 처리)"""
        # 리스트인 경우 처리 (JSON 파일이 리스트 형태일 수 있음)
        if not isinstance(data, list):
            data = [data]
        
        # 비동기 병렬 처리로 각 레코드 처리
        rows = []
        import logging
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for item in data:
                task = self._process_single_resume(item, session)
                tasks.append(task)
            
            results = await atqdm.gather(*tasks, desc="이력서 전처리", unit="건")
            
            for idx, result in enumerate(results):
                try:
                    if result is not None:
                        rows.append(result)
                except Exception as e:
                    item = data[idx]
                    seek_id = item.get("JHNT_MBN", "") or item.get("SEEK_CUST_NO", "unknown")
                    print(f"⚠️ 이력서 처리 오류 (JHNT_MBN: {seek_id}): {e}")
                    rows.append({
                        "JHNT_MBN": str(seek_id),  # 문자열로 변환
                        "resume_score": None,
                        "items_num": 0
                    })
        
        # DataFrame 생성 전에 Logger 객체 확인 및 제거
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
        
        df = pd.DataFrame(cleaned_rows)
        
        # SEEK_CUST_NO를 JHNT_MBN으로 rename (있는 경우)
        if 'SEEK_CUST_NO' in df.columns and 'JHNT_MBN' not in df.columns:
            df = df.rename(columns={'SEEK_CUST_NO': 'JHNT_MBN'})
            print(f"✅ 이력서 데이터: SEEK_CUST_NO를 JHNT_MBN으로 변경")
        elif 'SEEK_CUST_NO' in df.columns and 'JHNT_MBN' in df.columns:
            # 둘 다 있으면 SEEK_CUST_NO의 값으로 JHNT_MBN을 채우고 SEEK_CUST_NO 제거
            df['JHNT_MBN'] = df['JHNT_MBN'].fillna(df['SEEK_CUST_NO'])
            df = df.drop(columns=['SEEK_CUST_NO'])
            print(f"✅ 이력서 데이터: SEEK_CUST_NO 값을 JHNT_MBN에 병합 후 SEEK_CUST_NO 제거")
        
        # JHNT_MBN을 문자열로 통일
        if 'JHNT_MBN' in df.columns:
            df['JHNT_MBN'] = df['JHNT_MBN'].astype(str)
        
        return df


    async def _process_single_cover_letter(self, item, session: aiohttp.ClientSession):
        """단일 자기소개서 레코드 처리 (비동기)"""
        # SEEK_CUST_NO를 JHNT_MBN으로 변환
        seek_id = item.get("JHNT_MBN", "") or item.get("SEEK_CUST_NO", "")
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
        
        # 점수 계산과 오탈자 수 계산을 비동기로 병렬 실행
        score_task = self.llm_scorer.score_async("자기소개서", job_name, job_examples, full_text, session)
        typo_task = self.llm_scorer.count_typos_async(full_text, session)
        score, _ = await score_task
        typo_count = await typo_task
        
        # score와 오탈자 수만 반환 (그래프 변수명과 일치)
        return {
            "JHNT_MBN": str(seek_id),  # 문자열로 변환
            "cover_letter_score": score,  # 그래프: cover_letter_score
            "cover_letter_typo_count": typo_count  # 그래프: cover_letter_typo_count
        }
    
    async def _preprocess_cover_letter(self, data):
        """자기소개서 특화 전처리 (비동기 병렬 처리)"""
        if not isinstance(data, list):
            data = [data]
        
        # 비동기 병렬 처리로 각 레코드 처리
        rows = []
        import logging
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for item in data:
                task = self._process_single_cover_letter(item, session)
                tasks.append(task)
            
            results = await atqdm.gather(*tasks, desc="자기소개서 전처리", unit="건")
            
            for idx, result in enumerate(results):
                try:
                    if result is not None:
                        rows.append(result)
                except Exception as e:
                    item = data[idx]
                    seek_id = item.get("JHNT_MBN", "") or item.get("SEEK_CUST_NO", "unknown")
                    print(f"⚠️ 자기소개서 처리 오류 (JHNT_MBN: {seek_id}): {e}")
                    rows.append({
                        "JHNT_MBN": str(seek_id),  # 문자열로 변환
                        "cove_letter_score": None,
                        "cover_letter_typo_count": 0
                    })
        
        # DataFrame 생성 전에 Logger 객체 확인 및 제거
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
        
        df = pd.DataFrame(cleaned_rows)
        
        # SEEK_CUST_NO를 JHNT_MBN으로 rename (있는 경우)
        if 'SEEK_CUST_NO' in df.columns and 'JHNT_MBN' not in df.columns:
            df = df.rename(columns={'SEEK_CUST_NO': 'JHNT_MBN'})
            print(f"✅ 자기소개서 데이터: SEEK_CUST_NO를 JHNT_MBN으로 변경")
        elif 'SEEK_CUST_NO' in df.columns and 'JHNT_MBN' in df.columns:
            # 둘 다 있으면 SEEK_CUST_NO의 값으로 JHNT_MBN을 채우고 SEEK_CUST_NO 제거
            df['JHNT_MBN'] = df['JHNT_MBN'].fillna(df['SEEK_CUST_NO'])
            df = df.drop(columns=['SEEK_CUST_NO'])
            print(f"✅ 자기소개서 데이터: SEEK_CUST_NO 값을 JHNT_MBN에 병합 후 SEEK_CUST_NO 제거")
        
        # JHNT_MBN을 문자열로 통일
        if 'JHNT_MBN' in df.columns:
            df['JHNT_MBN'] = df['JHNT_MBN'].astype(str)
        
        return df


    async def _process_single_training(self, item, session: aiohttp.ClientSession):
        """단일 직업훈련 레코드 처리 (비동기)"""
        # JHNT_CTN을 키로 사용
        jhnt_ctn = item.get("JHNT_CTN", "")
        if not jhnt_ctn:
            return None
        
        # 구직인증 일자 가져오기
        jhcr_de = item.get("JHCR_DE", "")  # 구직인증 일자
        
        # CONTENTS에서 훈련 데이터 추출
        trainings = item.get("CONTENTS", [])
        
        # 날짜 파싱 헬퍼 함수 (여러 형식 지원)
        def parse_date(date_str):
            if not date_str:
                return None
            date_str = date_str.strip()
            for fmt in ["%Y-%m-%d", "%Y%m%d"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
            return None
        
        # CONTENTS에서 모든 TRNG_ENDE 가져와서 datetime 객체 리스트로 변환
        training_end_dates = []
        for tr in trainings:
            trng_ende = tr.get("TRNG_ENDE", "")
            date_obj = parse_date(trng_ende)
            if date_obj:
                training_end_dates.append(date_obj)
        
        # 경과일 계산: JHCR_DE - 최근 TRNG_ENDE (일수 차이)
        elapsed_days = None
        if jhcr_de and training_end_dates:
            try:
                # 구직인증 일자를 datetime 객체로 변환
                jhcr_date = parse_date(jhcr_de)
                # 가장 최근 훈련 종료일 (최대값)    
                latest_end_date = max(training_end_dates)
                # 일수 차이 계산 (둘 다 유효한 경우에만)
                if jhcr_date and latest_end_date:
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
        
        # seek_id는 HOPE_JSCD1 매핑을 위해 사용 (JHNT_MBN이 있으면 사용, 없으면 None)
        seek_id = item.get("JHNT_MBN", "") or item.get("SEEK_CUST_NO", "")
        
        # HOPE_JSCD1 정보 가져와서 직종명으로 변환
        hope_jscd1 = self.hope_jscd1_map.get(seek_id, "")
        job_name = self.get_job_name_from_code(hope_jscd1)
        job_examples = []  # 필요시 HOPE_JSCD1로부터 직종 예시 리스트 생성 가능
        
        # 점수 계산 (비동기)
        score, why = await self.llm_scorer.score_async("직업훈련", job_name, job_examples, text, session)
        
        return {
            "JHNT_CTN": str(jhnt_ctn),  # 문자열로 변환
            "training_score": score,
            "days_last_training_to_jobseek": elapsed_days if elapsed_days is not None else None  # 그래프: days_last_training_to_jobseek
        }
    
    async def _preprocess_training(self, data):
        """직업훈련 특화 전처리 (비동기 병렬 처리)"""
        if not isinstance(data, list):
            data = [data]
        
        # 비동기 병렬 처리로 각 레코드 처리
        rows = []
        import logging
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for item in data:
                task = self._process_single_training(item, session)
                tasks.append(task)
            
            results = await atqdm.gather(*tasks, desc="직업훈련 전처리", unit="건")
            
            for idx, result in enumerate(results):
                try:
                    if result is not None:
                        rows.append(result)
                except Exception as e:
                    item = data[idx]
                    jhnt_ctn = item.get("JHNT_CTN", "unknown")
                    print(f"⚠️ 직업훈련 처리 오류 (JHNT_CTN: {jhnt_ctn}): {e}")
                    rows.append({
                        "JHNT_CTN": str(jhnt_ctn),  # 문자열로 변환
                        "training_score": None,
                        "days_last_training_to_jobseek": None
                    })
        
        # DataFrame 생성 전에 Logger 객체 확인 및 제거
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


    async def _process_single_certification(self, item, session: aiohttp.ClientSession):
        """단일 자격증 레코드 처리 (비동기)"""
        # JHNT_CTN을 키로 사용
        jhnt_ctn = item.get("JHNT_CTN", "")
        if not jhnt_ctn:
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
        
        # seek_id는 HOPE_JSCD1 매핑을 위해 사용 (JHNT_MBN이 있으면 사용, 없으면 None)
        seek_id = item.get("JHNT_MBN", "") or item.get("SEEK_CUST_NO", "")
        
        # HOPE_JSCD1 정보 가져와서 직종명으로 변환
        hope_jscd1 = self.hope_jscd1_map.get(seek_id, "")
        job_name = self.get_job_name_from_code(hope_jscd1)
        job_examples = []  # 필요시 HOPE_JSCD1로부터 직종 예시 리스트 생성 가능
        
        # 점수 계산 (비동기)
        score, _ = await self.llm_scorer.score_async("자격증", job_name, job_examples, text, session)
        
        # score만 반환 (그래프 변수명과 일치)
        return {
            "JHNT_CTN": str(jhnt_ctn),  # 문자열로 변환
            "certification_score": score  # 그래프: certification_score
        }
    
    async def _preprocess_certification(self, data):
        """자격증 특화 전처리 (비동기 병렬 처리)"""
        if not isinstance(data, list):
            data = [data]
        
        # 비동기 병렬 처리로 각 레코드 처리
        rows = []
        import logging
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for item in data:
                task = self._process_single_certification(item, session)
                tasks.append(task)
            
            results = await atqdm.gather(*tasks, desc="자격증 전처리", unit="건")
            
            for idx, result in enumerate(results):
                try:
                    if result is not None:
                        rows.append(result)
                except Exception as e:
                    item = data[idx]
                    jhnt_ctn = item.get("JHNT_CTN", "unknown")
                    print(f"⚠️ 자격증 처리 오류 (JHNT_CTN: {jhnt_ctn}): {e}")
                    rows.append({
                        "JHNT_CTN": str(jhnt_ctn),  # 문자열로 변환
                        "certification_score": None
                    })
        
        # DataFrame 생성 전에 Logger 객체 확인 및 제거
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


    def load_and_preprocess_data(self, data_file, json_name=None, limit_data=False, limit_size=5000):
        """
        데이터를 로드하고 전처리하는 함수
        
        Args:
            data_file (str): 데이터 파일 경로
            sheet_name (str): 엑셀 시트명 (Excel 파일용)
            json_name (str): JSON 데이터 타입 ('이력서', '자기소개서', '직업훈련', '자격증')
            limit_data (bool): 테스트 모드로 데이터 제한 여부
            limit_size (int): 제한할 데이터 크기
        
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        # 데이터 로드
        if data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
            # 테스트 모드일 경우 CSV 파일도 제한
            if limit_data and len(data) > limit_size:
                original_count = len(data)
                data = data.head(limit_size)
                print(f"📊 CSV 데이터 제한: {len(data)}개 행 사용 (전체 {original_count}개 중 앞 {limit_size}개)")
            data_processed = self.basic_preprocessing(data)
        elif data_file.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(data_file, sheet_name=self.sheet_name)
            # 테스트 모드일 경우 Excel 파일도 제한
            if limit_data and len(data) > limit_size:
                original_count = len(data)
                data = data.head(limit_size)
                print(f"📊 Excel 데이터 제한: {len(data)}개 행 사용 (전체 {original_count}개 중 앞 {limit_size}개)")
            data_processed = self.basic_preprocessing(data)
        elif data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON 파일의 경우 json_name을 데이터 타입으로 사용 (비동기)
                data_processed = asyncio.run(self.nlp_preprocessing(data, json_name=json_name, limit_data=limit_data, limit_size=limit_size))
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. CSV, Excel 또는 JSON 파일을 사용하세요.")
        
        return data_processed


    async def get_merged_df(self, file_list, limit_data=False, limit_size=5000):
        """
        파일명 리스트를 받아 각 파일을 load_and_preprocess_data로 읽고 self.df_list에 append,
        이후 JHNT_MBN 또는 JHNT_CTN 컬럼 기준으로 순차적으로 조인하여 데이터프레임 반환
        
        첫 번째 파일(CSV)은 순차 처리하고, 나머지 4개 JSON 파일은 병렬로 처리합니다.
        CSV를 먼저 로드하여 상위 직종 소분류를 필터링하고, 해당하는 JHNT_MBN/JHNT_CTN만 사용합니다.

        Args:
            file_list (list): 파일명(str) 리스트
            limit_data (bool): 테스트 모드로 데이터 제한 여부
            limit_size (int): 제한할 데이터 크기
 
        Returns:
            pd.DataFrame: JHNT_MBN 또는 JHNT_CTN 기준으로 조인된 데이터프레임 -> repeat 처리 필요
        """
        self.df_list = []
        result = None
        
        # 필터링된 JHNT_MBN, JHNT_CTN 집합 (초기값은 None - 필터링 안 함)
        filtered_jhnt_mbn_set = None
        filtered_jhnt_ctn_set = None
        
        # 첫 번째 파일(정형 데이터 CSV) 먼저 처리 - HOPE_JSCD1(희망 직종 코드) 정보 저장 및 필터링
        if file_list:
            # 첫 번째 파일은 정형 데이터이므로 json_name=None
            csv_start_time = time.time()
            print(f"[DEBUG] 첫 번째 파일 처리 시작: {file_list[0]}, 타입: 정형 데이터 (CSV)")
            df = self.load_and_preprocess_data(file_list[0], json_name=None, limit_data=limit_data, limit_size=limit_size)
            
            # 상위 직종 소분류 필터링 (top_job_categories가 -1이 아니고 HOPE_JSCD1이 있는 경우)
            if self.top_job_categories != -1 and 'HOPE_JSCD1' in df.columns:
                print(f"\n📊 직종 소분류 필터링 시작 (상위 {self.top_job_categories}개)")
                print("="*60)
                
                # HOPE_JSCD1 빈도수 계산 (결측치 제외)
                job_counts = df['HOPE_JSCD1'].value_counts()
                print(f"전체 직종 소분류 수: {len(job_counts)}개")
                
                # 상위 N개 선택
                top_jobs = job_counts.head(self.top_job_categories)
                top_job_codes = set(top_jobs.index.tolist())
                
                print(f"\n상위 {self.top_job_categories}개 직종 소분류:")
                for idx, (job_code, count) in enumerate(top_jobs.items(), 1):
                    job_name = self.get_job_name_from_code(job_code)
                    print(f"  {idx}. {job_code} ({job_name}): {count}건")
                
                # 필터링된 데이터프레임 생성
                original_count = len(df)
                df = df[df['HOPE_JSCD1'].isin(top_job_codes)].copy()
                filtered_count = len(df)
                
                print(f"\n필터링 결과: {original_count}건 → {filtered_count}건 ({filtered_count/original_count*100:.1f}%)")
                print("="*60)
                
                # 필터링된 JHNT_MBN, JHNT_CTN 추출
                if 'JHNT_MBN' in df.columns:
                    filtered_jhnt_mbn_set = set(df['JHNT_MBN'].dropna().unique())
                    print(f"필터링된 JHNT_MBN 수: {len(filtered_jhnt_mbn_set)}개")
                if 'JHNT_CTN' in df.columns:
                    filtered_jhnt_ctn_set = set(df['JHNT_CTN'].dropna().unique())
                    print(f"필터링된 JHNT_CTN 수: {len(filtered_jhnt_ctn_set)}개")
            else:
                if self.top_job_categories == -1:
                    print("📊 직종 소분류 필터링 비활성화 (전체 사용)")
                else:
                    print("⚠️ HOPE_JSCD1 컬럼이 없어 직종 소분류 필터링을 건너뜁니다.")
            
            csv_elapsed = time.time() - csv_start_time
            print(f"⏱️ 정형 데이터(CSV) 처리 소요 시간: {csv_elapsed:.2f}초")
            self.df_list.append(df)
            result = df
            
            print(f"[DEBUG] 첫 번째 데이터프레임 크기: {result.shape}")
            print(f"[DEBUG] 첫 번째 데이터프레임 컬럼: {list(result.columns)}")
            print(f"[DEBUG] 첫 번째 데이터프레임에 JHNT_MBN 존재: {'JHNT_MBN' in result.columns}")
            print(f"[DEBUG] 첫 번째 데이터프레임에 JHNT_CTN 존재: {'JHNT_CTN' in result.columns}")
            
            # HOPE_JSCD1 정보를 JHNT_MBN 기준으로 매핑하여 저장
            if 'HOPE_JSCD1' in df.columns and 'JHNT_MBN' in df.columns:
                self.hope_jscd1_map = df.set_index('JHNT_MBN')['HOPE_JSCD1'].to_dict()
                print(f"[DEBUG] HOPE_JSCD1 매핑 생성 완료: {len(self.hope_jscd1_map)}개")
            else:
                print(f"[DEBUG] 경고: HOPE_JSCD1 또는 JHNT_MBN이 없어 매핑을 생성할 수 없습니다.")
        
        # 나머지 4개 파일을 비동기 병렬로 처리
        async def process_json_file_async(file_info):
            """단일 JSON 파일 처리 함수 (비동기)"""
            file, json_name, idx = file_info
            try:
                file_start_time = time.time()
                print(f"[DEBUG] {idx+1}번째 파일 처리 시작: {file}, 타입: {json_name}")
                # JSON 파일 로드
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 필터링된 키값만 사용 (필터링이 활성화된 경우)
                if isinstance(data, list):
                    # 먼저 키값들을 문자열로 통일
                    for item in data:
                        if "JHNT_MBN" in item:
                            item["JHNT_MBN"] = str(item["JHNT_MBN"])
                        if "JHNT_CTN" in item:
                            item["JHNT_CTN"] = str(item["JHNT_CTN"])
                        if "SEEK_CUST_NO" in item:
                            item["SEEK_CUST_NO"] = str(item["SEEK_CUST_NO"])
                    
                    original_count = len(data)
                    filtered_data = data
                    
                    # 이력서와 자기소개서는 JHNT_MBN 또는 SEEK_CUST_NO로 필터링
                    if json_name in ['이력서', '자기소개서']:
                        if filtered_jhnt_mbn_set is not None:
                            # 디버깅 로그
                            print(f"  [{json_name}] filtered_jhnt_mbn_set 길이: {len(filtered_jhnt_mbn_set)}, 처음 5개: {list(filtered_jhnt_mbn_set)[:5]}")
                            print(f"  [{json_name}] data 처음 5개 JHNT_MBN: {[item.get('JHNT_MBN', '') for item in data[:5]]}")
                            print(f"  [{json_name}] data 처음 5개 SEEK_CUST_NO: {[item.get('SEEK_CUST_NO', '') for item in data[:5]]}")
                            
                            filtered_data = [
                                item for item in data
                                if item.get('JHNT_MBN', '') in filtered_jhnt_mbn_set or 
                                   item.get('SEEK_CUST_NO', '') in filtered_jhnt_mbn_set
                            ]
                            print(f"  [{json_name}] 필터링: {original_count}건 → {len(filtered_data)}건 (JHNT_MBN/SEEK_CUST_NO 기준)")
                    
                    # 직업훈련과 자격증은 JHNT_CTN으로 필터링
                    elif json_name in ['직업훈련', '자격증']:
                        if filtered_jhnt_ctn_set is not None:
                            filtered_data = [
                                item for item in data
                                if item.get('JHNT_CTN', '') in filtered_jhnt_ctn_set
                            ]
                            print(f"  [{json_name}] 필터링: {original_count}건 → {len(filtered_data)}건 (JHNT_CTN 기준)")
                    
                    data = filtered_data
                
                # 비동기 전처리
                df = await self.nlp_preprocessing(data, json_name=json_name, limit_data=limit_data, limit_size=limit_size)
                file_elapsed = time.time() - file_start_time
                print(f"[DEBUG] {json_name} 데이터프레임 크기: {df.shape}")
                print(f"[DEBUG] {json_name} 데이터프레임 컬럼: {list(df.columns)}")
                print(f"⏱️ {json_name} 처리 소요 시간: {file_elapsed:.2f}초")
                return (json_name, df, idx, file_elapsed)
            except Exception as e:
                print(f"⚠️ {json_name} 파일 처리 오류: {e}")
                raise
        
        json_files = []
        for idx, file in enumerate(file_list[1:], start=0):
            if idx >= len(self.json_names):
                raise IndexError(f"JSON 파일 수({len(file_list)-1})가 json_names 길이({len(self.json_names)})를 초과합니다. file: {file}")
            current_json_name = self.json_names[idx]
            json_files.append((file, current_json_name, idx))
        
        # 비동기 병렬 처리로 4개 파일 동시 처리
        tasks = [process_json_file_async(file_info) for file_info in json_files]
        results = await asyncio.gather(*tasks)
        
        processed_dfs = {}
        json_file_times = {}
        for json_name, df, idx, file_elapsed in results:
            processed_dfs[idx] = (json_name, df)
            json_file_times[json_name] = file_elapsed
        
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
        import logging
        
        for idx in tqdm(sorted(processed_dfs.keys()), desc="데이터 병합", unit="파일"):
            json_name, df = processed_dfs[idx]
            self.df_list.append(df)
            
            # 직업훈련과 자격증은 JHNT_CTN 기준으로 merge
            if json_name in ['직업훈련', '자격증']:
                merge_key = "JHNT_CTN"
            else:
                merge_key = "JHNT_MBN"
            
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
            
            # 병합 키를 문자열로 통일 (타입 불일치 방지)
            if merge_key in result.columns:
                result[merge_key] = result[merge_key].astype(str)
            if merge_key in df.columns:
                df[merge_key] = df[merge_key].astype(str)
            
            # 병합 전에 Logger 객체가 있는지 확인
            for col in df.columns:
                if df[col].dtype == 'object' and len(df) > 0:
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        first_val = non_null_values.iloc[0]
                        if isinstance(first_val, logging.Logger) or 'Logger' in str(type(first_val)):
                            print(f"⚠️ [병합 전] {json_name}의 컬럼 '{col}'에 Logger 객체 발견! (타입: {type(first_val).__name__})")
                            # Logger 객체를 NaN으로 대체
                            df[col] = df[col].apply(lambda x: np.nan if (isinstance(x, logging.Logger) or 'Logger' in str(type(x))) else x)
            
            # 테이블을 기준으로 inner join
            result = result.merge(df, on=merge_key, how="inner", suffixes=('', f'_df{idx+1}'))
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
        
        # 병합 후 결측치가 존재하는 row의 비율 확인 (inner join으로 인한 결측치 확인)
        total_rows = len(result)
        rows_with_missing = result.isnull().any(axis=1).sum()
        missing_ratio = (rows_with_missing / total_rows * 100) if total_rows > 0 else 0
        print(f"\n📊 병합 후 결측치 분석:")
        print(f"   전체 행 수: {total_rows}개")
        print(f"   결측치가 있는 행 수: {rows_with_missing}개")
        print(f"   결측치가 있는 행 비율: {missing_ratio:.2f}%")
        
        # 컬럼별 결측치 비율도 출력
        missing_by_column = result.isnull().sum()
        columns_with_missing = missing_by_column[missing_by_column > 0]
        if len(columns_with_missing) > 0:
            print(f"\n📊 컬럼별 결측치 현황:")
            for col, missing_count in columns_with_missing.items():
                missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
                print(f"   {col}: {missing_count}개 ({missing_pct:.2f}%)")
        
        # 범주형으로 처리해야 할 컬럼들을 문자열로 변환 (최빈값 보간을 위해)
        categorical_cols = ['HOPE_JSCD1', 'HOPE_JSCD2', 'HOPE_JSCD3']
        for col in categorical_cols:
            if col in result.columns:
                result[col] = result[col].astype(str).replace('nan', np.nan)
        
        # 결측치 보간 (평균값 또는 최빈값으로)
        from . import utils
        print(f"\n📊 결측치 보간 시작...")
        result = utils.impute_missing_values(result)
        print(f"✅ 결측치 보간 완료")
        
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