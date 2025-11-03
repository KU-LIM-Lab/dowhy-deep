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

from config import (
    JSON_NAMES, RESUME_SECTIONS, SUPPORTED_SECTIONS, 
    DEFAULT_MAX_COVER_LEN, DEFAULT_COVER_EXCEED_RATIO, DEFAULT_DATE_FORMAT
)
from llm_scorer import LLMScorer



class Preprocessor:
    def __init__(self, df_list, api_key=None):
        self.json_names = JSON_NAMES
        self.df_list = []
        self.variable_mapping = self.load_variable_mapping()
        self.llm_scorer = LLMScorer(api_key)

    def load_variable_mapping(self):
        with open('../data/variable_mapping.json', encoding='utf-8') as f:
            variable_mapping = json.load(f)
        return variable_mapping

    @staticmethod
    def _parse_date(s: Optional[str]) -> Optional[datetime]:
        """날짜 문자열을 datetime 객체로 변환"""
        if s in (None, "", "null"):
            return None
        try:
            return datetime.strptime(str(s)[:10], DEFAULT_DATE_FORMAT)
        except Exception:
            return None

    @staticmethod
    def _days_between(d1: Optional[datetime], d2: Optional[datetime]) -> Optional[int]:
        """두 날짜 사이의 일수 계산"""
        if d1 is None or d2 is None:
            return None
        return (d1 - d2).days

    @staticmethod
    def _estimate_typos_korean(text: str) -> int:
        """한국어 텍스트의 오타 추정"""
        if not text:
            return 0
        dbl_spaces = len(re.findall(r" {2,}", text))
        repeat_punct = len(re.findall(r"([\.?!,~\-])\1{2,}", text))
        latin_tokens = re.findall(r"\b[A-Za-z]{2,}\b", text)
        return dbl_spaces + repeat_punct + len(latin_tokens)

    def _build_resume_sections(self, data):
        """이력서 섹션을 구축하는 헬퍼 메서드"""
        sections = RESUME_SECTIONS.copy()
        
        for resume in data.get("RESUMES", []):
            for it in (resume.get("ITEMS") or []):
                sec = it.get("RESUME_ITEM_CLCD") or it.get("DS_RESUME_ITEM_CLCD") or ""
                nm = it.get("RESUME_ITEM_1_NM") or ""
                val = it.get("RESUME_ITEM_1_VAL") or ""
                st = it.get("HIST_STDT") or ""
                en = it.get("HIST_ENDT") or ""
                rec = {"sec": sec, "name": nm, "value": val, "start": st, "end": en}
                sec_norm = sec.replace(" ", "") if isinstance(sec, str) else sec
                if sec_norm in SUPPORTED_SECTIONS:
                    if sec_norm in ["전산능력", "자격면허"]:
                        sections["전산자격통합"].append(rec)
                    elif sec_norm in ["훈련", "직업훈련"]:
                        sections["훈련통합"].append(rec)
                    else:
                        sections[sec_norm].append(rec)
        return sections

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
        
        # LLM scorer에 전달하여 점수 계산
        score, _ = self.llm_scorer.score(selection="이력서", job_name="미상", job_examples=[], text=formatting_sentence)
        
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
                
            # 자기소개서 데이터 추출
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
            lens = [len(it.get("SELF_INTRO_CONT", "") or "") for it in items] if items else []
            max_len = max(lens) if lens else 0
            typo = sum(self._estimate_typos_korean(it.get("SELF_INTRO_CONT", "") or "") for it in items) if items else 0
            
            # 점수 계산
            score, why = self.llm_scorer.score("자기소개서", "미상", [], full_text)
            
            rows.append({
                "SEEK_CUST_NO": seek_id,
                "cover_items_count": len(items),
                "cover_max_chars": max_len,
                "cover_exceed_85pct": int(max_len >= DEFAULT_MAX_COVER_LEN * DEFAULT_COVER_EXCEED_RATIO),
                "cover_typo_count": typo,
                "cover_score": score,
                "cover_why": why
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
                
            # 이력서에서 훈련 섹션 추출
            secs = self._build_resume_sections(item)
            resume_train = secs.get("훈련통합", [])
            
            # JSON에서 훈련 데이터 추출
            tr_json = item.get("TRAININGS", [])
            
            def _to_rec_from_json(t):
                return {
                    "name": t.get("TRNG_NM") or "",
                    "start": t.get("TRNG_BGDE") or "",
                    "end": t.get("TRNG_ENDE") or ""
                }
            
            def _to_rec_from_resume(t):
                return {
                    "name": t.get("name") or t.get("value") or "",
                    "start": t.get("start") or "",
                    "end": t.get("end") or ""
                }
            
            # JSON과 이력서 데이터 결합
            combined = [_to_rec_from_json(t) for t in tr_json] + [_to_rec_from_resume(t) for t in resume_train]
            
            # 중복 제거
            seen = set()
            uniq = []
            for r in combined:
                key = (r["name"], r["start"], r["end"])
                if key not in seen:
                    seen.add(key)
                    uniq.append(r)
            
            # 마지막 훈련 종료일 계산
            ends = [self._parse_date(r["end"]) for r in uniq if r.get("end")]
            ends = [d for d in ends if d]
            last_end = max(ends).strftime("%Y-%m-%d") if ends else None
            
            # 구직 등록일과의 간격 계산
            jobseek = item.get("JHCR_DE")
            gap = self._days_between(self._parse_date(jobseek), self._parse_date(last_end)) if (jobseek and last_end) else None
            
            # 텍스트 생성
            text = "\n".join([f"{r['name']} ({r['start']}~{r['end']})" for r in uniq]) if uniq else "정보 없음"
            
            # 점수 계산
            score, why = self.llm_scorer.score("직업훈련", "미상", [], text)
            
            rows.append({
                "SEEK_CUST_NO": seek_id,
                "training_count_total": len(uniq),
                "training_last_end": last_end,
                "jobseek_date": jobseek,
                "days_last_training_to_jobseek": gap,
                "training_score": score,
                "training_why": why
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
                
            # 이력서에서 자격증 섹션 추출
            secs = self._build_resume_sections(item)
            resume_itlic = secs.get("전산자격통합", [])
            
            # JSON에서 자격증 데이터 추출
            lic_json = item.get("LICENSES", [])
            
            def _to_rec_from_json(l):
                return {
                    "cat": l.get("QULF_LCNS_LCFN") or "",
                    "name": l.get("QULF_LCNS_NM") or "",
                    "acq": l.get("ACQ_DE") or ""
                }
            
            def _to_rec_from_resume(l):
                return {
                    "cat": l.get("sec") or "",
                    "name": l.get("name") or l.get("value") or "",
                    "acq": l.get("end") or ""
                }
            
            # JSON과 이력서 데이터 결합
            combined = [_to_rec_from_json(l) for l in lic_json] + [_to_rec_from_resume(l) for l in resume_itlic]
            
            # 중복 제거
            seen = set()
            uniq = []
            for r in combined:
                key = (r["cat"], r["name"], r["acq"])
                if key not in seen:
                    seen.add(key)
                    uniq.append(r)
            
            # 자격증 카테고리 분석
            cats = [r["cat"] for r in uniq if r.get("cat")]
            cnt = Counter(cats) if cats else Counter()
            has_nat_tech = int(cnt.get("국가기술자격", 0) > 0)
            has_nat_prof = int(cnt.get("국가전문자격", 0) > 0)
            has_priv = int(cnt.get("민간자격", 0) > 0)
            
            top_cat = None
            if cnt:
                top_cat = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            
            # 텍스트 생성
            text = "\n".join([f"{r['cat']} - {r['name']} (취득:{r['acq']})" for r in uniq]) if uniq else "정보 없음"
            
            # 점수 계산
            score, why = self.llm_scorer.score("자격증", "미상", [], text)
            
            rows.append({
                "SEEK_CUST_NO": seek_id,
                "license_total": len(uniq),
                "has_국가기술자격": has_nat_tech,
                "has_국가전문자격": has_nat_prof,
                "has_민간자격": has_priv,
                "top_license_category": top_cat,
                "license_score": score,
                "license_why": why
            })
        
        return pd.DataFrame(rows)


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