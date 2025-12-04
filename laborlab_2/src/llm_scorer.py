"""
LLM 기반 점수 계산 모듈
"""
import json
import os
import logging
from typing import List, Tuple, Dict, Any

# HTTP 요청 로깅 비활성화
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ollama는 optional dependency - 없어도 작동하도록 처리
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

from .llm_reference import HR_SYSTEM_PROMPT, FEWSHOT_EXAMPLES, TYPO_CHECK_SYSTEM_PROMPT, TYPO_CHECK_USER_PROMPT


class LLMScorer:
    """LLM을 사용한 점수 계산 클래스"""
    
    def __init__(self):
        # 무조건 로컬 Ollama만 사용
        # OLLAMA_HOST 환경변수에서 로컬 Ollama 주소 가져오기
        self.ollama_host = self._get_ollama_host()
        if not self.ollama_host:
            raise ValueError("OLLAMA_HOST 환경변수가 설정되지 않았습니다. 로컬 Ollama 주소를 설정하세요.")
    
    def _get_ollama_host(self) -> str:
        """Ollama 호스트 주소 결정 (로컬만 사용)"""
        # 환경변수에서 로컬 Ollama 주소 가져오기
        env_host = os.getenv("OLLAMA_HOST")
        if env_host:
            # http:// 또는 https:// 제거 (ollama 클라이언트가 자동 추가)
            if env_host.startswith("http://"):
                env_host = env_host.replace("http://", "")
            elif env_host.startswith("https://"):
                env_host = env_host.replace("https://", "")
            return env_host
        
        # 환경변수가 없으면 None 반환 (에러 발생)
        return None
    
        
    def count_typos(self, text: str) -> int:
        """TYPO_CHECK 프롬프트를 사용한 오탈자 개수 계산"""
        if not text or text.strip() == "":
            return 0

        if not OLLAMA_AVAILABLE or ollama is None:
            # ollama가 설치되지 않은 경우 기본값 반환
            return 0

        try:
            sys_msg = {"role": "system", "content": TYPO_CHECK_SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": TYPO_CHECK_USER_PROMPT.format(text=text)}
            
            # 로컬 Ollama 클라이언트 생성 (무조건 환경변수에서 가져온 호스트 사용)
            client = ollama.Client(host=self.ollama_host)
            
            resp = client.chat(
                model="llama3.2:1b",
                messages=[sys_msg, user_msg],
                options={"temperature": 0.1}
            )
            
            content = resp["message"]["content"]
            data = json.loads(content)
            typo_count = int(max(0, int(data.get("typo_count", 0))))
            return typo_count
            
        except Exception as e:
            print(f"오탈자 개수 계산 실패: {e}")
            return 0


    def _build_prompt(self, section: str, job_name: str, job_examples: List[str], text: str) -> str:
        """LLM 프롬프트 구축"""
        def shot(s):
            return f"[예시]\n직무: {s['input']['job']}\n자료:\n{s['input']['text']}\n=> {json.dumps(s['output'], ensure_ascii=False)}"
        
        examples = "\n\n".join(shot(s) for s in FEWSHOT_EXAMPLES.get(section, []))
        job_hint = f"참고 직무 예시: {', '.join(job_examples[:12])}" if job_examples else "참고 직무 예시: 없음"
        
        return f"""[평가 섹션] {section}
[목표 직종] {job_name}
{job_hint}

{examples}

[지원자 자료]
{text}

[응답 형식] JSON 한 줄 ({{"score": 0-100 정수, "rationale": "간단한 이유"}})
"""
    
    def _score_with_llm(self, section: str, job_name: str, job_examples: List[str], text: str) -> Tuple[int, str]:
        """LLM을 사용한 점수 계산"""
        if not OLLAMA_AVAILABLE or ollama is None:
            # ollama가 설치되지 않은 경우 기본값 반환
            return 50, "LLM API 사용 불가 (ollama 패키지 미설치)"
            
        try:
            sys_msg = {"role": "system", "content": HR_SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": self._build_prompt(section, job_name, job_examples, text)}
            
            # 로컬 Ollama 클라이언트 생성 (무조건 환경변수에서 가져온 호스트 사용)
            client = ollama.Client(host=self.ollama_host)
            
            resp = client.chat(
                model="llama3.2:1b",
                messages=[sys_msg, user_msg],
                options={"temperature": 0.2}
            )
            
            content = resp["message"]["content"]
            data = json.loads(content)
            score = int(max(0, min(100, int(data.get("score", 0)))))
            why = str(data.get("rationale", ""))[:240]
            return score, why
            
        except Exception as e:
            # 오류 발생 시 기본값 반환
            return 50, f"LLM API 오류: {str(e)[:200]}"
    
    def score(self, section: str, job_name: str, job_examples: List[str], text: str) -> Tuple[int, str]:
        """점수 계산 메인 메서드 - (score, rationale) 반환"""
        return self._score_with_llm(section, job_name, job_examples, text)
    
    def score_batch(self, requests: List[Dict[str, Any]], batch_size: int = 20, desc: str = "점수 계산") -> List[Tuple[int, str]]:
        """
        배치 단위로 점수 계산
        
        Args:
            requests: [{"section": str, "job_name": str, "job_examples": List[str], "text": str}, ...]
            batch_size: 배치 크기 (기본값: 20)
            desc: tqdm 진행 표시 설명 (기본값: "점수 계산")
        
        Returns:
            List[Tuple[int, str]]: [(score, rationale), ...] 리스트
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        results = [None] * len(requests)
        total_batches = (len(requests) + batch_size - 1) // batch_size
        
        # 배치 단위로 처리
        with tqdm(total=len(requests), desc=desc, unit="건", ncols=100) as pbar:
            for batch_start in range(0, len(requests), batch_size):
                batch_end = min(batch_start + batch_size, len(requests))
                batch_requests = requests[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))
                
                # 배치 내에서 병렬 처리
                with ThreadPoolExecutor(max_workers=min(batch_size, len(batch_requests))) as executor:
                    futures = {
                        executor.submit(
                            self._score_with_llm,
                            req["section"],
                            req["job_name"],
                            req["job_examples"],
                            req["text"]
                        ): idx for idx, req in zip(batch_indices, batch_requests)
                    }
                    
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            # 오류 발생 시 기본값 반환
                            results[idx] = (50, f"LLM API 오류: {str(e)[:200]}")
                        finally:
                            pbar.update(1)
        
        return results
    
    def count_typos_batch(self, texts: List[str], batch_size: int = 20, desc: str = "오탈자 계산") -> List[int]:
        """
        배치 단위로 오탈자 개수 계산
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기 (기본값: 20)
            desc: tqdm 진행 표시 설명 (기본값: "오탈자 계산")
        
        Returns:
            List[int]: 오탈자 개수 리스트
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        results = [0] * len(texts)
        
        # 배치 단위로 처리
        with tqdm(total=len(texts), desc=desc, unit="건", ncols=100) as pbar:
            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))
                
                # 배치 내에서 병렬 처리
                with ThreadPoolExecutor(max_workers=min(batch_size, len(batch_texts))) as executor:
                    futures = {
                        executor.submit(self.count_typos, text): idx 
                        for idx, text in zip(batch_indices, batch_texts)
                    }
                    
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            # 오류 발생 시 기본값 반환
                            results[idx] = 0
                        finally:
                            pbar.update(1)
        
        return results

