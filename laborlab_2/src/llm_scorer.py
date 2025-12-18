"""
LLM 기반 점수 계산 모듈
"""
import json
import os
import asyncio
import aiohttp
from typing import List, Tuple

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
        """오탈자 개수 계산 로직 비활성화."""
        # 아래는 기존 오탈자 계산 로직(주석 처리)
        # if not text or text.strip() == "":
        #     return 0
        # if not OLLAMA_AVAILABLE or ollama is None:
        #     return 0
        # try:
        #     sys_msg = {"role": "system", "content": TYPO_CHECK_SYSTEM_PROMPT}
        #     user_msg = {"role": "user", "content": TYPO_CHECK_USER_PROMPT.format(text=text)}
        #     client = ollama.Client(host=self.ollama_host)
        #     resp = client.chat(
        #         model="llama3.2:1b",
        #         messages=[sys_msg, user_msg],
        #         options={"temperature": 0.1},
        #         stream=False
        #     )
        #     content = resp["message"]["content"]
        #     data = json.loads(content)
        #     return int(max(0, int(data.get("typo_count", 0))))
        # except Exception as e:
        #     print(f"오탈자 개수 계산 실패: {type(e).__name__}: {e}")
        #     return 0
        return 0
    
    async def count_typos_async(self, text: str, session: aiohttp.ClientSession) -> int:
        """오탈자 개수 계산 로직 비활성화 (비동기)."""
        # 아래는 기존 오탈자 계산 로직(주석 처리)
        # if not text or text.strip() == "":
        #     return 0
        # try:
        #     sys_msg = {"role": "system", "content": TYPO_CHECK_SYSTEM_PROMPT}
        #     user_msg = {"role": "user", "content": TYPO_CHECK_USER_PROMPT.format(text=text)}
        #     url = f"http://{self.ollama_host}/api/chat"
        #     payload = {
        #         "model": "llama3.2:1b",
        #         "messages": [sys_msg, user_msg],
        #         "options": {"temperature": 0.1},
        #         "stream": False
        #     }
        #     async with session.post(url, json=payload) as resp:
        #         resp.raise_for_status()
        #         result = await resp.json()
        #         content = result["message"]["content"]
        #         data = json.loads(content)
        #         return int(max(0, int(data.get("typo_count", 0))))
        # except Exception as e:
        #     print(f"오탈자 개수 계산 실패: {type(e).__name__}: {e}")
        #     return 0
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

[응답 형식] score(0-100 정수)※rationale(간단한 이유)
[응답 예제] 75※직무 관련 경력이 있습니다
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
                options={"temperature": 0.2},
                stream=False
            )
            
            content = resp["message"]["content"]
            score = content.split("※")[0].strip()
            why = content.split("※")[1].strip()
            score = int(max(0, min(100, int(score))))
            return score, why
            
        except Exception as e:
            # 오류 발생 시 기본값 반환
            return 50, f"LLM API 오류: {str(e)[:200]}"
    
    def score(self, section: str, job_name: str, job_examples: List[str], text: str) -> Tuple[int, str]:
        """점수 계산 메인 메서드 - (score, rationale) 반환 (동기 버전)"""
        return self._score_with_llm(section, job_name, job_examples, text)
    
    async def score_async(self, section: str, job_name: str, job_examples: List[str], text: str, session: aiohttp.ClientSession) -> Tuple[int, str]:
        """점수 계산 메인 메서드 - (score, rationale) 반환 (비동기 버전)"""
        if not OLLAMA_AVAILABLE:
            return 50, "LLM API 사용 불가 (ollama 패키지 미설치)"
            
        try:
            sys_msg = {"role": "system", "content": HR_SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": self._build_prompt(section, job_name, job_examples, text)}
            
            # Ollama API 엔드포인트 URL 구성
            url = f"http://{self.ollama_host}/api/chat"
            payload = {
                "model": "llama3.2:1b",
                "messages": [sys_msg, user_msg],
                "options": {"temperature": 0.2},
                "stream": False
            }
            
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                content = result["message"]["content"]
                score = content.split("※")[0].strip()
                why = content.split("※")[1].strip()
                score = int(max(0, min(100, int(score))))
                return score, why
                
        except Exception as e:
            return 50, f"LLM API 오류: {str(e)[:200]}"

