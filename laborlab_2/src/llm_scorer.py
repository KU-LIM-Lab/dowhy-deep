"""
LLM 기반 점수 계산 모듈
"""
import os
import asyncio
import aiohttp
import logging
import re
from typing import List, Optional

# ollama는 optional dependency - 없어도 작동하도록 처리
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

from .llm_reference import HR_SYSTEM_PROMPT, FEWSHOT_EXAMPLES, TYPO_CHECK_SYSTEM_PROMPT, TYPO_CHECK_USER_PROMPT

# 로거 설정
logger = logging.getLogger(__name__)


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


    def _parse_llm_response(self, content: str, section: str) -> Optional[int]:
        """
        LLM 응답을 파싱하는 함수 (숫자만 추출)
        
        Args:
            content: LLM 응답 내용
            section: 섹션 이름 (로깅용)
        
        Returns:
            Optional[int]: 0-100 범위의 점수 또는 None
        """
        if not content or not content.strip():
            return None
        
        content = content.strip()
        
        # 정규표현식으로 숫자 추출 (0-100 범위)
        try:
            # 0-100 범위의 숫자 찾기
            score_match = re.search(r'\b([0-9]|[1-9][0-9]|100)\b', content)
            if score_match:
                score = int(score_match.group())
                return max(0, min(100, score))
        except Exception as e:
            logger.debug(f"[{section}] 정규표현식 파싱 실패: {e}")
        
        return None
    
    def _build_prompt(self, section: str, job_name: str, job_examples: List[str], text: str) -> str:
        """LLM 프롬프트 구축"""
        def shot(s):
            return f"[예시]\n직무: {s['input']['job']}\n자료:\n{s['input']['text']}\n=> {s['output']}"
        
        examples = "\n\n".join(shot(s) for s in FEWSHOT_EXAMPLES.get(section, []))
        job_hint = f"참고 직무 예시: {', '.join(job_examples[:12])}" if job_examples else "참고 직무 예시: 없음"
        
        return f"""[평가 섹션] {section}
[목표 직종] {job_name}
{job_hint}

{examples}

[지원자 자료]
{text}

[응답 형식] 0-100 사이의 정수 하나만 출력 (다른 텍스트 절대 금지)
올바른 예시: 75, 0, 100, 50
잘못된 예시: -5, 105, "75점", "점수는 75입니다", 75.5
"""
    
    def _score_with_llm(self, section: str, job_name: str, job_examples: List[str], text: str) -> int:
        """LLM을 사용한 점수 계산"""
        if not OLLAMA_AVAILABLE or ollama is None:
            return 50
            
        content = None
        try:
            sys_msg = {"role": "system", "content": HR_SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": self._build_prompt(section, job_name, job_examples, text)}
            
            client = ollama.Client(host=self.ollama_host)
            
            resp = client.chat(
                model="llama3.2:1b",
                messages=[sys_msg, user_msg],
                options={"temperature": 0.2},
                stream=False
            )
            
            content = resp["message"]["content"]
            score = self._parse_llm_response(content, section)
            
            if score is not None:
                return score
            else:
                raise ValueError(f"LLM 응답 파싱 실패: {content[:100]}")
            
        except Exception as e:
            if content is not None:
                logger.error(f"[{section}] LLM 응답 파싱 실패 - 응답: {content[:200]}, 에러: {type(e).__name__}: {e}")
            else:
                logger.error(f"[{section}] LLM API 호출 실패 - 에러: {type(e).__name__}: {e}")
            return 50
    
    def score(self, section: str, job_name: str, job_examples: List[str], text: str) -> int:
        """점수 계산 메인 메서드 - int 반환 (동기 버전)"""
        return self._score_with_llm(section, job_name, job_examples, text)
    
    async def score_async(self, section: str, job_name: str, job_examples: List[str], text: str, session: aiohttp.ClientSession) -> int:
        """점수 계산 메인 메서드 - int 반환 (비동기 버전)"""
        if not OLLAMA_AVAILABLE:
            return 50
            
        content = None
        try:
            sys_msg = {"role": "system", "content": HR_SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": self._build_prompt(section, job_name, job_examples, text)}
            
            url = f"http://{self.ollama_host}/api/chat"
            payload = {
                "model": "llama3.2:1b",
                "messages": [sys_msg, user_msg],
                "options": {"temperature": 0.2},
                "stream": False
            }
            
            # 각 LLM 요청마다 3분(180초) timeout 설정
            timeout = aiohttp.ClientTimeout(total=180)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                result = await resp.json()
                content = result["message"]["content"]
                
                score = self._parse_llm_response(content, section)
                
                if score is not None:
                    return score
                else:
                    raise ValueError(f"LLM 응답 파싱 실패: {content[:100]}")
                
        except asyncio.TimeoutError as e:
            logger.error(f"[{section}] LLM API 호출 타임아웃 (3분 초과) - 에러: {type(e).__name__}: {e}")
            return 50
        except Exception as e:
            if content is not None:
                logger.error(f"[{section}] LLM 응답 파싱 실패 - 응답: {content[:200]}, 에러: {type(e).__name__}: {e}")
            else:
                logger.error(f"[{section}] LLM API 호출 실패 - 에러: {type(e).__name__}: {e}")
            return 50

