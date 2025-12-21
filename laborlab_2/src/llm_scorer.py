"""
LLM 기반 점수 계산 모듈
"""
import json
import os
import asyncio
import aiohttp
import logging
import re
from typing import List, Tuple, Optional

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


    def _parse_llm_response(self, content: str, section: str) -> Tuple[Optional[int], Optional[str]]:
        """
        LLM 응답을 파싱하는 함수 (fallback 방식)
        
        Args:
            content: LLM 응답 내용
            section: 섹션 이름 (로깅용)
        
        Returns:
            Tuple[Optional[int], Optional[str]]: (score, rationale) 또는 (None, None)
        """
        if not content or not content.strip():
            return None, None
        
        content = content.strip()
        
        # 방법 1: "::" 구분자로 파싱 시도
        try:
            parts = content.split("::")
            if len(parts) >= 2:
                score_str = parts[0].strip()
                rationale = "::".join(parts[1:]).strip()  # rationale에 "::"가 포함될 수 있음
                
                # 점수 추출 (숫자만)
                score_match = re.search(r'\d+', score_str)
                if score_match:
                    score = int(score_match.group())
                    score = max(0, min(100, score))
                    return score, rationale
        except Exception as e:
            logger.debug(f"[{section}] '::' 구분자 파싱 실패: {e}")
        
        # 방법 2: JSON 형식으로 파싱 시도
        try:
            # JSON 객체 찾기 (중괄호로 감싸진 부분, 중첩 지원)
            # 먼저 전체 텍스트에서 JSON 객체 시도
            json_start = content.find('{')
            if json_start != -1:
                # 중괄호 매칭으로 JSON 객체 끝 찾기
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                    score = data.get("score")
                    rationale = data.get("rationale", "")
                    if score is not None:
                        score = max(0, min(100, int(score)))
                        return score, rationale
        except Exception as e:
            logger.debug(f"[{section}] JSON 형식 파싱 실패: {e}")
        
        # 방법 3: 정규표현식으로 숫자 추출 (0-100 범위)
        try:
            # 0-100 범위의 숫자 찾기
            score_match = re.search(r'\b([0-9]|[1-9][0-9]|100)\b', content)
            if score_match:
                score = int(score_match.group())
                score = max(0, min(100, score))
                # rationale은 점수 다음 텍스트로 추출
                rationale_start = score_match.end()
                rationale = content[rationale_start:].strip()
                # 불필요한 문자 제거
                rationale = re.sub(r'^[^\w가-힣]+', '', rationale)  # 앞의 특수문자 제거
                if rationale:
                    return score, rationale[:200]  # rationale 최대 200자
        except Exception as e:
            logger.debug(f"[{section}] 정규표현식 파싱 실패: {e}")
        
        return None, None
    
    def _build_prompt(self, section: str, job_name: str, job_examples: List[str], text: str) -> str:
        """LLM 프롬프트 구축"""
        def shot(s):
            # "::" 구분자 형식으로 예시 출력
            score = s['output']['score']
            rationale = s['output']['rationale']
            return f"[예시]\n직무: {s['input']['job']}\n자료:\n{s['input']['text']}\n=> {score}::{rationale}"
        
        examples = "\n\n".join(shot(s) for s in FEWSHOT_EXAMPLES.get(section, []))
        job_hint = f"참고 직무 예시: {', '.join(job_examples[:12])}" if job_examples else "참고 직무 예시: 없음"
        
        return f"""[평가 섹션] {section}
[목표 직종] {job_name}
{job_hint}

{examples}

[지원자 자료]
{text}

[응답 형식] score(0-100 정수)::rationale(간단한 이유)
[응답 예제] 75::직무 관련 경력이 있습니다
"""
    
    def _score_with_llm(self, section: str, job_name: str, job_examples: List[str], text: str) -> Tuple[int, str]:
        """LLM을 사용한 점수 계산"""
        if not OLLAMA_AVAILABLE or ollama is None:
            # ollama가 설치되지 않은 경우 기본값 반환
            return 50, "LLM API 사용 불가 (ollama 패키지 미설치)"
            
        content = None
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
            
            # Fallback 방식으로 파싱 시도
            score, why = self._parse_llm_response(content, section)
            
            if score is not None and why is not None:
                return score, why
            else:
                # 파싱 실패 시 에러 발생시켜 except 블록으로 이동
                raise ValueError(f"LLM 응답 파싱 실패: 예상 형식과 일치하지 않음")
            
        except Exception as e:
            # 오류 발생 시 LLM 응답 내용 로깅
            error_msg = f"LLM API 오류: {str(e)[:200]}"
            if content is not None:
                logger.error(f"[{section}] LLM 응답 파싱 실패 - 응답 내용: {content[:500]}")
                logger.error(f"[{section}] LLM 응답 파싱 실패 - 에러: {error_msg}")
            else:
                logger.error(f"[{section}] LLM API 호출 실패 - 에러: {error_msg}")
            return 50, error_msg
    
    def score(self, section: str, job_name: str, job_examples: List[str], text: str) -> Tuple[int, str]:
        """점수 계산 메인 메서드 - (score, rationale) 반환 (동기 버전)"""
        return self._score_with_llm(section, job_name, job_examples, text)
    
    async def score_async(self, section: str, job_name: str, job_examples: List[str], text: str, session: aiohttp.ClientSession) -> Tuple[int, str]:
        """점수 계산 메인 메서드 - (score, rationale) 반환 (비동기 버전)"""
        if not OLLAMA_AVAILABLE:
            return 50, "LLM API 사용 불가 (ollama 패키지 미설치)"
            
        content = None
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
                
                # Fallback 방식으로 파싱 시도
                score, why = self._parse_llm_response(content, section)
                
                if score is not None and why is not None:
                    return score, why
                else:
                    # 파싱 실패 시 에러 발생시켜 except 블록으로 이동
                    raise ValueError(f"LLM 응답 파싱 실패: 예상 형식과 일치하지 않음")
                
        except Exception as e:
            # 오류 발생 시 LLM 응답 내용 로깅
            error_msg = f"LLM API 오류: {str(e)[:200]}"
            if content is not None:
                logger.error(f"[{section}] LLM 응답 파싱 실패 - 응답 내용: {content[:500]}")
                logger.error(f"[{section}] LLM 응답 파싱 실패 - 에러: {error_msg}")
            else:
                logger.error(f"[{section}] LLM API 호출 실패 - 에러: {error_msg}")
            return 50, error_msg

