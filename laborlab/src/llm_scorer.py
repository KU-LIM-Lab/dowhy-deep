"""
LLM 기반 점수 계산 모듈
"""
import json
from typing import List, Tuple, Optional

# ollama는 optional dependency - 없어도 작동하도록 처리
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

from .llm_reference import HR_SYSTEM_PROMPT, FEWSHOT_EXAMPLES, SCORING_KEYWORDS, TYPO_CHECK_SYSTEM_PROMPT, TYPO_CHECK_USER_PROMPT


class LLMScorer:
    """LLM을 사용한 점수 계산 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        # api_key는 더 이상 사용하지 않지만 하위 호환성을 위해 유지
        pass
    
        
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
            
            resp = ollama.chat(
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
            
            resp = ollama.chat(
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

