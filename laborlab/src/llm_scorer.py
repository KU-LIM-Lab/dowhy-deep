"""
LLM 기반 점수 계산 모듈
"""

import os
import json
from typing import List, Tuple, Optional
from config import HR_SYSTEM_PROMPT, FEWSHOT_EXAMPLES, SCORING_KEYWORDS


class LLMScorer:
    """LLM을 사용한 점수 계산 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def _offline_score(self, section: str, text: str, job_examples: List[str]) -> Tuple[int, str]:
        """오프라인 점수 계산 (키워드 기반)"""
        if not text or text.strip() == "정보 없음":
            return 10, "자료 부족"
        
        hits = sum(1 for k in SCORING_KEYWORDS if k.lower() in text.lower())
        base = 45
        return min(95, base + hits*5), f"키워드 {hits}개 매칭"
    
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
        if not self.api_key:
            return self._offline_score(section, text, job_examples)
        
        try:
            os.environ["OPENAI_API_KEY"] = self.api_key
            from openai import OpenAI
            client = OpenAI()
            
            sys_msg = {"role": "system", "content": HR_SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": self._build_prompt(section, job_name, job_examples, text)}
            
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[sys_msg, user_msg],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            
            content = resp.choices[0].message.content
            data = json.loads(content)
            score = int(max(0, min(100, int(data.get("score", 0)))))
            why = str(data.get("rationale", ""))[:240]
            return score, why
            
        except Exception:
            return self._offline_score(section, text, job_examples)
    
    def score(self, section: str, job_name: str, job_examples: List[str], text: str) -> Tuple[int, str]:
        """점수 계산 메인 메서드"""
        return self._score_with_llm(section, job_name, job_examples, text)

