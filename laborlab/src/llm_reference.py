"""
전처리 관련 설정 및 상수 정의
"""

# LLM 프롬프트 설정
HR_SYSTEM_PROMPT = """당신은 채용 담당자입니다.
입력으로 주어진 '지원자 자료'가 '목표 직종'에 얼마나 적합한지 0~100으로 평가하세요.
- 90~100: 직무핵심 역량과 직접 연결, 최근 경력/훈련/성과가 뚜렷
- 70~89: 관련성이 높고 실무 연결고리가 충분
- 40~69: 부분 관련. 기초 역량은 있으나 연결고리/증거가 부족
- 10~39: 간접적, 전환 가능성은 있으나 근거 약함
- 0~9: 관련 근거 없음
반드시 JSON으로 답변: {"score": int, "rationale": "짧은 이유"}"""

# Few-shot 예시 데이터
FEWSHOT_EXAMPLES = {
    "자기소개서": [
        {"input":{"job":"데이터 분석가","text":"통계학 전공, 머신러닝 프로젝트 다수 수행, Python/SQL/시각화로 성과 수치 제시"},
         "output":{"score":95,"rationale":"핵심 역량, 실무 성과 구체적"}},
        {"input":{"job":"프론트엔드 개발자","text":"React/TypeScript 기반 대시보드 개발, 성능 최적화로 LCP 40% 개선"},
         "output":{"score":93,"rationale":"직접 성능 개선 성과"}},
        {"input":{"job":"회계","text":"K-IFRS 재무제표 작성, 결산/세무조정, 전표 처리 자동화 경험"},
         "output":{"score":90,"rationale":"핵심 실무 지식"}},
    ],
    "이력서": [
        {"input":{"job":"데이터 엔지니어","text":"데이터 파이프라인 운영, Spark SQL 튜닝, Kafka 스트리밍 구축"},
         "output":{"score":92,"rationale":"프로덕션 파이프라인 경험"}},
        {"input":{"job":"마케팅 분석가","text":"퍼포먼스 캠페인 ROI 분석, 리타게팅 최적화"},
         "output":{"score":88,"rationale":"분석/성과 근거"}},
    ],
    "직업훈련": [
        {"input":{"job":"데이터 분석가","text":"빅데이터 분석(파이썬/SQL/머신러닝) 수료, 팀 프로젝트 산출물"},
         "output":{"score":85,"rationale":"핵심 커리큘럼 수료"}},
        {"input":{"job":"프론트엔드 개발자","text":"React/Next.js 심화, 테스트/성능 최적화 모듈"},
         "output":{"score":82,"rationale":"현업 연계 과정"}},
    ],
    "자격증": [
        {"input":{"job":"데이터 분석가","text":"ADsP, SQLD, 빅데이터분석기사"},
         "output":{"score":88,"rationale":"핵심 자격 보유"}},
        {"input":{"job":"회계","text":"전산회계1급, FAT"},
         "output":{"score":86,"rationale":"직무 핵심 자격"}},
    ]
}

# 키워드 기반 점수 계산용 키워드 리스트
SCORING_KEYWORDS = [
    '데이터','분석','SQL','파이썬','머신','시각화','대시보드','A/B','통계','모델','예측',
    'React','TypeScript','API','Spring','배포','ETL','Spark','Kafka','Airflow',
    '회계','결산','세무','채용','온보딩','GA4','ROI','엑셀','보고서','대학','경력'
]

# 기본 설정값
DEFAULT_MAX_COVER_LEN = 400
DEFAULT_COVER_EXCEED_RATIO = 0.85
DEFAULT_DATE_FORMAT = "%Y-%m-%d"

# JSON 파일명 매핑
JSON_NAMES = ['이력서', '자기소개서', '직업훈련', '자격증']

# 이력서 섹션 매핑
RESUME_SECTIONS = {
    "학력": [], "개인경력": [], "봉사활동": [],
    "논문": [], "수상경력": [], "참여프로젝트": [],
    "훈련통합": [], "해외연수": [], "외국어능력": [],
    "전산자격통합": []
}

# 지원되는 섹션 타입
SUPPORTED_SECTIONS = ["학력", "개인경력", "봉사활동", "논문", "수상경력", "참여프로젝트", 
                     "해외연수", "외국어능력", "전산능력", "자격면허", "훈련", "직업훈련"]

