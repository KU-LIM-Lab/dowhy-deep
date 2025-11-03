
'''
Graph 시각화, Treatment 불러오기 예시
- 각 그래프 dot 파일에 subgraph에 treatment 정의 (아래는 graph1 예시)

subgraph cluster_treatments {
        label="Treatments (outcome: ACQ_180_YN)";
        style=dashed;
        color="#CCCCCC";
        node [shape=note, style="rounded,filled", fillcolor="#FFF7E6", fontname="Helvetica"];

        // T1: 이전 직업훈련 경험 (training exposure)
        T1 [
            label="T1: prev_training_any",
            role="treatment_meta",
            treatment_var="BFR_OCTR_CT",
            treatment_name="prev_training_any",
            treatment_def="BFR_OCTR_CT > 0",
            treatment_question="이전 직업훈련을 한 번이라도 받은 사람은 그렇지 않은 사람보다 6개월 이내 취업 확률이 얼마나 높은가?"
        ];

        // T2: 최종 학력 수준 (education level)
        T2 [
            label="T2: high_education",
            role="treatment_meta",
            treatment_var="ACCR_CD",
            treatment_name="high_education",
            treatment_def="전문대/대졸 이상=1, 고졸 이하=0 (실제 cut-off는 코드값 보고 결정)",
            treatment_question="높은 학력을 가진 구직자는 낮은 학력자에 비해 6개월 이내 취업 확률이 얼마나 다른가?"
        ];

        // T3: 관련 경력 보유 여부 (experience in desired occupation)
        T3 [
            label="T3: has_experience_hope1",
            role="treatment_meta",
            treatment_var="CARR_MYCT1",
            treatment_name="has_experience_hope1",
            treatment_def="CARR_MYCT1 > 0",
            treatment_question="희망직종1에 대한 관련 경력이 있는 구직자는 무경력자에 비해 6개월 이내 취업 확률이 얼마나 다른가?"
        ];

        // T4: 노동시장 매칭 여건 (모집비율 높은 시장 vs 낮은 시장)
        T4 [
            label="T4: high_vacancy_rate",
            role="treatment_meta",
            treatment_var="IPS_VS_RCIT_RATE",
            treatment_name="high_vacancy_rate",
            treatment_def="직종×지역별 모집비율이 상위 q분위(예: 상위 30%)이면 1, 그 외 0",
            treatment_question="채용공고가 상대적으로 많은 직종·지역에서 구직하는 사람은 6개월 이내 취업 확률이 얼마나 더 높은가?"
        ];

        // T5: 직업훈련-희망직종 일치도 (training-job match quality)
        T5 [
            label="T5: high_training_match",
            role="treatment_meta",
            treatment_var="OCTR_HOPE_JSFC_CSCY_SCOR1",
            treatment_name="high_training_match",
            treatment_def="직업훈련-희망직종 일치도 점수가 상위 q분위(예: 상위 30%)이면 1, 그 외 0",
            treatment_question="직업훈련 내용이 본인이 희망하는 직종과 더 잘 맞는 구직자는 그렇지 않은 사람보다 6개월 이내 취업 확률이 얼마나 높은가?"
        ];
    }
'''

# 시각화
import pygraphviz as pgv

G5 = pgv.AGraph("graph_5.dot")

## treatment 메타데이터 추출
treat_nodes5 = [n for n in G5.nodes() if n.attr.get("role") == "treatment_meta"]
treatments5 = [{
    "node": str(n),
    "var": n.attr.get("treatment_var"),
    "name": n.attr.get("treatment_name"),
    "def": n.attr.get("treatment_def"),
    "question": n.attr.get("treatment_question"),
} for n in treat_nodes5]

# DoWhy/TabPFN용 DAG 노드/엣지 분리
dag_nodes5 = [n for n in G5.nodes() if n.attr.get("role") != "treatment_meta"]
edges5 = [
    (str(u), str(v))
    for u, v in G5.edges()
    if G5.get_node(u).attr.get("role") != "treatment_meta"
    and G5.get_node(v).attr.get("role") != "treatment_meta"
]