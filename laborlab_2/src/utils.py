"""
유틸리티 모듈 - 그래프 파싱, 데이터 로드, 전처리, 로깅 기능
"""
import re
import logging
import os
import json
import time
import itertools
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split

# DoWhy 라이브러리 임포트
from dowhy import CausalModel


# ============================================================================
# 그래프 파싱 함수
# ============================================================================

def extract_treatments_from_dot(dot_file_path: Path) -> List[Dict[str, str]]:
    """
    .dot 파일에서 treatment 메타데이터를 추출합니다.
    
    Input:
        dot_file_path (Path): .dot 파일 경로
    
    Output:
        List[Dict[str, str]]: treatment 정보 리스트 (각 treatment는 dict 형태)
            - 각 dict는 다음 키를 포함: node, treatment_var, treatment_name, 
              treatment_def, treatment_question, label, outcome
    """
    with open(dot_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    treatments = []
    
    # subgraph cluster_treatments 블록 찾기
    subgraph_pattern = r'subgraph\s+cluster_treatments\s*\{[^}]*\}'
    subgraph_match = re.search(subgraph_pattern, content, re.DOTALL)
    
    if not subgraph_match:
        return treatments
    
    subgraph_content = subgraph_match.group(0)
    
    # T1, T2, ... 형태의 treatment 노드 찾기
    treatment_pattern = r'(T\d+)\s*\[([^\]]+)\]'
    treatment_matches = re.finditer(treatment_pattern, subgraph_content, re.DOTALL)
    
    for match in treatment_matches:
        node_name = match.group(1)  # T1, T2, etc.
        node_attrs = match.group(2)
        
        # 속성 추출
        treatment_var = re.search(r'treatment_var\s*=\s*"([^"]+)"', node_attrs)
        treatment_name = re.search(r'treatment_name\s*=\s*"([^"]+)"', node_attrs)
        treatment_def = re.search(r'treatment_def\s*=\s*"([^"]+)"', node_attrs)
        treatment_question = re.search(r'treatment_question\s*=\s*"([^"]+)"', node_attrs)
        label = re.search(r'label\s*=\s*"([^"]+)"', node_attrs)
        
        treatment_info = {
            "node": node_name,
            "treatment_var": treatment_var.group(1) if treatment_var else "",
            "treatment_name": treatment_name.group(1) if treatment_name else "",
            "treatment_def": treatment_def.group(1) if treatment_def else "",
            "treatment_question": treatment_question.group(1) if treatment_question else "",
            "label": label.group(1) if label else node_name,
        }
        
        # outcome 정보도 추출 (subgraph의 label에서)
        outcome_match = re.search(r'label\s*=\s*"Treatments\s*\(outcome:\s*([^)]+)\)"', subgraph_content)
        if outcome_match:
            treatment_info["outcome"] = outcome_match.group(1).strip()
        
        treatments.append(treatment_info)
    
    return treatments


def extract_treatments_from_gml(gml_file_path: Path) -> List[Dict[str, str]]:
    """
    GML 형식 파일에서 treatment 메타데이터를 추출합니다.
    
    Input:
        gml_file_path (Path): GML 파일 경로
    
    Output:
        List[Dict[str, str]]: treatment 정보 리스트 (현재는 빈 리스트 반환)
    """
    # GML 형식은 현재 사용하지 않지만, 확장성을 위해 함수 정의
    # 필요시 구현
    return []


def extract_treatments_from_graph(graph_file_path: Path) -> List[Dict[str, str]]:
    """
    그래프 파일에서 treatment 정보를 추출합니다.
    파일 확장자에 따라 적절한 파서를 선택합니다.
    
    Input:
        graph_file_path (Path): 그래프 파일 경로
    
    Output:
        List[Dict[str, str]]: treatment 정보 리스트
    """
    graph_path = Path(graph_file_path)
    
    if graph_path.suffix == '.dot':
        return extract_treatments_from_dot(graph_path)
    elif graph_path.suffix == '' or 'graph' in graph_path.name:
        # GML 형식 파일 (확장자 없음)
        return extract_treatments_from_gml(graph_path)
    else:
        return []


def find_all_graph_files(data_dir: Path, graph_data_dir: Optional[str] = None) -> List[Path]:
    """
    데이터 디렉토리에서 모든 그래프 파일을 찾습니다.
    
    Input:
        data_dir (Path): 데이터 디렉토리 경로
        graph_data_dir (Optional[str]): 그래프 데이터 디렉토리 이름 (기본값: "graph_data")
    
    Output:
        List[Path]: 그래프 파일 경로 리스트 (정렬된 순서)
    """
    if graph_data_dir is None:
        graph_data_dir = "graph_data"
    
    graph_data_path = Path(data_dir) / graph_data_dir
    
    if not graph_data_path.exists():
        return []
    
    # .dot 파일과 확장자 없는 graph 파일 찾기
    graph_files = []
    
    # .dot 파일
    graph_files.extend(graph_data_path.glob("graph_*.dot"))
    
    # 확장자 없는 graph 파일 (GML 형식)
    for graph_file in graph_data_path.glob("graph_*"):
        if not graph_file.suffix and graph_file.is_file():
            graph_files.append(graph_file)
    
    # 정렬하여 반환
    return sorted(graph_files)


def get_treatments_from_all_graphs(data_dir: Path, graph_data_dir: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
    """
    모든 그래프 파일에서 treatment 정보를 추출합니다.
    
    Input:
        data_dir (Path): 데이터 디렉토리 경로
        graph_data_dir (Optional[str]): 그래프 데이터 디렉토리 이름
    
    Output:
        Dict[str, List[Dict[str, str]]]: {graph_file_name: [treatment_info, ...]} 딕셔너리
    """
    graph_files = find_all_graph_files(data_dir, graph_data_dir)
    
    result = {}
    
    for graph_file in graph_files:
        treatments = extract_treatments_from_graph(graph_file)
        if treatments:
            result[str(graph_file)] = treatments
    
    return result


# ============================================================================
# 로깅 함수
# ============================================================================

def setup_logging(
    log_dir: Optional[Path] = None,
    log_filename: Optional[str] = None,
    level: int = logging.INFO,
    no_logs: bool = False
) -> Optional[logging.Logger]:
    """
    로깅을 설정하는 함수
    
    Input:
        log_dir (Optional[Path]): 로그 디렉토리 경로 (None이면 기본값 사용)
        log_filename (Optional[str]): 로그 파일명 (None이면 자동 생성)
        level (int): 로깅 레벨 (기본값: logging.INFO)
        no_logs (bool): 로그 저장 비활성화 여부
    
    Output:
        Optional[logging.Logger]: Logger 객체 또는 None (no_logs=True인 경우)
    """
    if no_logs:
        return None
    
    # 로그 디렉토리 설정
    if log_dir is None:
        script_dir = Path(__file__).parent.parent
        log_dir = script_dir / "log"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일명 설정
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"experiment_{timestamp}.log"
    
    log_filepath = log_dir / log_filename
    
    # 로깅 설정
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"로깅 시작 - 로그 파일: {log_filepath}")
    
    return logger


# ============================================================================
# 그래프 생성 함수
# ============================================================================

def create_causal_graph(graph_file: str) -> nx.DiGraph:
    """
    DOT 형식 그래프 파일을 읽어서 NetworkX 인과 그래프를 생성하는 함수
    
    Input:
        graph_file (str): 그래프 파일 경로 (DOT 형식)
    
    Output:
        nx.DiGraph: 인과 그래프 객체 (NetworkX 방향성 그래프)
    """
    return _parse_dot_graph(graph_file)


def _parse_dot_graph(graph_file: str) -> nx.DiGraph:
    """
    DOT 형식 그래프 파일을 파싱합니다.
    
    Input:
        graph_file (str): 그래프 파일 경로
    
    Output:
        nx.DiGraph: 파싱된 NetworkX 방향성 그래프
    """
    try:
        # pydot을 사용하여 DOT 파일 읽기
        import pydot
        graphs = pydot.graph_from_dot_file(graph_file)
        if not graphs:
            raise ValueError(f"DOT 파일에서 그래프를 찾을 수 없습니다: {graph_file}")
        
        # 첫 번째 그래프 사용
        dot_graph = graphs[0]
        
        # NetworkX 그래프로 변환
        G = nx.drawing.nx_pydot.from_pydot(dot_graph)
        
        # 방향성 그래프로 변환 (digraph인 경우)
        if not G.is_directed():
            with open(graph_file, 'r', encoding='utf-8') as f:
                content = f.read()
            if content.strip().startswith('digraph'):
                G = G.to_directed()
        
        return G
    except ImportError:
        # pydot이 없으면 수동 파싱
        return _parse_dot_manual(graph_file)
    except Exception as e:
        # pydot 파싱 실패 시 수동 파싱 시도
        try:
            return _parse_dot_manual(graph_file)
        except Exception as e2:
            raise ValueError(f"DOT 파일 파싱 실패: {e}. 수동 파싱도 실패: {e2}")


def _parse_dot_manual(graph_file: str) -> nx.DiGraph:
    """
    DOT 형식을 수동으로 파싱합니다 (pydot 없이).
    
    Input:
        graph_file (str): 그래프 파일 경로
    
    Output:
        nx.DiGraph: 파싱된 NetworkX 방향성 그래프
    """
    with open(graph_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    G = nx.DiGraph()
    
    # digraph인지 확인
    is_digraph = content.strip().startswith('digraph')
    
    # subgraph cluster_treatments 블록 제거
    content_without_subgraph = re.sub(
        r'subgraph\s+cluster_treatments\s*\{[^}]*\}',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 노드 정의 찾기
    node_pattern = r'([A-Za-z_][A-Za-z0-9_]*)\s*\[[^\]]*label\s*=\s*"([^"]+)"'
    for match in re.finditer(node_pattern, content_without_subgraph):
        node_id = match.group(1)
        label = match.group(2)
        if not re.match(r'^T\d+$', node_id):
            G.add_node(node_id, label=label)
    
    # 엣지 찾기
    edge_pattern = r'([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)'
    for match in re.finditer(edge_pattern, content_without_subgraph):
        source = match.group(1)
        target = match.group(2)
        if not re.match(r'^T\d+$', source) and not re.match(r'^T\d+$', target):
            G.add_edge(source, target)
    
    # 방향성 그래프로 변환
    if is_digraph and not G.is_directed():
        G = G.to_directed()
    
    return G


# ============================================================================
# 데이터 로드 함수
# ============================================================================

def load_all_data(data_dir: str, seis_data_dir: str, graph_file: Optional[str] = None) -> Tuple[List[str], nx.DiGraph]:
    """
    정형 데이터와 비정형 데이터(JSON)를 모두 로드하는 함수
    
    Input:
        data_dir (str): 데이터 디렉토리 경로
        seis_data_dir (str): seis_data 디렉토리 이름
        graph_file (Optional[str]): 그래프 파일 경로 (None이면 자동으로 찾음)
    
    Output:
        Tuple[List[str], nx.DiGraph]: (파일경로_리스트, 인과그래프)
            - 파일경로_리스트: [정형데이터경로, 이력서경로, 자기소개서경로, 직업훈련경로, 자격증경로]
            - 인과그래프: NetworkX 방향성 그래프 객체
    """
    data_path = Path(data_dir)
    
    # 1. 정형 데이터 파일 경로 확인 (seis_data 폴더에서)
    structured_data_path = data_path / seis_data_dir / "seis_data.csv"
    
    if not structured_data_path.exists():
        raise FileNotFoundError(f"정형 데이터 파일을 찾을 수 없습니다: {structured_data_path}")
    
    print(f"✅ 정형 데이터 파일 경로: {structured_data_path}")
    
    # 2. 비정형 데이터(JSON) 파일 경로 리스트 생성
    seis_data_path = data_path / seis_data_dir
    file_list = []
    
    json_files = [
        ("resume.json", "이력서"),
        ("coverletters.json", "자기소개서"),
        ("trainings.json", "직업훈련"),
        ("licenses.json", "자격증")
    ]
    
    # 정형 데이터 파일을 먼저 추가
    file_list.append(str(structured_data_path))
    
    # JSON 파일 경로 추가
    for filename, json_type in json_files:
        json_path = seis_data_path / filename
        if json_path.exists():
            file_list.append(str(json_path))
            print(f"✅ {json_type} 파일 경로 추가: {json_path}")
        else:
            print(f"⚠️ {json_type} 파일을 찾을 수 없습니다: {json_path}")
    
    # 3. 인과 그래프 로드 (graph_file이 제공되지 않으면 첫 번째 그래프 사용)
    if graph_file is None:
        graph_data_path = data_path / "graph_data"
        if graph_data_path.exists():
            graph_files = list(graph_data_path.glob("graph_*.dot"))
            if graph_files:
                graph_file = sorted(graph_files)[0]
                print(f"⚠️ 그래프 파일이 지정되지 않아 {graph_file.name}을 사용합니다.")
            else:
                raise FileNotFoundError(f"그래프 파일을 찾을 수 없습니다: {graph_data_path}")
        else:
            raise FileNotFoundError(f"그래프 데이터 디렉토리를 찾을 수 없습니다: {graph_data_path}")
    else:
        graph_file = Path(graph_file)
    
    if not graph_file.exists():
        raise FileNotFoundError(f"그래프 파일을 찾을 수 없습니다: {graph_file}")
    
    causal_graph = create_causal_graph(str(graph_file))
    print(f"✅ 인과 그래프 로드 완료: {causal_graph.number_of_nodes()}개 노드, {causal_graph.number_of_edges()}개 엣지")
    
    return file_list, causal_graph


# ============================================================================
# 데이터 전처리 함수
# ============================================================================

def clean_dataframe_for_causal_model(df: pd.DataFrame, required_vars: Optional[List[str]] = None, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    CausalModel 생성 전에 데이터프레임을 정리하는 함수
    Logger 객체나 다른 비데이터 타입 컬럼 제거
    
    Input:
        df (pd.DataFrame): 원본 데이터프레임
        required_vars (Optional[List[str]]): 반드시 유지해야 할 변수 리스트 (treatment, outcome 등)
        logger (Optional[logging.Logger]): 로거 객체
    
    Output:
        pd.DataFrame: 정리된 데이터프레임 (Logger 객체 등이 제거됨)
    """
    df_clean = df.copy()
    cols_to_drop = []
    
    if required_vars is None:
        required_vars = []
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            if len(df_clean) > 0:
                non_null_values = df_clean[col].dropna()
                if len(non_null_values) > 0:
                    first_val = non_null_values.iloc[0]
                    is_logger_object = isinstance(first_val, logging.Logger) or 'Logger' in str(type(first_val))
                    is_invalid_type = not isinstance(first_val, (str, int, float, bool, type(None)))
                    
                    if is_logger_object or is_invalid_type:
                        if col in required_vars:
                            if logger:
                                logger.warning(f"필수 변수 '{col}'의 값이 객체 타입({type(first_val).__name__})이어서 NaN으로 대체합니다.")
                            else:
                                print(f"⚠️ 필수 변수 '{col}'의 값이 객체 타입({type(first_val).__name__})이어서 NaN으로 대체합니다.")
                            df_clean[col] = np.nan
                        else:
                            cols_to_drop.append(col)
    
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    return df_clean


def preprocess_and_merge_data(file_list: List[str], data_dir: str, limit_data: bool = False, limit_size: int = 5000, job_category_file: str = "KSIC") -> pd.DataFrame:
    """
    Preprocessor 클래스를 사용하여 모든 데이터를 전처리하고 병합하는 함수
    
    Input:
        file_list (List[str]): 파일 경로 리스트 [정형데이터, 이력서, 자기소개서, 직업훈련, 자격증]
        data_dir (str): 데이터 디렉토리 경로
        limit_data (bool): 테스트 모드로 데이터 제한 여부
        limit_size (int): 제한할 데이터 크기
        job_category_file (str): 직종 소분류 파일명 (KECO, KSCO, KSIC 중 선택, 기본값: KSIC)
    
    Output:
        pd.DataFrame: 병합된 데이터프레임
    """
    from . import preprocess
    preprocessor = preprocess.Preprocessor([], job_category_file=job_category_file)
    absolute_file_list = [str(Path(f).resolve()) for f in file_list]
    merged_df = preprocessor.get_merged_df(absolute_file_list, limit_data=limit_data, limit_size=limit_size)
    print(f"✅ 모든 데이터 전처리 및 병합 완료")
    return merged_df


# ============================================================================
# 결과 저장 함수
# ============================================================================

def save_predictions_to_excel(df_with_predictions: pd.DataFrame, output_dir: Optional[Path] = None, filename: Optional[str] = None, logger: Optional[logging.Logger] = None) -> str:
    """
    예측값이 포함된 데이터프레임을 Excel 파일로 저장
    
    Input:
        df_with_predictions (pd.DataFrame): 예측값이 포함된 데이터프레임
        output_dir (Optional[Path]): 출력 디렉토리 (None이면 log 폴더 사용)
        filename (Optional[str]): 파일명 (None이면 자동 생성)
        logger (Optional[logging.Logger]): 로거 객체
    
    Output:
        str: 저장된 파일 경로
    """
    if output_dir is None:
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "log"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.xlsx"
    
    filepath = output_dir / filename
    
    df_with_predictions.to_excel(filepath, index=False, engine='openpyxl')
    
    if logger:
        logger.info(f"예측 결과 저장 완료: {filepath}")
    
    return str(filepath)


# ============================================================================
# 분석 실행 함수
# ============================================================================

def run_analysis_without_preprocessing(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    logger: Optional[logging.Logger] = None,
    experiment_id: Optional[str] = None,
    job_category: Optional[str] = None
) -> Dict[str, Any]:
    """
    전처리된 데이터를 사용하여 인과추론 분석을 수행하는 함수
    (estimation → refutation → prediction만 수행)
    
    Input:
        merged_df_clean (pd.DataFrame): 전처리 및 정리된 데이터프레임
        graph_file (str): 그래프 파일 경로
        treatment (str): 처치 변수명
        outcome (str): 결과 변수명
        estimator (str): 추정 방법
        logger (Optional[logging.Logger]): 로거 객체
        experiment_id (Optional[str]): 실험 ID (선택적)
        job_category (Optional[str]): 직종소분류명 (checkpoint 저장 경로에 사용)
    
    Output:
        Dict[str, Any]: 분석 결과 딕셔너리
            - status: "success" 또는 "failed"
            - estimate: 추정된 인과효과 객체
            - validation_results: 검증 결과
            - sensitivity_df: 민감도 분석 결과
            - metrics: 예측 메트릭
            - excel_path: 예측 결과 Excel 파일 경로
            - step_times: 단계별 소요 시간
            - train_size: 학습 데이터 크기
            - test_size: 테스트 데이터 크기
    """
    try:
        from . import estimation
        step_times = {}
        step_start = time.time()
        
        if experiment_id:
            print(f"\n{'='*80}")
            print(f"실험 ID: {experiment_id}")
            print(f"그래프: {Path(graph_file).name}")
            print(f"Treatment: {treatment}, Outcome: {outcome}")
            print(f"Estimator: {estimator}")
            print(f"{'='*80}\n")
        
        # 1. 그래프 로드
        print("1️⃣ 인과 그래프 로드 중...")
        step_start = time.time()
        causal_graph = create_causal_graph(graph_file)
        step_times['그래프 로드'] = time.time() - step_start
        
        # 2. 데이터 필터링
        print("2️⃣ 그래프 변수에 맞게 데이터 필터링 중...")
        step_start = time.time()
        
        graph_variables = set(causal_graph.nodes())
        data_variables = set(merged_df_clean.columns)
        essential_vars = {treatment, outcome, "SEEK_CUST_NO", "JHNT_CTN", "JHNT_MBN"}
        # HOPE_JSCD3_NAME은 그래프에 포함되지 않지만 데이터에는 유지해야 함
        stratification_vars = {"HOPE_JSCD3_NAME"}
        vars_to_keep = (graph_variables | essential_vars | stratification_vars) & data_variables
        df_for_analysis = merged_df_clean[list(vars_to_keep)].copy()
        
        missing_vars = [var for var in [treatment, outcome] if var not in df_for_analysis.columns]
        if missing_vars:
            raise ValueError(f"필수 변수가 데이터에 없습니다: {missing_vars}")
        
        step_times['데이터 필터링'] = time.time() - step_start
        
        # 3. Train/Test Split
        print("3️⃣ Train/Test Split 중 (1:99)...")
        step_start = time.time()
        
        outcome_data = df_for_analysis[outcome]
        is_binary = outcome_data.nunique() <= 2 and outcome_data.dtype in ['int64', 'int32', 'bool']
        
        if is_binary:
            df_train, df_test = train_test_split(
                df_for_analysis,
                test_size=0.99,  # 1:99 split
                random_state=42,
                stratify=outcome_data
            )
        else:
            df_train, df_test = train_test_split(
                df_for_analysis,
                test_size=0.99,  # 1:99 split
                random_state=42
            )
        
        step_times['Train/Test Split'] = time.time() - step_start
        
        # 4. 인과모델 생성
        print("4️⃣ 인과모델 생성 중...")
        step_start = time.time()
        model = CausalModel(
            data=df_train,
            treatment=treatment,
            outcome=outcome,
            graph=causal_graph
        )
        step_times['인과모델 생성'] = time.time() - step_start
        
        # 5. 인과효과 식별
        print("5️⃣ 인과효과 식별 중...")
        step_start = time.time()
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        step_times['인과효과 식별'] = time.time() - step_start
        
        # 6. 인과효과 추정
        print("6️⃣ 인과효과 추정 중...")
        step_start = time.time()
        estimate = estimation.estimate_causal_effect(
            model,
            identified_estimand,
            estimator,
            logger
        )
        step_times['인과효과 추정'] = time.time() - step_start
        
        # 6-1. Checkpoint 저장 (learning 모드일 때만)
        checkpoint_path = None
        if experiment_id:
            try:
                # checkpoint 디렉토리 경로 생성 (data/checkpoint)
                script_dir = Path(__file__).parent.parent
                checkpoint_dir = script_dir / "data" / "checkpoint"
                
                # 직종소분류별 폴더 생성
                if job_category:
                    job_category_safe = str(job_category).replace("/", "_").replace("\\", "_").replace(" ", "_")
                    checkpoint_dir = checkpoint_dir / job_category_safe
                
                graph_name = Path(graph_file).stem if graph_file else None
                checkpoint_path = estimation.save_checkpoint(
                    estimate,
                    checkpoint_dir,
                    experiment_id,
                    graph_name=graph_name,
                    logger=logger
                )
            except Exception as e:
                if logger:
                    logger.warning(f"Checkpoint 저장 실패 (계속 진행): {e}")
                print(f"⚠️ Checkpoint 저장 실패 (계속 진행): {e}")
        
        # 7. 예측
        print("7️⃣ 예측 중...")
        step_start = time.time()
        essential_vars_for_pred = {treatment, outcome}
        # 예측 전에 실제값 저장 (예측 후 outcome이 덮어씌워지므로)
        if outcome in df_test.columns:
            df_test = df_test.copy()
            df_test[f"{outcome}_actual"] = df_test[outcome].copy()
        
        df_test_clean = clean_dataframe_for_causal_model(
            df_test,
            required_vars=list(essential_vars_for_pred) + [f"{outcome}_actual"] if f"{outcome}_actual" in df_test.columns else list(essential_vars_for_pred),
            logger=logger
        )
        metrics, df_with_predictions = estimation.predict_conditional_expectation(
            estimate, df_test_clean, logger=logger
        )
        step_times['예측'] = time.time() - step_start
        
        # 예측 결과 저장
        if experiment_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{experiment_id}_{timestamp}.xlsx"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.xlsx"
        
        step_start = time.time()
        excel_path = save_predictions_to_excel(df_with_predictions, filename=filename, logger=logger)
        step_times['예측 결과 저장'] = time.time() - step_start
        
        # 8. 검증 테스트
        print("8️⃣ 검증 테스트 실행 중...")
        step_start = time.time()
        validation_results = estimation.run_validation_tests(
            model,
            identified_estimand,
            estimate,
            logger
        )
        step_times['검증 테스트'] = time.time() - step_start
        
        # 9. 민감도 분석
        print("9️⃣ 민감도 분석 실행 중...")
        step_start = time.time()
        sensitivity_df = estimation.run_sensitivity_analysis(
            model,
            identified_estimand,
            estimate,
            logger
        )
        step_times['민감도 분석'] = time.time() - step_start
        
        # 10. 시각화
        print("🔟 시각화 생성 중...")
        step_start = time.time()
        heatmap_path = estimation.create_sensitivity_heatmap(
            sensitivity_df,
            logger
        ) if not sensitivity_df.empty else None
        step_times['시각화 생성'] = time.time() - step_start
        
        # 11. 요약 보고서
        print("1️⃣1️⃣ 최종 요약 보고서 출력 중...")
        step_start = time.time()
        estimation.print_summary_report(estimate, validation_results, sensitivity_df)
        step_times['요약 보고서'] = time.time() - step_start
        
        total_time = sum(step_times.values())
        step_times['전체'] = total_time
        
        print(f"\n✅ 분석 완료! (총 소요 시간: {total_time:.2f}초)")
        
        return {
            "status": "success",
            "estimate": estimate,
            "validation_results": validation_results,
            "sensitivity_df": sensitivity_df,
            "metrics": metrics,
            "excel_path": excel_path,
            "checkpoint_path": checkpoint_path,
            "step_times": step_times,
            "train_size": len(df_train),
            "test_size": len(df_test)
        }
        
    except Exception as e:
        if logger:
            logger.error(f"분석 중 오류 발생: {e}")
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_single_experiment(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    experiment_id: str,
    logger: Optional[logging.Logger] = None,
    split_by_job_category: bool = True
) -> Dict[str, Any]:
    """
    단일 실험을 실행합니다
    
    Input:
        merged_df_clean (pd.DataFrame): 전처리 및 정리된 데이터프레임
        graph_file (str): 그래프 파일 경로
        treatment (str): 처치 변수명
        outcome (str): 결과 변수명
        estimator (str): 추정 방법
        experiment_id (str): 실험 ID
        logger (Optional[logging.Logger]): 로거 객체
    
    Output:
        Dict[str, Any]: 실험 결과 딕셔너리
            - experiment_id: 실험 ID
            - status: "success" 또는 "failed"
            - duration_seconds: 소요 시간 (초)
            - graph: 그래프 파일 경로
            - graph_name: 그래프 파일명
            - treatment: 처치 변수명
            - outcome: 결과 변수명
            - estimator: 추정 방법
            - ate_value: 추정된 ATE 값
            - metrics: 예측 메트릭
            - refutation 결과들 (placebo_passed, unobserved_passed 등)
            - excel_path: 예측 결과 Excel 파일 경로
            - train_size: 학습 데이터 크기
            - test_size: 테스트 데이터 크기
            - start_time: 시작 시간
            - end_time: 종료 시간
            - error: 오류 메시지 (실패한 경우)
    """
    from . import estimation
    start_time = datetime.now()
    try:
        # 직종소분류별로 분리하여 실험 실행
        if split_by_job_category and "HOPE_JSCD3_NAME" in merged_df_clean.columns:
            # 직종소분류별로 데이터 분리
            job_categories = merged_df_clean["HOPE_JSCD3_NAME"].dropna().unique()
            print(f"📊 직종소분류별 실험 실행: {len(job_categories)}개 직종소분류")
            
            all_results = []
            all_predictions = []
            all_metrics = []
            
            for job_category in job_categories:
                job_df = merged_df_clean[merged_df_clean["HOPE_JSCD3_NAME"] == job_category].copy()
                
                if len(job_df) < 10:  # 최소 데이터 수 체크
                    if logger:
                        logger.warning(f"직종소분류 '{job_category}' 데이터가 너무 적어 건너뜁니다: {len(job_df)}건")
                    print(f"⚠️ 직종소분류 '{job_category}' 데이터가 너무 적어 건너뜁니다: {len(job_df)}건")
                    continue
                
                # 직종소분류별 experiment_id 생성
                job_category_safe = str(job_category).replace("/", "_").replace("\\", "_").replace(" ", "_")
                job_experiment_id = f"{experiment_id}_{job_category_safe}"
                
                print(f"\n  🔹 직종소분류: {job_category} ({len(job_df)}건)")
                
                try:
                    job_result = run_analysis_without_preprocessing(
                        merged_df_clean=job_df,
                        graph_file=graph_file,
                        treatment=treatment,
                        outcome=outcome,
                        estimator=estimator,
                        logger=logger,
                        experiment_id=job_experiment_id,
                        job_category=job_category
                    )
                    
                    all_results.append(job_result)
                    
                    # 예측 결과 수집
                    if job_result.get("excel_path"):
                        try:
                            pred_df = pd.read_excel(job_result["excel_path"])
                            all_predictions.append(pred_df)
                        except:
                            pass
                    
                    # 메트릭 수집
                    if job_result.get("metrics"):
                        all_metrics.append(job_result["metrics"])
                        
                except Exception as e:
                    if logger:
                        logger.error(f"직종소분류 '{job_category}' 실험 실패: {e}")
                    print(f"  ❌ 직종소분류 '{job_category}' 실험 실패: {e}")
                    continue
            
            # 모든 직종소분류 결과 통합
            if not all_results:
                raise ValueError("모든 직종소분류 실험이 실패했습니다.")
            
            # 예측 결과 합치기
            if all_predictions:
                combined_predictions = pd.concat(all_predictions, ignore_index=True)
                
                # 통합 메트릭 계산
                combined_metrics = {}
                if all_metrics:
                    # Accuracy, F1, AUC는 전체 예측 결과로 계산
                    actual_outcome_col = f"{outcome}_actual"
                    if actual_outcome_col in combined_predictions.columns and outcome in combined_predictions.columns:
                        actual_y = combined_predictions[actual_outcome_col]
                        predicted_y = combined_predictions[outcome]  # 예측값
                        
                        if pd.api.types.is_numeric_dtype(actual_y):
                            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                            unique_values = set(actual_y.dropna().unique())
                            is_binary = len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values if not pd.isna(v))
                            
                            if is_binary:
                                predicted_classes = (predicted_y > 0.5).astype(int) if pd.api.types.is_numeric_dtype(predicted_y) else predicted_y
                                combined_metrics['accuracy'] = accuracy_score(actual_y, predicted_classes)
                                combined_metrics['f1_score'] = f1_score(actual_y, predicted_classes, zero_division=0)
                                try:
                                    combined_metrics['auc'] = roc_auc_score(actual_y, predicted_y)
                                except:
                                    combined_metrics['auc'] = None
                
                # 예측 결과 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_path = save_predictions_to_excel(
                    combined_predictions, 
                    filename=f"predictions_{experiment_id}_combined_{timestamp}.xlsx",
                    logger=logger
                )
            else:
                combined_metrics = {}
                excel_path = None
            
            # 첫 번째 결과를 기본으로 사용 (ATE는 평균 계산 가능)
            base_result = all_results[0]
            ate_values = [r.get("estimate", {}).get("value") if hasattr(r.get("estimate"), "value") else None 
                         for r in all_results if r.get("estimate")]
            avg_ate = sum([v for v in ate_values if v is not None]) / len([v for v in ate_values if v is not None]) if ate_values else None
            
            result = {
                "status": "success",
                "estimate": base_result.get("estimate"),
                "validation_results": base_result.get("validation_results", {}),
                "sensitivity_df": base_result.get("sensitivity_df"),
                "metrics": combined_metrics,
                "excel_path": excel_path,
                "step_times": base_result.get("step_times", {}),
                "train_size": sum([r.get("train_size", 0) for r in all_results]),
                "test_size": sum([r.get("test_size", 0) for r in all_results]),
                "job_category_results": all_results,
                "num_job_categories": len(all_results)
            }
        else:
            # 직종소분류별 분리 없이 기존 방식
            result = run_analysis_without_preprocessing(
                merged_df_clean=merged_df_clean,
                graph_file=graph_file,
                treatment=treatment,
                outcome=outcome,
                estimator=estimator,
                logger=logger,
                experiment_id=experiment_id
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        metrics = result.get("metrics", {})
        estimate = result.get("estimate")
        validation_results = result.get("validation_results", {})
        
        # ATE 값 추출
        ate_value = None
        if estimate and hasattr(estimate, 'value'):
            ate_value = estimate.value
        
        # Refutation 결과 추출
        refutation_data = {}
        refutation_types = ['placebo', 'unobserved', 'subset', 'dummy']
        for ref_type in refutation_types:
            ref_result = validation_results.get(ref_type)
            if ref_result is not None:
                if ref_type == 'placebo':
                    effect_change = abs(ref_result.new_effect - ref_result.estimated_effect)
                    refutation_data[f'{ref_type}_passed'] = effect_change < 0.01
                elif ref_type == 'unobserved':
                    change_rate = abs(ref_result.new_effect - ref_result.estimated_effect) / abs(ref_result.estimated_effect) if abs(ref_result.estimated_effect) > 0 else float('inf')
                    refutation_data[f'{ref_type}_passed'] = change_rate < 0.2
                elif ref_type == 'subset':
                    effect_change = abs(ref_result.new_effect - ref_result.estimated_effect)
                    change_rate = abs(ref_result.estimated_effect) > 0 and abs(effect_change / ref_result.estimated_effect) or float('inf')
                    refutation_data[f'{ref_type}_passed'] = change_rate < 0.1
                elif ref_type == 'dummy':
                    refutation_data[f'{ref_type}_passed'] = abs(ref_result.new_effect) < 0.01
                
                p_value = estimation.calculate_refutation_pvalue(ref_result, ref_type)
                refutation_data[f'{ref_type}_pvalue'] = p_value
            else:
                refutation_data[f'{ref_type}_passed'] = None
                refutation_data[f'{ref_type}_pvalue'] = None
        
        return {
            "experiment_id": experiment_id,
            "status": "success",
            "duration_seconds": duration,
            "graph": graph_file,
            "graph_name": Path(graph_file).stem,
            "treatment": treatment,
            "outcome": outcome,
            "estimator": estimator,
            "ate_value": ate_value,
            "metrics": metrics,
            "accuracy": metrics.get("accuracy") if metrics else None,
            "f1_score": metrics.get("f1_score") if metrics else None,
            "auc": metrics.get("auc") if metrics else None,
            "excel_path": result.get("excel_path"),
            "train_size": result.get("train_size"),
            "test_size": result.get("test_size"),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            **refutation_data
        }
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "experiment_id": experiment_id,
            "status": "failed",
            "duration_seconds": duration,
            "graph": graph_file,
            "treatment": treatment,
            "outcome": outcome,
            "estimator": estimator,
            "error": str(e),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }


def run_inference(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    checkpoint_dir: Path,
    treatment: str,
    outcome: str,
    estimator: str,
    logger: Optional[logging.Logger] = None,
    experiment_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Inference 모드: checkpoint에서 모델을 로드하여 예측만 수행하는 함수
    직종소분류별로 분리하여 checkpoint를 찾고 예측한 후 합칩니다.
    
    Input:
        merged_df_clean (pd.DataFrame): 전처리 및 정리된 데이터프레임
        graph_file (str): 그래프 파일 경로
        checkpoint_dir (Path): checkpoint 디렉토리 경로
        treatment (str): 처치 변수명
        outcome (str): 결과 변수명
        estimator (str): 추정 방법
        logger (Optional[logging.Logger]): 로거 객체
        experiment_id (Optional[str]): 실험 ID (선택적)
    
    Output:
        Dict[str, Any]: 예측 결과 딕셔너리
            - status: "success" 또는 "failed"
            - metrics: 예측 메트릭 (통합)
            - excel_path: 예측 결과 Excel 파일 경로 (통합)
            - step_times: 단계별 소요 시간
    """
    try:
        from . import estimation
        step_times = {}
        step_start = time.time()
        
        if experiment_id:
            print(f"\n{'='*80}")
            print(f"Inference 모드 - 실험 ID: {experiment_id}")
            print(f"그래프: {Path(graph_file).name}")
            print(f"Treatment: {treatment}, Outcome: {outcome}, Estimator: {estimator}")
            print(f"{'='*80}\n")
        
        graph_name = Path(graph_file).stem
        
        # 직종소분류별로 분리하여 예측
        if "HOPE_JSCD3_NAME" in merged_df_clean.columns:
            job_categories = merged_df_clean["HOPE_JSCD3_NAME"].dropna().unique()
            print(f"📊 직종소분류별 Inference 실행: {len(job_categories)}개 직종소분류")
            
            all_predictions = []
            all_metrics = []
            
            for job_category in job_categories:
                job_df = merged_df_clean[merged_df_clean["HOPE_JSCD3_NAME"] == job_category].copy()
                
                if len(job_df) == 0:
                    continue
                
                job_category_safe = str(job_category).replace("/", "_").replace("\\", "_").replace(" ", "_")
                job_checkpoint_dir = checkpoint_dir / job_category_safe
                
                print(f"\n  🔹 직종소분류: {job_category} ({len(job_df)}건)")
                
                # 해당 직종소분류의 checkpoint 찾기
                checkpoint_file = estimation.find_checkpoint(
                    job_checkpoint_dir,
                    graph_name,
                    treatment,
                    outcome,
                    estimator,
                    logger
                )
                
                if not checkpoint_file:
                    print(f"  ⚠️ Checkpoint를 찾을 수 없어 건너뜁니다: {job_category}")
                    continue
                
                try:
                    # Checkpoint에서 모델 로드
                    estimate = estimation.load_checkpoint(checkpoint_file, logger)
                    
                    # 데이터 필터링
                    essential_vars = {treatment, outcome, "SEEK_CUST_NO", "JHNT_CTN", "JHNT_MBN"}
                    data_variables = set(job_df.columns)
                    vars_to_keep = essential_vars & data_variables
                    
                    missing_vars = [var for var in [treatment, outcome] if var not in job_df.columns]
                    if missing_vars:
                        print(f"  ⚠️ 필수 변수가 없어 건너뜁니다: {missing_vars}")
                        continue
                    
                    df_for_prediction = job_df[list(vars_to_keep)].copy()
                    
                    # 예측 전에 실제값 저장 (예측 후 outcome이 덮어씌워지므로)
                    if outcome in df_for_prediction.columns:
                        df_for_prediction[f"{outcome}_actual"] = df_for_prediction[outcome].copy()
                    
                    # 예측
                    df_pred_clean = clean_dataframe_for_causal_model(
                        df_for_prediction,
                        required_vars=list(essential_vars) + [f"{outcome}_actual"] if f"{outcome}_actual" in df_for_prediction.columns else list(essential_vars),
                        logger=logger
                    )
                    metrics, df_with_predictions = estimation.predict_conditional_expectation(
                        estimate, df_pred_clean, logger=logger
                    )
                    
                    all_predictions.append(df_with_predictions)
                    if metrics:
                        all_metrics.append(metrics)
                    
                    print(f"  ✅ 예측 완료: {len(df_with_predictions)}건")
                    
                except Exception as e:
                    print(f"  ❌ 직종소분류 '{job_category}' 예측 실패: {e}")
                    if logger:
                        logger.error(f"직종소분류 '{job_category}' 예측 실패: {e}")
                    continue
            
            # 모든 직종소분류 예측 결과 합치기
            if not all_predictions:
                raise ValueError("모든 직종소분류 예측이 실패했습니다.")
            
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # 통합 메트릭 계산
            combined_metrics = {}
            actual_outcome_col = f"{outcome}_actual"
            if actual_outcome_col in combined_predictions.columns and outcome in combined_predictions.columns:
                actual_y = combined_predictions[actual_outcome_col]
                predicted_y = combined_predictions[outcome]  # 예측값
                
                if pd.api.types.is_numeric_dtype(actual_y):
                    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                    unique_values = set(actual_y.dropna().unique())
                    is_binary = len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values if not pd.isna(v))
                    
                    if is_binary:
                        predicted_classes = (predicted_y > 0.5).astype(int) if pd.api.types.is_numeric_dtype(predicted_y) else predicted_y
                        valid_mask = ~(pd.isna(actual_y) | pd.isna(predicted_classes))
                        if valid_mask.sum() > 0:
                            combined_metrics['accuracy'] = accuracy_score(actual_y[valid_mask], predicted_classes[valid_mask])
                            combined_metrics['f1_score'] = f1_score(actual_y[valid_mask], predicted_classes[valid_mask], zero_division=0)
                            try:
                                combined_metrics['auc'] = roc_auc_score(actual_y[valid_mask], predicted_y[valid_mask])
                            except:
                                combined_metrics['auc'] = None
            
            # 예측 결과 저장
            step_start = time.time()
            if experiment_id:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"predictions_inference_{experiment_id}_combined_{timestamp}.xlsx"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"predictions_inference_combined_{timestamp}.xlsx"
            
            excel_path = save_predictions_to_excel(combined_predictions, filename=filename, logger=logger)
            step_times['예측 결과 저장'] = time.time() - step_start
            
        else:
            # HOPE_JSCD3_NAME이 없으면 기존 방식 (단일 checkpoint)
            raise ValueError("HOPE_JSCD3_NAME 변수가 데이터에 없습니다. 직종소분류별 분리가 불가능합니다.")
        
        total_time = sum(step_times.values())
        step_times['전체'] = total_time
        
        print(f"\n✅ Inference 완료! (총 소요 시간: {total_time:.2f}초)")
        if combined_metrics:
            print(f"   Accuracy: {combined_metrics.get('accuracy', 'N/A')}")
            print(f"   F1 Score: {combined_metrics.get('f1_score', 'N/A')}")
            print(f"   AUC: {combined_metrics.get('auc', 'N/A')}")
        
        return {
            "status": "success",
            "metrics": combined_metrics,
            "excel_path": excel_path,
            "step_times": step_times,
            "data_size": len(combined_predictions)
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Inference 중 오류 발생: {e}")
        print(f"❌ Inference 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# 설정 파일 로드 함수
# ============================================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    설정 파일을 로드합니다
    
    Input:
        config_path (Path): 설정 파일 경로 (JSON 형식)
    
    Output:
        Dict[str, Any]: 설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

