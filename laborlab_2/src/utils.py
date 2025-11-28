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


def extract_treatments_from_graph(graph_file_path: Path) -> List[Dict[str, str]]:
    """
    DOT 그래프 파일에서 treatment 정보를 추출합니다.
    
    Input:
        graph_file_path (Path): 그래프 파일 경로 (.dot 파일)
    
    Output:
        List[Dict[str, str]]: treatment 정보 리스트
    """
    graph_path = Path(graph_file_path)
    
    if graph_path.suffix == '.dot':
        return extract_treatments_from_dot(graph_path)
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
    
    # .dot 파일만 찾기
    graph_files = list(graph_data_path.glob("graph_*.dot"))
    
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
# 분석 실행 함수는 estimation.py로 이동됨
# ============================================================================


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

