"""
유틸리티 모듈 - 그래프 파싱 및 로깅 기능
"""
import re
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


# ============================================================================
# 그래프 파싱 함수
# ============================================================================

def extract_treatments_from_dot(dot_file_path: Path) -> List[Dict[str, str]]:
    """
    .dot 파일에서 treatment 메타데이터를 추출합니다.
    
    Args:
        dot_file_path: .dot 파일 경로
    
    Returns:
        treatment 정보 리스트 (각 treatment는 dict)
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
    
    Args:
        gml_file_path: GML 파일 경로
    
    Returns:
        treatment 정보 리스트
    """
    # GML 형식은 현재 사용하지 않지만, 확장성을 위해 함수 정의
    # 필요시 구현
    return []


def extract_treatments_from_graph(graph_file_path: Path) -> List[Dict[str, str]]:
    """
    그래프 파일에서 treatment 정보를 추출합니다.
    파일 확장자에 따라 적절한 파서를 선택합니다.
    
    Args:
        graph_file_path: 그래프 파일 경로
    
    Returns:
        treatment 정보 리스트
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
    
    Args:
        data_dir: 데이터 디렉토리 경로
        graph_data_dir: 그래프 데이터 디렉토리 이름 (기본값: "graph_data")
    
    Returns:
        그래프 파일 경로 리스트
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
    
    Args:
        data_dir: 데이터 디렉토리 경로
        graph_data_dir: 그래프 데이터 디렉토리 이름
    
    Returns:
        {graph_file_name: [treatment_info, ...]} 딕셔너리
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
    
    Args:
        log_dir: 로그 디렉토리 경로 (None이면 기본값 사용)
        log_filename: 로그 파일명 (None이면 자동 생성)
        level: 로깅 레벨
        no_logs: 로그 저장 비활성화 여부
    
    Returns:
        Logger 객체 또는 None
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

