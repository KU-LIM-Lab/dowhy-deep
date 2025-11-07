"""
DoWhy 라이브러리를 이용한 인과모델 구축, 추정, 검증 End-to-End 파이프라인

수정 사항:
- 정형 데이터와 비정형 데이터(JSON) 통합 로드
- JHNT_CTN을 PK로 데이터 병합
- treatment 파라미터를 argparser로 입력받아 다양한 실험 지원
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import logging
from datetime import datetime
import os
import sys
import json
import re
import time

# DoWhy 라이브러리 임포트
import dowhy
from dowhy import CausalModel
import networkx as nx

# 로컬 DoWhy 라이브러리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 모듈 임포트
from . import preprocess
from . import estimation

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# DoWhy 로거 레벨 설정
import logging as dowhy_logging
dowhy_logging.getLogger("dowhy.causal_estimator").setLevel(dowhy_logging.WARNING)
dowhy_logging.getLogger("dowhy.causal_estimators").setLevel(dowhy_logging.WARNING)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# GPT API 키 설정
# ============================================================================
# API 키는 experiment_config.json의 api_key 필드에서 설정합니다.
# run_batch_experiments.py를 통해 실행하면 config의 api_key가 자동으로 전달됩니다.
# 직접 실행하는 경우 --api-key 인자로 전달할 수 있습니다.
# ============================================================================


def create_causal_graph(graph_file):
    """
    DOT 형식 그래프 파일을 읽어서 NetworkX 인과 그래프를 생성하는 함수
    
    Args:
        graph_file (str): 그래프 파일 경로 (DOT 형식)
    
    Returns:
        nx.DiGraph: 인과 그래프 객체
    """
    # 무조건 DOT 형식으로 파싱
    return _parse_dot_graph(graph_file)


def _parse_dot_graph(graph_file):
    """DOT 형식 그래프 파일을 파싱합니다."""
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
            # digraph인지 확인
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


def _parse_dot_manual(graph_file):
    """DOT 형식을 수동으로 파싱합니다 (pydot 없이)."""
    with open(graph_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    G = nx.DiGraph()
    
    # digraph인지 확인
    is_digraph = content.strip().startswith('digraph')
    
    # subgraph cluster_treatments 블록 제거 (treatment 메타데이터는 DAG에 포함하지 않음)
    # subgraph cluster_treatments { ... } 블록 제거
    content_without_subgraph = re.sub(
        r'subgraph\s+cluster_treatments\s*\{[^}]*\}',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 노드 정의 찾기: node_id [label="..."]
    # 노드 ID는 변수명 (예: ACQ_180_YN, cover_score 등)
    # 노드명이 라벨 정의에 나타남
    node_pattern = r'([A-Za-z_][A-Za-z0-9_]*)\s*\[[^\]]*label\s*=\s*"([^"]+)"'
    for match in re.finditer(node_pattern, content_without_subgraph):
        node_id = match.group(1)
        label = match.group(2)
        # T1, T2 등의 treatment 노드는 제외
        if not re.match(r'^T\d+$', node_id):
            G.add_node(node_id, label=label)
    
    # 엣지 찾기: source -> target; 또는 source -> target [label="..."]
    # 주석 처리된 라인은 제외
    edge_pattern = r'([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)'
    for match in re.finditer(edge_pattern, content_without_subgraph):
        source = match.group(1)
        target = match.group(2)
        # treatment 노드(T1, T2 등)는 제외
        if not re.match(r'^T\d+$', source) and not re.match(r'^T\d+$', target):
            G.add_edge(source, target)
    
    # 방향성 그래프로 변환
    if is_digraph and not G.is_directed():
        G = G.to_directed()
    
    return G


def _parse_gml_graph(graph_file):
    """GML 형식 그래프 파일을 파싱합니다."""
    # GML 파일 읽기
    with open(graph_file, 'r', encoding='utf-8') as f:
        gml_content = f.read()
    
    G = nx.DiGraph()
    
    # graph [ ... ] 블록 추출
    graph_match = re.search(r'graph\s*\[(.*?)\]', gml_content, re.DOTALL)
    if not graph_match:
        raise ValueError("GML 형식이 올바르지 않습니다: 'graph [' 블록을 찾을 수 없습니다.")
    
    graph_body = graph_match.group(1)
    
    # directed 플래그 확인
    directed = re.search(r'directed\s+(\d+)', graph_body)
    is_directed = directed and directed.group(1) == "1"
    
    # 모든 node 블록 추출
    node_pattern = r'node\s*\[(.*?)\]'
    for node_match in re.finditer(node_pattern, graph_body, re.DOTALL):
        node_content = node_match.group(1)
        
        # id와 label 추출 (따옴표 처리)
        id_match = re.search(r'id\s+"([^"]+)"', node_content)
        label_match = re.search(r'label\s+"([^"]+)"', node_content)
        
        if id_match:
            node_id = id_match.group(1)
            label = label_match.group(1) if label_match else node_id
            # treatment_meta role이 있는 노드는 제외
            role_match = re.search(r'role\s*=\s*"([^"]+)"', node_content)
            if role_match and role_match.group(1) == "treatment_meta":
                continue
            G.add_node(node_id, label=label)
    
    # 모든 edge 블록 추출
    edge_pattern = r'edge\s*\[(.*?)\]'
    for edge_match in re.finditer(edge_pattern, graph_body, re.DOTALL):
        edge_content = edge_match.group(1)
        
        # source와 target 추출 (따옴표 처리)
        source_match = re.search(r'source\s+"([^"]+)"', edge_content)
        target_match = re.search(r'target\s+"([^"]+)"', edge_content)
        
        if source_match and target_match:
            source = source_match.group(1)
            target = target_match.group(1)
            # treatment_meta 노드는 제외
            if source not in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
                G.add_edge(source, target)
    
    # 방향성 그래프로 변환
    if not G.is_directed() and is_directed:
        G = G.to_directed()
    
    return G


def load_all_data(data_dir, graph_file=None):
    """
    정형 데이터와 비정형 데이터(JSON)를 모두 로드하는 함수
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        graph_file (str, optional): 그래프 파일 경로. None이면 data_dir/main_graph 사용
    
    Returns:
        tuple: (파일경로_리스트, 인과그래프)
    """
    data_path = Path(data_dir)
    
    # 1. 정형 데이터 파일 경로 확인 (fixed_data 폴더에서)
    structured_data_path = data_path / "fixed_data" / "data.csv"
    if not structured_data_path.exists():
        # fallback: data_dir 직접 경로
        structured_data_path = data_path / "data.csv"
    
    if not structured_data_path.exists():
        raise FileNotFoundError(f"정형 데이터 파일을 찾을 수 없습니다: {structured_data_path}")
    
    print(f"✅ 정형 데이터 파일 경로: {structured_data_path}")
    
    # 2. 비정형 데이터(JSON) 파일 경로 리스트 생성 (variant_data 폴더에서)
    variant_data_path = data_path / "variant_data"
    file_list = []
    
    json_files = [
        ("RESUME_JSON.json", "이력서"),
        ("COVERLETTERS_JSON.json", "자기소개서"),
        ("TRAININGS_JSON.json", "직업훈련"),
        ("LICENSES_JSON.json", "자격증")
    ]
    
    # 정형 데이터 파일을 먼저 추가 (Preprocessor의 get_merged_df 방식과 일치)
    file_list.append(str(structured_data_path))
    
    # JSON 파일 경로 추가
    for filename, json_type in json_files:
        json_path = variant_data_path / filename
        if json_path.exists():
            file_list.append(str(json_path))
            print(f"✅ {json_type} 파일 경로 추가: {json_path}")
        else:
            print(f"⚠️ {json_type} 파일을 찾을 수 없습니다: {json_path}")
    
    # 3. 인과 그래프 로드
    # graph_file이 제공되지 않으면 data_dir/main_graph 또는 data_dir/graph_data/graph_1 사용
    if graph_file is None:
        graph_file = data_path / "main_graph"
        if not graph_file.exists():
            # fallback: graph_data 폴더에서 첫 번째 그래프 파일 찾기
            graph_data_path = data_path / "graph_data"
            if graph_data_path.exists():
                # .dot 파일 제외하고 GML 형식 파일 우선
                graph_files = [f for f in graph_data_path.glob("graph_*") if not f.suffix == '.dot']
                if not graph_files:
                    # .dot 파일도 포함
                    graph_files = list(graph_data_path.glob("graph_*"))
                if graph_files:
                    graph_file = sorted(graph_files)[0]  # 첫 번째 파일 사용
                    print(f"⚠️ main_graph를 찾을 수 없어 graph_data 폴더의 {graph_file.name}을 사용합니다.")
    else:
        graph_file = Path(graph_file)
    
    if not graph_file.exists():
        raise FileNotFoundError(f"그래프 파일을 찾을 수 없습니다: {graph_file}")
    
    causal_graph = create_causal_graph(str(graph_file))
    print(f"✅ 인과 그래프 로드 완료: {causal_graph.number_of_nodes()}개 노드, {causal_graph.number_of_edges()}개 엣지")
    
    return file_list, causal_graph


def clean_dataframe_for_causal_model(df, required_vars=None, logger=None):
    """
    CausalModel 생성 전에 데이터프레임을 정리하는 함수
    - Logger 객체나 다른 비데이터 타입 컬럼 제거
    - 숫자/문자열/불린 타입만 유지
    - required_vars에 지정된 변수는 항상 유지
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        required_vars (list, optional): 반드시 유지해야 할 변수 리스트 (treatment, outcome 등)
        logger: 로거 객체
    
    Returns:
        pd.DataFrame: 정리된 데이터프레임
    """
    df_clean = df.copy()
    cols_to_drop = []
    
    if required_vars is None:
        required_vars = []
    
    for col in df_clean.columns:
        # object 타입 컬럼 확인
        if df_clean[col].dtype == 'object':
            if len(df_clean) > 0:
                # NaN이 아닌 첫 번째 값 확인
                non_null_values = df_clean[col].dropna()
                if len(non_null_values) > 0:
                    first_val = non_null_values.iloc[0]
                    # Logger 같은 객체 타입인지 확인
                    is_logger_object = isinstance(first_val, logging.Logger) or 'Logger' in str(type(first_val))
                    is_invalid_type = not isinstance(first_val, (str, int, float, bool, type(None)))
                    
                    if is_logger_object or is_invalid_type:
                        # 필수 변수인 경우 Logger 객체를 NaN으로 대체
                        if col in required_vars:
                            if logger:
                                logger.warning(f"필수 변수 '{col}'의 값이 객체 타입({type(first_val).__name__})이어서 NaN으로 대체합니다.")
                            else:
                                print(f"⚠️ 필수 변수 '{col}'의 값이 객체 타입({type(first_val).__name__})이어서 NaN으로 대체합니다.")
                            df_clean[col] = np.nan
                        else:
                            # 필수 변수가 아닌 경우 컬럼 제거
                            cols_to_drop.append(col)
                            if logger:
                                logger.warning(f"컬럼 '{col}'이 객체 타입({type(first_val).__name__})이어서 제거합니다.")
                            else:
                                print(f"⚠️ 컬럼 '{col}'이 객체 타입({type(first_val).__name__})이어서 제거합니다.")
    
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        if logger:
            logger.info(f"제거된 컬럼: {cols_to_drop}")
        else:
            print(f"제거된 컬럼: {cols_to_drop}")
    
    return df_clean


def preprocess_and_merge_data(file_list, data_dir, api_key=None):
    """
    Preprocessor 클래스를 사용하여 모든 데이터를 전처리하고 병합하는 함수
    
    Args:
        file_list (list): 파일 경로 리스트 [정형데이터, 이력서, 자기소개서, 직업훈련, 자격증]
        data_dir (str): 데이터 디렉토리 경로
        api_key (str, optional): LLM API 키
    
    Returns:
        pd.DataFrame: 병합된 데이터프레임
    """
    # Preprocessor 인스턴스 생성
    # preprocess.py는 __file__ 기준으로 경로를 계산하므로 작업 디렉토리 변경 불필요
    preprocessor = preprocess.Preprocessor([], api_key=api_key)
    
    # file_list의 경로를 절대 경로로 변환
    absolute_file_list = [str(Path(f).resolve()) for f in file_list]
    
    # get_merged_df를 사용하여 모든 파일을 로드, 전처리, 병합
    merged_df = preprocessor.get_merged_df(absolute_file_list)
    
    print(f"✅ 모든 데이터 전처리 및 병합 완료")
    return merged_df


def save_predictions_to_excel(df_with_predictions, output_dir=None, filename=None, logger=None):
    """
    예측값이 포함된 데이터프레임을 Excel 파일로 저장
    
    Args:
        df_with_predictions: 예측값이 포함된 데이터프레임
        output_dir: 출력 디렉토리 (None이면 log 폴더 사용)
        filename: 파일명 (None이면 자동 생성)
        logger: 로거 객체
    
    Returns:
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
    
    # Excel 파일로 저장
    df_with_predictions.to_excel(filepath, index=False, engine='openpyxl')
    
    if logger:
        logger.info(f"예측 결과 저장 완료: {filepath}")
        file_size = os.path.getsize(filepath)
        logger.info(f"파일 크기: {file_size:,} bytes")
    
    return str(filepath)


def setup_logging(args):
    """로깅을 설정하는 통합 함수"""
    if args.no_logs:
        return None
    
    # log 폴더 생성
    script_dir = Path(__file__).parent.parent
    log_dir = script_dir / "log"
    log_dir.mkdir(exist_ok=True)
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 터미널 출력 로깅 설정
    output_dir = os.environ.get('TERMINAL_OUTPUT_DIR', 'log')
    terminal_output_file = os.path.join(output_dir, f'python_output_{timestamp}.log')
    
    # 터미널 출력을 파일로 리다이렉션
    import sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    output_file = open(terminal_output_file, 'w', encoding='utf-8')
    sys.stdout = TeeOutput(original_stdout, output_file)
    sys.stderr = TeeOutput(original_stderr, output_file)
    
    # 로그 파일 설정
    if args.graph:
        graph_name = Path(args.graph).stem
    else:
        graph_name = "main_graph"
    log_filename = f"{graph_name}_{args.treatment}_{timestamp}.log"
    log_filepath = log_dir / log_filename
    
    # 로깅 설정
    logging.basicConfig(
        level=20,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"분석 시작 - {timestamp}")
    graph_display = args.graph if args.graph else f"{args.data_dir}/main_graph"
    logger.info(f"데이터: {args.data_dir}, 그래프: {graph_display}")
    logger.info(f"처치: {args.treatment}, 결과: {args.outcome}, 추정방법: {args.estimator}")
    
    return logger


def parse_arguments():
    """명령행 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description="DoWhy 인과추론 분석")
    
    parser.add_argument('--data-dir', type=str, required=True, help='데이터 디렉토리 경로')
    parser.add_argument('--graph', type=str, default=None, help='그래프 파일 경로 (기본값: data_dir/main_graph)')
    parser.add_argument('--estimator', type=str, choices=['tabpfn', 'linear_regression', 'propensity_score', 'instrumental_variable'],
                       default='linear_regression', help='추정 방법')
    parser.add_argument('--treatment', type=str, required=True, help='처치 변수명')
    parser.add_argument('--outcome', type=str, required=True, help='결과 변수명')
    parser.add_argument('--api-key', type=str, default=None, help='GPT API 키 (experiment_config.json에서 설정)')
    parser.add_argument('--no-logs', action='store_true', help='로그 저장 비활성화')
    parser.add_argument('--verbose', action='store_true', help='상세 출력 활성화')
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    logger = setup_logging(args)
    
    try:
        # 전체 시작 시간
        total_start_time = time.time()
        step_times = {}
        
        print(f"\n🚀 DoWhy 인과추론 분석 시작")
        print(f"📊 데이터 디렉토리: {args.data_dir}")
        graph_display = args.graph if args.graph else f"{args.data_dir}/main_graph"
        print(f"🕸️ 그래프: {graph_display}")
        print(f"🎯 처치: {args.treatment}, 📈 결과: {args.outcome}")
        print(f"🔧 추정방법: {args.estimator}")
        print(f"📦 DoWhy 버전: {dowhy.__version__}")
        print("="*60)
        
        # 1. 데이터 로드
        print("1️⃣ 데이터 로드 중...")
        step_start = time.time()
        # graph 인자가 없으면 data_dir/main_graph를 기본값으로 사용
        graph_path = args.graph if args.graph else None
        file_list, causal_graph = load_all_data(
            args.data_dir,
            graph_path
        )
        step_times['데이터 로드'] = time.time() - step_start
        print(f"⏱️ 데이터 로드 소요 시간: {step_times['데이터 로드']:.2f}초")
        
        # 2. 데이터 전처리 및 병합 (Preprocessor 사용)
        print("2️⃣ 데이터 전처리 및 병합 중...")
        print("⚡ JSON 파일 4개(이력서, 자기소개서, 직업훈련, 자격증) 병렬 처리 시작")
        step_start = time.time()
        # API 키는 config 파일에서 설정 (run_batch_experiments.py를 통해 전달됨)
        api_key = args.api_key
        if api_key:
            print(f"🔑 API 키: config 파일에서 사용")
        else:
            print(f"⚠️ API 키가 설정되지 않았습니다. LLM 기능을 사용할 수 없습니다.")
        
        merged_df = preprocess_and_merge_data(file_list, args.data_dir, api_key=api_key)
        step_times['데이터 전처리 및 병합'] = time.time() - step_start
        print(f"⏱️ 데이터 전처리 및 병합 소요 시간: {step_times['데이터 전처리 및 병합']:.2f}초")
        print(f"✅ 최종 병합 데이터: {len(merged_df)}건, {len(merged_df.columns)}개 변수")
        
        # merged_df의 head() 로깅
        print("\n" + "="*60)
        print("📊 병합된 데이터프레임 미리보기 (head):")
        print("="*60)
        print(merged_df.head())
        print("="*60 + "\n")
        
        if logger:
            logger.info("="*60)
            logger.info("데이터 로드 및 병합 완료")
            logger.info("="*60)
            logger.info(f"최종 데이터 크기: {merged_df.shape}")
            logger.info(f"컬럼 목록: {list(merged_df.columns)}")
            logger.info(f"노드 수: {causal_graph.number_of_nodes()}")
            logger.info(f"엣지 수: {causal_graph.number_of_edges()}")
            logger.info("\n병합된 데이터프레임 head():")
            logger.info("\n" + str(merged_df.head()))
        
        # 3. 데이터 정리 (Logger 객체 등 제거)
        print("3️⃣ 데이터 정리 중...")
        step_start = time.time()
        
        # 그래프에 정의된 모든 변수 추출
        graph_variables = set(causal_graph.nodes())
        print(f"📋 그래프에 정의된 변수 수: {len(graph_variables)}개")
        
        # treatment와 outcome 변수는 반드시 유지해야 함
        required_vars = [args.treatment, args.outcome]
        # 그래프에 정의된 모든 변수도 필수 변수로 추가
        required_vars.extend(list(graph_variables))
        required_vars = list(set(required_vars))  # 중복 제거
        
        # Logger 객체가 데이터프레임에 포함되어 있는지 사전 검사
        logger_columns = []
        for col in merged_df.columns:
            if merged_df[col].dtype == 'object' and len(merged_df) > 0:
                non_null_values = merged_df[col].dropna()
                if len(non_null_values) > 0:
                    first_val = non_null_values.iloc[0]
                    # Logger 객체인지 확인
                    if isinstance(first_val, logging.Logger) or 'Logger' in str(type(first_val)):
                        logger_columns.append((col, type(first_val).__name__))
                        if logger:
                            logger.error(f"⚠️ 경고: 컬럼 '{col}'에 Logger 객체가 포함되어 있습니다! (타입: {type(first_val).__name__})")
                        else:
                            print(f"⚠️ 경고: 컬럼 '{col}'에 Logger 객체가 포함되어 있습니다! (타입: {type(first_val).__name__})")
        
        if logger_columns:
            print(f"\n❌ 오류: 다음 컬럼에 Logger 객체가 발견되었습니다:")
            for col, col_type in logger_columns:
                print(f"   - {col} (타입: {col_type})")
            print(f"\n이 문제를 해결하기 위해 데이터 정리 과정에서 Logger 객체를 제거합니다.")
        
        merged_df_clean = clean_dataframe_for_causal_model(merged_df, required_vars=required_vars, logger=logger)
        
        # 그래프 변수와 데이터 변수 일치 여부 검증
        data_variables = set(merged_df_clean.columns)
        missing_graph_vars = graph_variables - data_variables
        extra_data_vars = data_variables - graph_variables
        
        if missing_graph_vars:
            print(f"\n⚠️ 경고: 그래프에 정의된 변수 중 데이터에 없는 변수:")
            for var in sorted(missing_graph_vars):
                print(f"   - {var}")
            if logger:
                logger.warning(f"그래프에 정의된 변수 중 데이터에 없는 변수: {sorted(missing_graph_vars)}")
        
        # 그래프에 정의되지 않은 변수 제거 (필수 변수 제외)
        essential_vars = {args.treatment, args.outcome, "SEEK_CUST_NO", "JHNT_CTN", "JHNT_MBN"}
        vars_to_keep = set()
        
        # 1. 그래프에 정의된 모든 변수 추가
        vars_to_keep.update(graph_variables)
        
        # 2. 필수 변수 추가 (treatment, outcome, 병합 키)
        vars_to_keep.update(essential_vars)
        
        # 3. 실제 데이터에 존재하는 변수만 필터링
        vars_to_keep = vars_to_keep & data_variables
        
        # 4. 그래프에 정의되지 않은 변수 제거
        vars_to_remove = data_variables - vars_to_keep
        
        if vars_to_remove:
            print(f"\n🗑️ 그래프에 정의되지 않은 변수 제거 중 ({len(vars_to_remove)}개):")
            for var in sorted(list(vars_to_remove)[:20]):  # 처음 20개만 출력
                print(f"   - {var}")
            if len(vars_to_remove) > 20:
                print(f"   ... 외 {len(vars_to_remove) - 20}개")
            if logger:
                logger.info(f"그래프에 정의되지 않은 변수 제거: {sorted(list(vars_to_remove))}")
            
            # 변수 제거
            merged_df_clean = merged_df_clean[list(vars_to_keep)]
            print(f"✅ 변수 제거 완료: {len(merged_df_clean.columns)}개 변수 유지")
        
        step_times['데이터 정리'] = time.time() - step_start
        print(f"⏱️ 데이터 정리 소요 시간: {step_times['데이터 정리']:.2f}초")
        print(f"✅ 정리된 데이터: {len(merged_df_clean)}건, {len(merged_df_clean.columns)}개 변수")
        
        # 최종 검증: 그래프 변수와 데이터 변수 일치 여부
        final_data_variables = set(merged_df_clean.columns)
        final_missing_graph_vars = graph_variables - final_data_variables
        final_extra_data_vars = final_data_variables - graph_variables - essential_vars
        
        if final_missing_graph_vars:
            print(f"\n⚠️ 경고: 그래프에 정의된 변수 중 최종 데이터에 없는 변수:")
            for var in sorted(final_missing_graph_vars):
                print(f"   - {var}")
            if logger:
                logger.warning(f"그래프에 정의된 변수 중 최종 데이터에 없는 변수: {sorted(final_missing_graph_vars)}")
        
        if final_extra_data_vars:
            print(f"\n⚠️ 경고: 최종 데이터에 있지만 그래프에 정의되지 않은 변수 ({len(final_extra_data_vars)}개):")
            for var in sorted(list(final_extra_data_vars)[:10]):
                print(f"   - {var}")
            if len(final_extra_data_vars) > 10:
                print(f"   ... 외 {len(final_extra_data_vars) - 10}개")
            if logger:
                logger.warning(f"최종 데이터에 있지만 그래프에 정의되지 않은 변수: {sorted(list(final_extra_data_vars))}")
        
        # treatment와 outcome 변수가 있는지 확인
        missing_vars = [var for var in [args.treatment, args.outcome] if var not in merged_df_clean.columns]
        if missing_vars:
            raise ValueError(f"필수 변수가 데이터에 없습니다: {missing_vars}")
        
        # 그래프의 핵심 변수들이 모두 있는지 확인
        critical_missing = missing_graph_vars - {args.treatment, args.outcome}  # treatment/outcome은 이미 체크됨
        if critical_missing:
            print(f"\n❌ 오류: 그래프의 핵심 변수들이 데이터에 없습니다:")
            for var in sorted(critical_missing):
                print(f"   - {var}")
            if logger:
                logger.error(f"그래프의 핵심 변수들이 데이터에 없습니다: {sorted(critical_missing)}")
            # 경고만 출력하고 계속 진행 (일부 변수가 없어도 분석 가능할 수 있음)
            # raise ValueError(f"그래프의 핵심 변수들이 데이터에 없습니다: {sorted(critical_missing)}")
        
        # 4. 인과모델 생성 및 분석
        print("4️⃣ 인과모델 생성 중...")
        step_start = time.time()
        model = CausalModel(
            data=merged_df_clean,
            treatment=args.treatment,
            outcome=args.outcome,
            graph=causal_graph
        )
        step_times['인과모델 생성'] = time.time() - step_start
        print(f"⏱️ 인과모델 생성 소요 시간: {step_times['인과모델 생성']:.2f}초")
        
        print("5️⃣ 인과효과 식별 중...")
        step_start = time.time()
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        step_times['인과효과 식별'] = time.time() - step_start
        print(f"⏱️ 인과효과 식별 소요 시간: {step_times['인과효과 식별']:.2f}초")
        
        print("6️⃣ 인과효과 추정 중...")
        step_start = time.time()
        estimate = estimation.estimate_causal_effect(
            model,
            identified_estimand,
            args.estimator,
            logger
        )
        step_times['인과효과 추정'] = time.time() - step_start
        print(f"⏱️ 인과효과 추정 소요 시간: {step_times['인과효과 추정']:.2f}초")
        
        step_start = time.time()
        # 예측 전에 한 번 더 Logger 객체 제거 (안전장치)
        # treatment와 outcome 변수는 필수이므로 유지
        essential_vars_for_pred = {args.treatment, args.outcome}
        merged_df_clean_final = clean_dataframe_for_causal_model(
            merged_df_clean, 
            required_vars=list(essential_vars_for_pred), 
            logger=logger
        )
        accuracy, df_with_predictions = estimation.predict_conditional_expectation(estimate, merged_df_clean_final, logger)
        step_times['예측'] = time.time() - step_start
        print(f"⏱️ 예측 소요 시간: {step_times['예측']:.2f}초")
        print(f"✅ 취업 확률 예측 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        step_start = time.time()
        excel_path = save_predictions_to_excel(df_with_predictions, logger=logger)
        step_times['예측 결과 저장'] = time.time() - step_start
        print(f"⏱️ 예측 결과 저장 소요 시간: {step_times['예측 결과 저장']:.2f}초")
        print(f"✅ 예측 결과 저장 완료: {excel_path}")

        print("7️⃣ 검증 테스트 실행 중...")
        step_start = time.time()
        validation_results = estimation.run_validation_tests(
            model,
            identified_estimand,
            estimate,
            logger
        )
        step_times['검증 테스트'] = time.time() - step_start
        print(f"⏱️ 검증 테스트 소요 시간: {step_times['검증 테스트']:.2f}초")
        
        print("8️⃣ 민감도 분석 실행 중...")
        step_start = time.time()
        sensitivity_df = estimation.run_sensitivity_analysis(
            model,
            identified_estimand,
            estimate,
            logger
        )
        step_times['민감도 분석'] = time.time() - step_start
        print(f"⏱️ 민감도 분석 소요 시간: {step_times['민감도 분석']:.2f}초")
        
        print("9️⃣ 시각화 생성 중...")
        step_start = time.time()
        heatmap_path = estimation.create_sensitivity_heatmap(
            sensitivity_df,
            logger
        ) if not sensitivity_df.empty else None
        step_times['시각화 생성'] = time.time() - step_start
        print(f"⏱️ 시각화 생성 소요 시간: {step_times['시각화 생성']:.2f}초")
        
        print("🔟 최종 요약 보고서 출력 중...")
        step_start = time.time()
        estimation.print_summary_report(estimate, validation_results, sensitivity_df)
        step_times['요약 보고서'] = time.time() - step_start
        print(f"⏱️ 요약 보고서 출력 소요 시간: {step_times['요약 보고서']:.2f}초")
        
        # 전체 소요 시간 계산
        total_time = time.time() - total_start_time
        step_times['전체'] = total_time
        
        # 시간 요약 출력
        print("\n" + "="*60)
        print("⏱️ 단계별 소요 시간 요약")
        print("="*60)
        for step_name, elapsed_time in step_times.items():
            percentage = (elapsed_time / total_time * 100) if step_name != '전체' else 100
            print(f"  {step_name:20s}: {elapsed_time:7.2f}초 ({percentage:5.1f}%)")
        print("="*60)
        
        if logger:
            logger.info("분석 완료")
            logger.info("="*60)
            logger.info("단계별 소요 시간 요약")
            logger.info("="*60)
            for step_name, elapsed_time in step_times.items():
                percentage = (elapsed_time / total_time * 100) if step_name != '전체' else 100
                logger.info(f"  {step_name:20s}: {elapsed_time:7.2f}초 ({percentage:5.1f}%)")
            logger.info("="*60)
        
        print(f"\n✅ 전체 분석 완료! (총 소요 시간: {total_time:.2f}초)")
        
    except Exception as e:
        if logger:
            logger.error(f"분석 중 오류 발생: {e}")
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
