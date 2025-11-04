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


def create_causal_graph(graph_file):
    """
    GML 형식 그래프 파일을 읽어서 NetworkX 인과 그래프를 생성하는 함수
    
    사용자 제공 GML 형식:
    graph [
        directed 1
        node [id "gps" label "gps"]
        edge [source "gps" target "hippocampus"]
    ]
    
    Args:
        graph_file (str): 그래프 파일 경로 (GML 형식)
    
    Returns:
        nx.DiGraph: 인과 그래프 객체
    """
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
    preprocessor = preprocess.Preprocessor([], api_key=api_key)
    
    # 작업 디렉토리 변경 저장
    original_cwd = os.getcwd()
    data_path = Path(data_dir).resolve()
    
    try:
        # preprocess.py의 load_variable_mapping과 load_job_mapping이 
        # '../data/' 상대 경로를 사용하므로, src/ 폴더가 기준이 됨
        # 따라서 data_dir의 상위 폴더에서 src/를 찾아 작업 디렉토리 설정
        # 일반적으로 laborlab/data -> laborlab/src 기준으로 '../data/' 사용
        script_dir = Path(__file__).parent  # src/ 폴더
        laborlab_dir = script_dir.parent     # laborlab/ 폴더
        
        # laborlab 폴더로 이동하여 preprocess.py의 상대 경로가 작동하도록 함
        os.chdir(str(laborlab_dir))
        
        # file_list의 경로를 절대 경로로 변환
        absolute_file_list = [str(Path(f).resolve()) for f in file_list]
        
        # get_merged_df를 사용하여 모든 파일을 로드, 전처리, 병합
        merged_df = preprocessor.get_merged_df(absolute_file_list)
        
        print(f"✅ 모든 데이터 전처리 및 병합 완료")
        return merged_df
    
    finally:
        # 원래 작업 디렉토리로 복원
        os.chdir(original_cwd)


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
    parser.add_argument('--no-logs', action='store_true', help='로그 저장 비활성화')
    parser.add_argument('--verbose', action='store_true', help='상세 출력 활성화')
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    logger = setup_logging(args)
    
    try:
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
        # graph 인자가 없으면 data_dir/main_graph를 기본값으로 사용
        graph_path = args.graph if args.graph else None
        file_list, causal_graph = load_all_data(
            args.data_dir,
            graph_path
        )
        
        # 2. 데이터 전처리 및 병합 (Preprocessor 사용)
        print("2️⃣ 데이터 전처리 및 병합 중...")
        # 환경변수에서 API 키 가져오기 (선택사항)
        api_key = os.environ.get('LLM_API_KEY', None)
        merged_df = preprocess_and_merge_data(file_list, args.data_dir, api_key=api_key)
        print(f"✅ 최종 병합 데이터: {len(merged_df)}건, {len(merged_df.columns)}개 변수")
        
        if logger:
            logger.info("="*60)
            logger.info("데이터 로드 및 병합 완료")
            logger.info("="*60)
            logger.info(f"최종 데이터 크기: {merged_df.shape}")
            logger.info(f"노드 수: {causal_graph.number_of_nodes()}")
            logger.info(f"엣지 수: {causal_graph.number_of_edges()}")
        
        # 4. 인과모델 생성 및 분석
        print("4️⃣ 인과모델 생성 중...")
        model = CausalModel(
            data=merged_df,
            treatment=args.treatment,
            outcome=args.outcome,
            graph=causal_graph
        )
        
        print("5️⃣ 인과효과 식별 중...")
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        print("6️⃣ 인과효과 추정 중...")
        estimate = estimation.estimate_causal_effect(
            model,
            identified_estimand,
            args.estimator,
            logger
        )
        
        print("7️⃣ 검증 테스트 실행 중...")
        validation_results = estimation.run_validation_tests(
            model,
            identified_estimand,
            estimate,
            logger
        )
        
        print("8️⃣ 민감도 분석 실행 중...")
        sensitivity_df = estimation.run_sensitivity_analysis(
            model,
            identified_estimand,
            estimate,
            logger
        )
        
        print("9️⃣ 시각화 생성 중...")
        heatmap_path = estimation.create_sensitivity_heatmap(
            sensitivity_df,
            logger
        ) if not sensitivity_df.empty else None
        
        print("🔟 최종 요약 보고서 출력 중...")
        estimation.print_summary_report(estimate, validation_results, sensitivity_df)
        
        if logger:
            logger.info("분석 완료")
        
        print(f"\n✅ 전체 분석 완료!")
        
    except Exception as e:
        if logger:
            logger.error(f"분석 중 오류 발생: {e}")
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
