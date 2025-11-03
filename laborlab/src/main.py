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
        tuple: (정형데이터_df, 비정형데이터_딕셔너리, 인과그래프)
    """
    data_path = Path(data_dir)
    
    # 1. 정형 데이터 로드
    structured_data = pd.read_csv(data_path / "data.csv", encoding='utf-8')
    print(f"✅ 정형 데이터 로드 완료: {len(structured_data)}건")
    
    # 2. 비정형 데이터(JSON) 로드
    unstructured_data = {}
    
    json_files = [
        ("COVERLETTERS_JSON.json", "자기소개서"),
        ("RESUME_JSON.json", "이력서"),
        ("TRAININGS_JSON.json", "직업훈련"),
        ("LICENSES_JSON.json", "자격증")
    ]
    
    for filename, json_type in json_files:
        json_path = data_path / filename
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                unstructured_data[json_type] = json.load(f)
            print(f"✅ {json_type} 데이터 로드 완료: {len(unstructured_data[json_type])}건")
        else:
            print(f"⚠️ {json_type} 파일을 찾을 수 없습니다: {json_path}")
    
    # 3. 인과 그래프 로드
    # graph_file이 제공되지 않으면 data_dir/main_graph 사용
    if graph_file is None:
        graph_file = data_path / "main_graph"
    else:
        graph_file = Path(graph_file)
    
    if not graph_file.exists():
        raise FileNotFoundError(f"그래프 파일을 찾을 수 없습니다: {graph_file}")
    
    causal_graph = create_causal_graph(str(graph_file))
    print(f"✅ 인과 그래프 로드 완료: {causal_graph.number_of_nodes()}개 노드, {causal_graph.number_of_edges()}개 엣지")
    
    return structured_data, unstructured_data, causal_graph


def preprocess_unstructured_data(unstructured_data, data_dir):
    """
    비정형 데이터(JSON)를 정형 데이터로 변환하는 함수
    
    Args:
        unstructured_data (dict): 비정형 데이터 딕셔너리
        data_dir (str): 데이터 디렉토리 경로
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임 리스트
    """
    preprocessor = preprocess.Preprocessor([])
    
    processed_dfs = {}
    
    # 각 JSON 타입별로 전처리 수행
    for json_type, data in unstructured_data.items():
        try:
            # JSON 데이터를 DataFrame으로 변환
            if json_type == "자기소개서":
                df = _convert_coverletters_to_df(data)
            elif json_type == "이력서":
                df = _convert_resume_to_df(data)
            elif json_type == "직업훈련":
                df = _convert_trainings_to_df(data)
            elif json_type == "자격증":
                df = _convert_licenses_to_df(data)
            else:
                df = pd.DataFrame()
            
            if not df.empty:
                processed_dfs[json_type] = df
                print(f"✅ {json_type} 전처리 완료: {len(df)}건")
            
        except Exception as e:
            print(f"⚠️ {json_type} 전처리 실패: {e}")
            processed_dfs[json_type] = pd.DataFrame()
    
    return processed_dfs


def _convert_coverletters_to_df(data):
    """자기소개서 JSON을 DataFrame으로 변환"""
    rows = []
    for record in data:
        seek_cust_no = record.get("SEEK_CUST_NO")
        for coverletter in record.get("COVERLETTERS", []):
            row = {"SEEK_CUST_NO": seek_cust_no}
            row["SFID_NO"] = coverletter.get("SFID_NO")
            row["SFID_IEM_NUM"] = len(coverletter.get("ITEMS", []))
            row["SFID_LTTR_NUM"] = sum(
                len(item.get("SELF_INTRO_CONT", "")) 
                for item in coverletter.get("ITEMS", [])
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _convert_resume_to_df(data):
    """이력서 JSON을 DataFrame으로 변환"""
    rows = []
    for record in data:
        seek_cust_no = record.get("SEEK_CUST_NO")
        # 간단한 집계 정보만 추출
        row = {"SEEK_CUST_NO": seek_cust_no}
        rows.append(row)
    return pd.DataFrame(rows)


def _convert_trainings_to_df(data):
    """직업훈련 JSON을 DataFrame으로 변환"""
    rows = []
    for record in data:
        seek_cust_no = record.get("SEEK_CUST_NO")
        jhnt_ctn = record.get("JHNT_CTN")
        for training in record.get("TRAININGS", []):
            row = {
                "SEEK_CUST_NO": seek_cust_no,
                "JHNT_CTN": jhnt_ctn,
                "KECO_CD": training.get("KECO_CD"),
                "TRNG_JSCD": training.get("TRNG_JSCD")
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    # JHNT_CTN별로 집계
    if not df.empty:
        df = df.groupby("JHNT_CTN").agg({
            "KECO_CD": lambda x: ",".join([str(v) for v in x if pd.notna(v)]) if len(x) > 0 else "",
            "TRNG_JSCD": lambda x: ",".join([str(v) for v in x if pd.notna(v)]) if len(x) > 0 else ""
        }).reset_index()
        return df
    else:
        return pd.DataFrame()


def _convert_licenses_to_df(data):
    """자격증 JSON을 DataFrame으로 변환"""
    rows = []
    for record in data:
        seek_cust_no = record.get("SEEK_CUST_NO")
        jhnt_ctn = record.get("JHNT_CTN")
        licenses = record.get("LICENSES", [])
        row = {
            "SEEK_CUST_NO": seek_cust_no,
            "JHNT_CTN": jhnt_ctn,
            "CRQF_CT": len(licenses)
        }
        rows.append(row)
    return pd.DataFrame(rows)


def merge_all_data(structured_data, processed_dfs):
    """
    정형 데이터와 비정형 데이터를 병합하는 함수
    
    Args:
        structured_data (pd.DataFrame): 정형 데이터
        processed_dfs (dict): 전처리된 비정형 데이터 딕셔너리
    
    Returns:
        pd.DataFrame: 병합된 데이터프레임
    """
    # 정형 데이터가 기준이 됨
    merged_df = structured_data.copy()
    
    # JHNT_CTN 또는 SEEK_CUST_NO를 기준으로 병합
    for json_type, df in processed_dfs.items():
        if df.empty:
            continue
        
        # 병합 키 결정
        if "JHNT_CTN" in df.columns:
            merge_key = "JHNT_CTN"
        elif "SEEK_CUST_NO" in df.columns:
            # SEEK_CUST_NO가 있는 경우, 정형 데이터의 JHNT_MBN과 매핑 필요
            # 여기서는 간단히 skip하고 나중에 구현
            continue
        else:
            continue
        
        if merge_key in merged_df.columns:
            merged_df = merged_df.merge(
                df,
                on=merge_key,
                how="left",
                suffixes=('', f'_{json_type}')
            )
            print(f"✅ {json_type} 데이터 병합 완료")
    
    return merged_df


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
        structured_data, unstructured_data, causal_graph = load_all_data(
            args.data_dir,
            graph_path
        )
        
        # 2. 비정형 데이터 전처리
        print("2️⃣ 비정형 데이터 전처리 중...")
        processed_dfs = preprocess_unstructured_data(unstructured_data, args.data_dir)
        
        # 3. 데이터 병합
        print("3️⃣ 데이터 병합 중...")
        merged_df = merge_all_data(structured_data, processed_dfs)
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
