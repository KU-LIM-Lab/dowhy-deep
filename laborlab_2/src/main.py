"""
LaborLab 2 - 인과추론 분석 메인 파이프라인

main.py와 run_batch_experiments.py를 병합한 통합 파이프라인
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
import itertools
from sklearn.model_selection import train_test_split

# DoWhy 라이브러리 임포트
import dowhy
from dowhy import CausalModel
import networkx as nx

# 로컬 DoWhy 라이브러리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 모듈 임포트
from . import preprocess
from . import estimation
from .utils import (
    extract_treatments_from_graph,
    find_all_graph_files,
    setup_logging
)

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
    DOT 형식 그래프 파일을 읽어서 NetworkX 인과 그래프를 생성하는 함수
    
    Args:
        graph_file (str): 그래프 파일 경로 (DOT 형식)
    
    Returns:
        nx.DiGraph: 인과 그래프 객체
    """
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


def load_all_data(data_dir, seis_data_dir, graph_file=None):
    """
    정형 데이터와 비정형 데이터(JSON)를 모두 로드하는 함수
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        seis_data_dir (str): seis_data 디렉토리 이름
        graph_file (str, optional): 그래프 파일 경로
    
    Returns:
        tuple: (파일경로_리스트, 인과그래프)
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


def clean_dataframe_for_causal_model(df, required_vars=None, logger=None):
    """
    CausalModel 생성 전에 데이터프레임을 정리하는 함수
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


def preprocess_and_merge_data(file_list, data_dir, api_key=None):
    """
    Preprocessor 클래스를 사용하여 모든 데이터를 전처리하고 병합하는 함수
    """
    preprocessor = preprocess.Preprocessor([], api_key=api_key)
    absolute_file_list = [str(Path(f).resolve()) for f in file_list]
    merged_df = preprocessor.get_merged_df(absolute_file_list)
    print(f"✅ 모든 데이터 전처리 및 병합 완료")
    return merged_df


def save_predictions_to_excel(df_with_predictions, output_dir=None, filename=None, logger=None):
    """예측값이 포함된 데이터프레임을 Excel 파일로 저장"""
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


def run_analysis_without_preprocessing(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    logger=None,
    experiment_id: str = None
):
    """
    전처리된 데이터를 사용하여 인과추론 분석을 수행하는 함수
    """
    try:
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
        vars_to_keep = (graph_variables | essential_vars) & data_variables
        df_for_analysis = merged_df_clean[list(vars_to_keep)].copy()
        
        missing_vars = [var for var in [treatment, outcome] if var not in df_for_analysis.columns]
        if missing_vars:
            raise ValueError(f"필수 변수가 데이터에 없습니다: {missing_vars}")
        
        step_times['데이터 필터링'] = time.time() - step_start
        
        # 3. Train/Test Split
        print("3️⃣ Train/Test Split 중 (80/20)...")
        step_start = time.time()
        
        outcome_data = df_for_analysis[outcome]
        is_binary = outcome_data.nunique() <= 2 and outcome_data.dtype in ['int64', 'int32', 'bool']
        
        if is_binary:
            df_train, df_test = train_test_split(
                df_for_analysis,
                test_size=0.2,
                random_state=42,
                stratify=outcome_data
            )
        else:
            df_train, df_test = train_test_split(
                df_for_analysis,
                test_size=0.2,
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
        
        # 7. 예측
        print("7️⃣ 예측 중...")
        step_start = time.time()
        essential_vars_for_pred = {treatment, outcome}
        df_test_clean = clean_dataframe_for_causal_model(
            df_test,
            required_vars=list(essential_vars_for_pred),
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
    merged_df_clean,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    experiment_id: str,
    logger=None
) -> dict:
    """단일 실험을 실행합니다"""
    start_time = datetime.now()
    try:
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


def load_config(config_path: Path) -> dict:
    """설정 파일을 로드합니다"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_batch_experiments(config: dict, base_dir: Path):
    """배치 실험을 실행합니다"""
    data_dir = config.get("data_dir", "data")
    seis_data_dir = config.get("seis_data_dir", "seis_data")
    graph_data_dir = config.get("graph_data_dir", "graph_data")
    output_dir = config.get("output_dir", "log")
    graphs = config.get("graphs", [])
    treatments = config.get("treatments", [])
    outcomes = config.get("outcomes", ["ACQ_180_YN"])
    estimators = config.get("estimators", ["tabpfn"])
    auto_extract_treatments = config.get("auto_extract_treatments", False)
    api_key = config.get("api_key", None) or os.environ.get("LLM_API_KEY", None)
    
    # 절대 경로로 변환
    data_dir_path = base_dir / data_dir
    
    # 그래프 파일 경로 처리
    graph_files = []
    
    if auto_extract_treatments:
        print(f"🔍 그래프 파일에서 자동으로 treatment 추출 중...")
        found_graphs = find_all_graph_files(data_dir_path, graph_data_dir)
        graph_files = [str(g) for g in found_graphs]
        
        if not graph_files:
            print(f"⚠️ {graph_data_dir} 폴더에서 그래프 파일을 찾을 수 없습니다.")
        else:
            print(f"✅ {len(graph_files)}개의 그래프 파일 발견")
    else:
        for graph in graphs:
            if isinstance(graph, str):
                graph_path = base_dir / data_dir / graph_data_dir / graph
                if graph_path.exists():
                    graph_files.append(str(graph_path))
                else:
                    graph_path = Path(graph)
                    if graph_path.exists():
                        graph_files.append(str(graph_path))
    
    if not graph_files:
        print("❌ 유효한 그래프 파일이 없습니다.")
        return
    
    # treatment 자동 추출
    graph_treatments_map = {}
    graph_outcomes_map = {}
    
    if auto_extract_treatments:
        print(f"\n🔍 각 그래프 파일에서 treatment 정보 추출 중...")
        
        for graph_file in graph_files:
            graph_path = Path(graph_file)
            extracted_treatments = extract_treatments_from_graph(graph_path)
            
            if extracted_treatments:
                graph_treatments_map[graph_file] = [t["treatment_var"] for t in extracted_treatments if t.get("treatment_var")]
                if extracted_treatments[0].get("outcome"):
                    graph_outcomes_map[graph_file] = extracted_treatments[0]["outcome"]
                print(f"   ✅ {graph_path.name}: {len(graph_treatments_map[graph_file])}개의 treatment 발견")
    
    # 실험 조합 생성
    sorted_estimators = []
    if "linear_regression" in estimators:
        sorted_estimators.append("linear_regression")
    if "tabpfn" in estimators:
        sorted_estimators.append("tabpfn")
    for est in estimators:
        if est not in sorted_estimators:
            sorted_estimators.append(est)
    
    if auto_extract_treatments and graph_treatments_map:
        experiment_combinations = []
        for graph_file in graph_files:
            graph_treatments = graph_treatments_map.get(graph_file, treatments)
            graph_outcome = graph_outcomes_map.get(graph_file, outcomes[0] if outcomes else "ACQ_180_YN")
            
            for treatment in graph_treatments:
                for estimator in sorted_estimators:
                    experiment_combinations.append((graph_file, treatment, graph_outcome, estimator))
    else:
        experiment_combinations = list(itertools.product(
            graph_files,
            treatments,
            outcomes,
            sorted_estimators
        ))
    
    total_experiments = len(experiment_combinations)
    print(f"\n📊 총 {total_experiments}개의 실험을 실행합니다.\n")
    
    # 전처리 수행
    print("="*80)
    print("🔄 데이터 전처리 시작")
    print("="*80)
    
    preprocessing_start = time.time()
    
    file_list, _ = load_all_data(str(data_dir_path), seis_data_dir, graph_file=None)
    
    print("⚡ JSON 파일 4개(이력서, 자기소개서, 직업훈련, 자격증) 병렬 처리 시작")
    merged_df = preprocess_and_merge_data(file_list, str(data_dir_path), api_key=api_key)
    print(f"✅ 최종 병합 데이터: {len(merged_df)}건, {len(merged_df.columns)}개 변수")
    
    # 모든 그래프의 변수 수집
    all_graph_variables = set()
    for graph_file in graph_files:
        graph_path = Path(graph_file)
        try:
            causal_graph = create_causal_graph(str(graph_path))
            all_graph_variables.update(causal_graph.nodes())
        except Exception as e:
            print(f"⚠️ 그래프 파일 로드 실패 ({graph_path.name}): {e}")
    
    all_treatments = set()
    all_outcomes = set()
    for graph_file in graph_files:
        if graph_file in graph_treatments_map:
            all_treatments.update(graph_treatments_map[graph_file])
        if graph_file in graph_outcomes_map:
            all_outcomes.add(graph_outcomes_map[graph_file])
    if not auto_extract_treatments:
        all_treatments.update(treatments)
    if not graph_outcomes_map:
        all_outcomes.update(outcomes)
    
    essential_vars = all_treatments | all_outcomes | {"SEEK_CUST_NO", "JHNT_CTN", "JHNT_MBN"}
    required_vars = list(all_graph_variables | essential_vars)
    
    merged_df_clean = clean_dataframe_for_causal_model(
        merged_df, 
        required_vars=required_vars, 
        logger=None
    )
    
    data_variables = set(merged_df_clean.columns)
    vars_to_keep = (all_graph_variables | essential_vars) & data_variables
    vars_to_remove = data_variables - vars_to_keep
    
    if vars_to_remove:
        print(f"🗑️ 그래프에 정의되지 않은 변수 제거 중 ({len(vars_to_remove)}개)...")
        merged_df_clean = merged_df_clean[list(vars_to_keep)]
    
    preprocessing_elapsed = time.time() - preprocessing_start
    print(f"⏱️ 전처리 완료! 소요 시간: {preprocessing_elapsed:.2f}초")
    print(f"✅ 정리된 데이터: {len(merged_df_clean)}건, {len(merged_df_clean.columns)}개 변수")
    print("="*80 + "\n")
    
    # 결과 저장
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = base_dir / output_dir
    output_dir_path.mkdir(exist_ok=True)
    
    results_file = output_dir_path / f"batch_experiments_{timestamp}.json"
    csv_results_file = output_dir_path / f"experiment_results_{timestamp}.csv"
    
    csv_columns = [
        'graph_name', 'treatment', 'estimator', 'ate_value',
        'placebo_passed', 'placebo_pvalue',
        'unobserved_passed', 'unobserved_pvalue',
        'subset_passed', 'subset_pvalue',
        'dummy_passed', 'dummy_pvalue',
        'f1_score', 'auc', 'duration_seconds'
    ]
    
    pd.DataFrame(columns=csv_columns).to_csv(csv_results_file, index=False, encoding='utf-8-sig')
    
    # 로거 설정
    logger = None
    if not config.get("no_logs", False):
        logger = setup_logging(
            log_dir=output_dir_path,
            log_filename=f"batch_experiments_{timestamp}.log"
        )
        if logger:
            logger.info(f"배치 실험 시작 - {timestamp}")
            logger.info(f"총 실험 수: {total_experiments}")
    
    # 실험 실행
    for idx, (graph_file, treatment, outcome, estimator) in enumerate(experiment_combinations, 1):
        experiment_id = f"exp_{idx:04d}_{Path(graph_file).stem}_{treatment}_{outcome}_{estimator}"
        
        print(f"\n[{idx}/{total_experiments}] 실험 실행 중...")
        
        result = run_single_experiment(
            merged_df_clean=merged_df_clean,
            graph_file=graph_file,
            treatment=treatment,
            outcome=outcome,
            estimator=estimator,
            experiment_id=experiment_id,
            logger=logger
        )
        
        results.append(result)
        
        if result["status"] == "failed":
            print(f"❌ 실패: {experiment_id}")
            if result.get("error"):
                print(f"   에러: {result['error']}")
        
        # CSV에 결과 추가
        if result["status"] == "success":
            csv_row = {
                'graph_name': result.get('graph_name', ''),
                'treatment': result.get('treatment', ''),
                'estimator': result.get('estimator', ''),
                'ate_value': result.get('ate_value'),
                'placebo_passed': result.get('placebo_passed'),
                'placebo_pvalue': result.get('placebo_pvalue'),
                'unobserved_passed': result.get('unobserved_passed'),
                'unobserved_pvalue': result.get('unobserved_pvalue'),
                'subset_passed': result.get('subset_passed'),
                'subset_pvalue': result.get('subset_pvalue'),
                'dummy_passed': result.get('dummy_passed'),
                'dummy_pvalue': result.get('dummy_pvalue'),
                'f1_score': result.get('f1_score'),
                'auc': result.get('auc'),
                'duration_seconds': result.get('duration_seconds')
            }
            
            try:
                existing_df = pd.read_csv(csv_results_file, encoding='utf-8-sig')
            except (FileNotFoundError, pd.errors.EmptyDataError):
                existing_df = pd.DataFrame(columns=csv_columns)
            
            new_row_df = pd.DataFrame([csv_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            updated_df.to_csv(csv_results_file, index=False, encoding='utf-8-sig')
        
        # 중간 결과 저장 (JSON)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        print(f"\n✅ 성공: {success_count}, ❌ 실패: {failed_count}")
    
    # 최종 요약
    print(f"\n{'='*80}")
    print("📋 배치 실험 완료")
    print(f"{'='*80}")
    print(f"총 실험 수: {total_experiments}")
    print(f"성공: {success_count}")
    print(f"실패: {failed_count}")
    print(f"JSON 결과 파일: {results_file}")
    print(f"CSV 결과 파일: {csv_results_file}")
    print(f"{'='*80}\n")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="LaborLab 2 인과추론 분석 파이프라인")
    
    default_config = os.environ.get(
        "EXPERIMENT_CONFIG",
        "config.json"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="설정 JSON 파일 경로"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    
    if os.path.isabs(args.config):
        config_path = Path(args.config)
    else:
        config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")
        print(f"   현재 디렉토리: {os.getcwd()}")
        print(f"   스크립트 디렉토리: {script_dir}")
        return
    
    print(f"📄 설정 파일 로드: {config_path}")
    config = load_config(config_path)
    
    # 배치 실험 실행
    run_batch_experiments(config, script_dir)


if __name__ == "__main__":
    main()

