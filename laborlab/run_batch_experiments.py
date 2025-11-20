"""
배치 실험 실행 스크립트

여러 treatment와 graph 조합으로 인과추론 분석을 실행합니다.
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import itertools
from typing import List, Dict, Any, Optional
import time
import logging
import pandas as pd

# graph_parser 모듈 임포트 (src/__init__.py를 거치지 않고 직접 임포트)
# __init__.py가 preprocess를 임포트하면서 의존성 문제가 발생할 수 있으므로
# 직접 경로에서 모듈을 임포트합니다.
import importlib.util

def load_graph_parser():
    """graph_parser 모듈을 직접 로드합니다."""
    graph_parser_path = Path(__file__).parent / "src" / "graph_parser.py"
    spec = importlib.util.spec_from_file_location("graph_parser", graph_parser_path)
    graph_parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_parser)
    return graph_parser

graph_parser = load_graph_parser()
find_all_graph_files = graph_parser.find_all_graph_files
extract_treatments_from_graph = graph_parser.extract_treatments_from_graph
get_treatments_from_all_graphs = graph_parser.get_treatments_from_all_graphs

# main 모듈 임포트 (전처리 및 분석 함수 사용)
sys.path.insert(0, str(Path(__file__).parent))
from src import main as main_module
from src import estimation as estimation_module


def load_experiment_config(config_file: str) -> Dict[str, Any]:
    """실험 설정 파일을 로드합니다."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_single_experiment(
    merged_df_clean,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    experiment_id: str,
    logger=None
) -> Dict[str, Any]:
    """단일 실험을 실행합니다 (전처리된 데이터 사용)."""
    start_time = datetime.now()
    try:
        result = main_module.run_analysis_without_preprocessing(
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
                # 성공 여부 판단
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
                
                # p-value 추출 (estimation 모듈의 함수 사용)
                p_value = estimation_module.calculate_refutation_pvalue(ref_result, ref_type)
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


def run_batch_experiments(config: Dict[str, Any], base_dir: Path):
    """배치 실험을 실행합니다."""
    data_dir = config.get("data_dir", "data")
    graphs = config.get("graphs", [])
    treatments = config.get("treatments", [])
    outcomes = config.get("outcomes", ["ACQ_180_YN"])
    estimators = config.get("estimators", ["tabpfn"])
    auto_extract_treatments = config.get("auto_extract_treatments", False)
    graph_data_dir = config.get("graph_data_dir", "graph_data")
    api_key = config.get("api_key", None)  # config에서 API 키 가져오기
    
    # 절대 경로로 변환
    data_dir_path = base_dir / data_dir
    if not data_dir_path.is_absolute():
        data_dir_path = base_dir / data_dir
    
    # 그래프 파일 경로 처리
    graph_files = []
    
    # auto_extract_treatments가 True이면 graph_data 폴더에서 자동으로 찾기
    if auto_extract_treatments:
        print(f"🔍 그래프 파일에서 자동으로 treatment 추출 중...")
        found_graphs = find_all_graph_files(data_dir_path, graph_data_dir)
        graph_files = [str(g) for g in found_graphs]
        
        if not graph_files:
            print(f"⚠️ {graph_data_dir} 폴더에서 그래프 파일을 찾을 수 없습니다.")
        else:
            print(f"✅ {len(graph_files)}개의 그래프 파일 발견:")
            for g in graph_files:
                print(f"   - {Path(g).name}")
    else:
        # 수동으로 지정된 그래프 파일들
        for graph in graphs:
            if isinstance(graph, str):
                graph_path = base_dir / data_dir / graph
                if graph_path.exists():
                    graph_files.append(str(graph_path))
                else:
                    # graph_data 폴더에서 찾기
                    graph_path = base_dir / data_dir / graph_data_dir / graph
                    if graph_path.exists():
                        graph_files.append(str(graph_path))
                    else:
                        # 절대 경로로 시도
                        graph_path = Path(graph)
                        if graph_path.exists():
                            graph_files.append(str(graph_path))
                        else:
                            print(f"⚠️ 그래프 파일을 찾을 수 없습니다: {graph}")
            else:
                print(f"⚠️ 잘못된 그래프 경로: {graph}")
    
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
                # outcome 추출 (첫 번째 treatment에서)
                if extracted_treatments[0].get("outcome"):
                    graph_outcomes_map[graph_file] = extracted_treatments[0]["outcome"]
                print(f"   ✅ {graph_path.name}: {len(graph_treatments_map[graph_file])}개의 treatment 발견")
                for t in extracted_treatments:
                    if t.get("treatment_var"):
                        print(f"      - {t['treatment_var']}: {t.get('label', '')}")
            else:
                print(f"   ⚠️ {graph_path.name}: treatment 정보를 찾을 수 없습니다.")
        
        # treatment가 자동 추출된 경우, 각 그래프별로 다른 treatment 사용
        if graph_treatments_map:
            print(f"\n📋 자동 추출된 treatment 정보를 사용합니다.")
    
    # 실험 조합 생성
    # linear_regression을 먼저 실행하고, 그 다음 tabpfn을 실행하도록 순서 변경
    # 빠른 결과 확인을 위해 빠른 추정기(linear_regression)를 먼저 실행
    
    # estimators 리스트를 명시적으로 정렬: linear_regression 먼저, 그 다음 tabpfn, 나머지
    sorted_estimators = []
    if "linear_regression" in estimators:
        sorted_estimators.append("linear_regression")
    if "tabpfn" in estimators:
        sorted_estimators.append("tabpfn")
    # 나머지 estimator 추가 (원래 순서 유지)
    for est in estimators:
        if est not in sorted_estimators:
            sorted_estimators.append(est)
    
    if auto_extract_treatments and graph_treatments_map:
        # 각 그래프별로 해당 그래프의 treatment만 사용
        experiment_combinations = []
        for graph_file in graph_files:
            graph_treatments = graph_treatments_map.get(graph_file, treatments)
            graph_outcome = graph_outcomes_map.get(graph_file, outcomes[0] if outcomes else "ACQ_180_YN")
            
            # 해당 그래프의 treatment와 outcome 조합 생성
            # sorted_estimators 순서대로 실행 (linear_regression 먼저, 그 다음 tabpfn)
            for treatment in graph_treatments:
                for estimator in sorted_estimators:
                    experiment_combinations.append((graph_file, treatment, graph_outcome, estimator))
    else:
        # 기존 방식: 모든 조합 생성하되, estimator 순서를 linear_regression 먼저로 변경
        experiment_combinations = list(itertools.product(
            graph_files,
            treatments,
            outcomes,
            sorted_estimators
        ))
    
    total_experiments = len(experiment_combinations)
    print(f"\n📊 총 {total_experiments}개의 실험을 실행합니다.")
    if auto_extract_treatments and graph_treatments_map:
        print(f"   - 그래프: {len(graph_files)}개 (각 그래프별 treatment 자동 추출)")
        total_treatments = sum(len(t) for t in graph_treatments_map.values())
        print(f"   - 총 Treatment: {total_treatments}개")
        print(f"   - Outcome: {len(set(graph_outcomes_map.values())) if graph_outcomes_map else len(outcomes)}개")
    else:
        print(f"   - 그래프: {len(graph_files)}개")
        print(f"   - Treatment: {len(treatments)}개")
        print(f"   - Outcome: {len(outcomes)}개")
    print(f"   - Estimator: {len(estimators)}개\n")
    
    # ============================================================
    # 전처리를 한 번만 수행
    # ============================================================
    print("="*80)
    print("🔄 데이터 전처리 시작 (한 번만 수행)")
    print("="*80)
    
    preprocessing_start = time.time()
    
    # 1. 데이터 파일 경로 수집
    print("1️⃣ 데이터 파일 경로 수집 중...")
    file_list, _ = main_module.load_all_data(str(data_dir_path), graph_file=None)
    
    # 2. 전처리 및 병합
    print("2️⃣ 데이터 전처리 및 병합 중...")
    print("⚡ JSON 파일 4개(이력서, 자기소개서, 직업훈련, 자격증) 병렬 처리 시작")
    if api_key:
        print(f"🔑 API 키: config 파일에서 사용")
    else:
        print(f"⚠️ API 키가 설정되지 않았습니다. LLM 기능을 사용할 수 없습니다.")
    
    merged_df = main_module.preprocess_and_merge_data(file_list, str(data_dir_path), api_key=api_key)
    print(f"✅ 최종 병합 데이터: {len(merged_df)}건, {len(merged_df.columns)}개 변수")
    
    # 3. 모든 그래프의 변수를 수집하여 데이터 정리
    print("3️⃣ 모든 그래프의 변수 수집 및 데이터 정리 중...")
    
    # 모든 그래프 파일에서 변수 수집
    all_graph_variables = set()
    for graph_file in graph_files:
        graph_path = Path(graph_file)
        try:
            causal_graph = main_module.create_causal_graph(str(graph_path))
            all_graph_variables.update(causal_graph.nodes())
        except Exception as e:
            print(f"⚠️ 그래프 파일 로드 실패 ({graph_path.name}): {e}")
    
    print(f"📋 모든 그래프에서 수집된 변수 수: {len(all_graph_variables)}개")
    
    # 필수 변수 (모든 treatment, outcome, 병합 키)
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
    
    # 데이터 정리
    merged_df_clean = main_module.clean_dataframe_for_causal_model(
        merged_df, 
        required_vars=required_vars, 
        logger=None
    )
    
    # 그래프에 정의되지 않은 변수 제거
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
    results_file = base_dir / "log" / f"batch_experiments_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    # CSV 결과 파일 경로
    csv_results_file = base_dir / "log" / f"experiment_results_{timestamp}.csv"
    
    # CSV 컬럼 정의
    csv_columns = [
        'graph_name', 'treatment', 'estimator', 'ate_value',
        'placebo_passed', 'placebo_pvalue',
        'unobserved_passed', 'unobserved_pvalue',
        'subset_passed', 'subset_pvalue',
        'dummy_passed', 'dummy_pvalue',
        'f1_score', 'auc', 'duration_seconds'
    ]
    
    # CSV 파일 초기화 (헤더만 작성)
    pd.DataFrame(columns=csv_columns).to_csv(csv_results_file, index=False, encoding='utf-8-sig')
    
    # 로거 설정 (선택적)
    logger = None
    if not config.get("no_logs", False):
        log_dir = base_dir / "log"
        log_dir.mkdir(exist_ok=True)
        log_filename = f"batch_experiments_{timestamp}.log"
        log_filepath = log_dir / log_filename
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
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
        
        # 실패한 경우 에러 메시지 출력
        if result["status"] == "failed":
            print(f"❌ 실패: {experiment_id}")
            if result.get("stderr"):
                # stderr의 마지막 몇 줄만 출력 (너무 길지 않게)
                stderr_lines = result["stderr"].strip().split('\n')
                error_preview = '\n'.join(stderr_lines[-10:])  # 마지막 10줄만
                print(f"   에러: {error_preview}")
            elif result.get("error"):
                print(f"   에러: {result['error']}")
        
        # CSV에 결과 추가 (성공한 경우만)
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
            
            # 기존 CSV 읽기
            try:
                existing_df = pd.read_csv(csv_results_file, encoding='utf-8-sig')
            except (FileNotFoundError, pd.errors.EmptyDataError):
                existing_df = pd.DataFrame(columns=csv_columns)
            
            # 새 행 추가
            new_row_df = pd.DataFrame([csv_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
            # CSV 파일 덮어쓰기
            updated_df.to_csv(csv_results_file, index=False, encoding='utf-8-sig')
            print(f"📊 CSV 결과 저장: {csv_results_file}")
        
        # 중간 결과 저장 (JSON)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 진행 상황 출력
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
    
    # 실패한 실험 목록 출력
    if failed_count > 0:
        print("\n❌ 실패한 실험 목록:")
        for result in results:
            if result["status"] == "failed":
                print(f"\n  - {result['experiment_id']}")
                if result.get("stderr"):
                    # stderr의 마지막 5줄만 출력
                    stderr_lines = result["stderr"].strip().split('\n')
                    error_preview = '\n'.join(stderr_lines[-5:])
                    print(f"    에러: {error_preview}")
                elif result.get("error"):
                    print(f"    에러: {result['error']}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="배치 인과추론 실험 실행")
    
    # 환경변수에서 기본값 가져오기 (Docker에서 사용)
    default_config = os.environ.get(
        "EXPERIMENT_CONFIG",
        "experiment_config.json"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        nargs='?',  # 선택적 인자로 만들기
        help="실험 설정 JSON 파일 경로"
    )
    
    args = parser.parse_args()
    
    # 현재 스크립트의 디렉토리를 기준으로 경로 설정
    script_dir = Path(__file__).parent
    
    # config 인자가 None이면 기본값 사용
    config_arg = args.config if args.config is not None else default_config
    
    # 설정 파일 경로 결정 (절대 경로 또는 상대 경로)
    if os.path.isabs(config_arg):
        config_path = Path(config_arg)
    else:
        config_path = script_dir / config_arg
    
    if not config_path.exists():
        # 상대 경로로 시도
        config_path = Path(config_arg)
        if not config_path.exists():
            print(f"❌ 설정 파일을 찾을 수 없습니다: {config_arg}")
            print(f"   현재 디렉토리: {os.getcwd()}")
            print(f"   스크립트 디렉토리: {script_dir}")
            print(f"   시도한 경로: {config_path}")
            print(f"\n설정 파일 예시를 생성합니다...")
            create_example_config(script_dir / "experiment_config.json")
            return
    
    # 설정 로드
    print(f"📄 설정 파일 로드: {config_path}")
    config = load_experiment_config(str(config_path))
    
    # 배치 실험 실행
    run_batch_experiments(config, script_dir)


def create_example_config(config_file: Path):
    """예시 설정 파일을 생성합니다."""
    example_config = {
        "data_dir": "data",
        "graph_data_dir": "graph_data",
        "api_key": None,
        "auto_extract_treatments": True,
        "graphs": [
            "main_graph",
            "dummy_graph"
        ],
        "treatments": [
            "ACCR_CD",
            "CARR_MYCT1",
            "NTR_BPLC_PSNT_WAGE_AMT"
        ],
        "outcomes": [
            "ACQ_180_YN"
        ],
        "estimators": [
            "tabpfn",
            "linear_regression"
        ],
        "no_logs": False,
        "verbose": False,
        "comment": "auto_extract_treatments가 true이면 graphs와 treatments는 무시되고, 각 graph 파일에서 자동으로 추출됩니다. api_key는 GPT API 키를 설정하세요 (예: \"sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"). null이면 환경변수 LLM_API_KEY를 사용합니다."
    }
    
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 예시 설정 파일 생성: {config_file}")
    print("\n설정 파일을 수정한 후 다시 실행해주세요.")


if __name__ == "__main__":
    main()

