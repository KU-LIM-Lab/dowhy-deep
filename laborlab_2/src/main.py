"""
LaborLab 2 - 인과추론 분석 메인 파이프라인

전체 파이프라인:
1. 경로를 통해 데이터 로드
1-1. (Test mode) 전처리과정이 잘 되는지를 확인하기 위해 로드된 데이터의 앞에서 5000개만 잘라서 사용
2. 데이터 전처리
3. 데이터 병합
4. train test split (1:99)
5. causal graph 로드해서 실험정의
6. 각 실험별 estimation - refutation - prediction 진행 후 결과저장
"""
import argparse
import pandas as pd
import warnings
from pathlib import Path
import os
import time
import itertools

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# DoWhy 로거 레벨 설정
import logging as dowhy_logging
dowhy_logging.getLogger("dowhy.causal_estimator").setLevel(dowhy_logging.WARNING)
dowhy_logging.getLogger("dowhy.causal_estimators").setLevel(dowhy_logging.WARNING)

# 유틸리티 함수 임포트
from .utils import (
    load_all_data,
    preprocess_and_merge_data,
    clean_dataframe_for_causal_model,
    create_causal_graph,
    extract_treatments_from_graph,
    find_all_graph_files,
    setup_logging,
    load_config,
    run_single_experiment,
    run_inference,
    save_predictions_to_excel
)
from . import estimation
from datetime import datetime
import json


def main():
    parser = argparse.ArgumentParser(description="LaborLab 2 인과추론 분석 파이프라인")
    
    default_config = os.environ.get(
        "config.json"
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
    
    # 설정 값 추출
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
    limit_data = config.get("limit_data", False)  # 5000개 제한 옵션
    limit_size = config.get("limit_size", 5000)  # 제한할 데이터 크기
    mode = config.get("mode", "learning")  # "learning" 또는 "inference"
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoint")  # checkpoint 디렉토리
    
    # 절대 경로로 변환
    data_dir_path = script_dir / data_dir
    
    # ========================================================================
    # 1. 경로를 통해 데이터 로드
    # ========================================================================
    print("="*80)
    print("1️⃣ 데이터 로드 시작")
    print("="*80)
    
    file_list, causal_graph = load_all_data(
        str(data_dir_path), 
        seis_data_dir, 
        graph_file=None
    )
    
    # ========================================================================
    # 2. 데이터 전처리 및 3. 데이터 병합
    # ========================================================================
    print("\n" + "="*80)
    print("2️⃣ 데이터 전처리 및 3️⃣ 데이터 병합 시작")
    print("="*80)
    
    # 테스트 모드 안내
    if limit_data:
        print(f"\n(Test mode): 전처리 전에 각 파일에서 앞 {limit_size}개만 사용합니다.")
    
    print("⚡ JSON 파일 4개(이력서, 자기소개서, 직업훈련, 자격증) 병렬 처리 시작")
    preprocessing_start = time.time()
    
    merged_df = preprocess_and_merge_data(file_list, str(data_dir_path), api_key=api_key, limit_data=limit_data, limit_size=limit_size)
    print(f"✅ 최종 병합 데이터: {len(merged_df)}건, {len(merged_df.columns)}개 변수")
    
    preprocessing_elapsed = time.time() - preprocessing_start
    print(f"⏱️ 전처리 및 병합 완료! 소요 시간: {preprocessing_elapsed:.2f}초")
    
    # ========================================================================
    # 그래프 파일 경로 처리
    # ========================================================================
    graph_files = []
    
    if auto_extract_treatments:
        print(f"\n🔍 그래프 파일에서 자동으로 treatment 추출 중...")
        found_graphs = find_all_graph_files(data_dir_path, graph_data_dir)
        graph_files = [str(g) for g in found_graphs]
        
        if not graph_files:
            print(f"⚠️ {graph_data_dir} 폴더에서 그래프 파일을 찾을 수 없습니다.")
        else:
            print(f"✅ {len(graph_files)}개의 그래프 파일 발견")
    else:
        for graph in graphs:
            if isinstance(graph, str):
                graph_path = data_dir_path / graph_data_dir / graph
                if graph_path.exists():
                    graph_files.append(str(graph_path))
                else:
                    graph_path = Path(graph)
                    if graph_path.exists():
                        graph_files.append(str(graph_path))
    
    if not graph_files:
        print("❌ 유효한 그래프 파일이 없습니다.")
        return
    
    # ========================================================================
    # 5. causal graph 로드해서 실험정의
    # ========================================================================
    print("\n" + "="*80)
    print("5️⃣ Causal Graph 로드 및 실험 정의")
    print("="*80)
    
    # 모든 그래프의 변수 수집
    all_graph_variables = set()
    for graph_file in graph_files:
        graph_path = Path(graph_file)
        try:
            causal_graph = create_causal_graph(str(graph_path))
            all_graph_variables.update(causal_graph.nodes())
        except Exception as e:
            print(f"⚠️ 그래프 파일 로드 실패 ({graph_path.name}): {e}")
    
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
    
    # 실험 조합 생성 (learning과 inference 모두에서 사용)
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
    
    # 데이터 정리
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
    # HOPE_JSCD3_NAME은 그래프에 포함되지 않지만 데이터에는 유지해야 함 (직종소분류별 분리용)
    stratification_vars = {"HOPE_JSCD3_NAME"}
    required_vars = list(all_graph_variables | essential_vars | stratification_vars)
    
    merged_df_clean = clean_dataframe_for_causal_model(
        merged_df, 
        required_vars=required_vars, 
        logger=None
    )
    
    data_variables = set(merged_df_clean.columns)
    vars_to_keep = (all_graph_variables | essential_vars | stratification_vars) & data_variables
    vars_to_remove = data_variables - vars_to_keep
    
    if vars_to_remove:
        print(f"🗑️ 그래프에 정의되지 않은 변수 제거 중 ({len(vars_to_remove)}개)...")
        merged_df_clean = merged_df_clean[list(vars_to_keep)]
    
    print(f"✅ 정리된 데이터: {len(merged_df_clean)}건, {len(merged_df_clean.columns)}개 변수")
    print("="*80 + "\n")
    
    # ========================================================================
    # 모드에 따른 분기: learning 또는 inference
    # ========================================================================
    if mode == "inference":
        # ========================================================================
        # Inference 모드: checkpoint 로드 후 바로 prediction
        # ========================================================================
        print("="*80)
        print("🔮 Inference 모드 실행")
        print("="*80)
        
        # checkpoint 디렉토리 경로 설정
        checkpoint_dir_path = script_dir / checkpoint_dir
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        logger = None
        if not config.get("no_logs", False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_path = script_dir / output_dir
            output_dir_path.mkdir(exist_ok=True)
            logger = setup_logging(
                log_dir=output_dir_path,
                log_filename=f"inference_{timestamp}.log"
            )
            if logger:
                logger.info(f"Inference 모드 시작 - {timestamp}")
        
        # 실험 조합은 이미 위에서 생성됨
        total_experiments = len(experiment_combinations)
        print(f"\n📊 총 {total_experiments}개의 inference 실험을 실행합니다.\n")
        
        # Inference 실행
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_path = script_dir / output_dir
        output_dir_path.mkdir(exist_ok=True)
        
        for idx, (graph_file, treatment, outcome, estimator) in enumerate(experiment_combinations, 1):
            experiment_id = f"exp_{idx:04d}_{Path(graph_file).stem}_{treatment}_{outcome}_{estimator}"
            
            print(f"\n[{idx}/{total_experiments}] Inference 실행 중...")
            
            try:
                result = run_inference(
                    merged_df_clean=merged_df_clean,
                    graph_file=graph_file,
                    checkpoint_dir=checkpoint_dir_path,
                    treatment=treatment,
                    outcome=outcome,
                    estimator=estimator,
                    logger=logger,
                    experiment_id=experiment_id
                )
                result["experiment_id"] = experiment_id
                result["graph_name"] = Path(graph_file).stem
                result["treatment"] = treatment
                result["outcome"] = outcome
                result["estimator"] = estimator
                results.append(result)
                
            except Exception as e:
                print(f"❌ Inference 실패: {experiment_id}")
                print(f"   에러: {e}")
                results.append({
                    "status": "failed",
                    "experiment_id": experiment_id,
                    "error": str(e)
                })
        
        # 결과 저장
        results_file = output_dir_path / f"inference_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        failed_count = sum(1 for r in results if r.get("status") == "failed")
        
        print(f"\n{'='*80}")
        print("📋 Inference 완료")
        print(f"{'='*80}")
        print(f"총 실험 수: {total_experiments}")
        print(f"성공: {success_count}")
        print(f"실패: {failed_count}")
        print(f"결과 파일: {results_file}")
        print(f"{'='*80}\n")
        
        return
    
    # ========================================================================
    # Learning 모드: 기존 파이프라인 (estimation → refutation → prediction)
    # ========================================================================
    print("="*80)
    print("🎓 Learning 모드 실행")
    print("="*80)
    
    # ========================================================================
    # 4. train test split (1:99) - 실제로는 run_analysis_without_preprocessing에서 수행
    # ========================================================================
    # 주의: train/test split은 각 실험 실행 시 run_analysis_without_preprocessing 내부에서 수행됩니다.
    # 여기서는 전체 데이터를 준비만 합니다.
    
    # ========================================================================
    # 6. 각 실험별 estimation - refutation - prediction 진행 후 결과저장
    # ========================================================================
    print("="*80)
    print("6️⃣ 각 실험별 estimation - refutation - prediction 진행")
    print("="*80)
    
    # 결과 저장
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = script_dir / output_dir
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


if __name__ == "__main__":
    main()
