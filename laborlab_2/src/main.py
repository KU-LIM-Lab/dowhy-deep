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
from typing import Dict, Any, List, Tuple, Optional

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
    setup_logging,
    load_config
)
from .estimation import (
    run_single_experiment,
    run_inference,
    create_experiment_list,
    prepare_data_for_causal_model
)
from datetime import datetime
import json


def preprocess(
    data_dir_path: Path,
    seis_data_dir: str,
    limit_data: bool = False,
    limit_size: int = 5000,
    job_category_file: str = "KSIC"
) -> pd.DataFrame:
    """
    전처리 함수 (limit_data 옵션으로 데이터 제한 가능)
    
    Args:
        data_dir_path: 데이터 디렉토리 경로
        seis_data_dir: seis_data 디렉토리명
        limit_data: 데이터 제한 여부 (기본값: False)
        limit_size: 제한할 데이터 크기 (기본값: 5000)
        job_category_file: 직종 소분류 파일명 (KECO, KSCO, KSIC 중 선택, 기본값: KSIC)
    
    Returns:
        merged_df: 전처리 및 병합된 데이터프레임
    """
    print("="*80)
    print("1️⃣ 데이터 로드 시작")
    print("="*80)
    
    file_list, causal_graph = load_all_data(
        str(data_dir_path), 
        seis_data_dir, 
        graph_file=None
    )
    
    print("\n" + "="*80)
    if limit_data:
        print("2️⃣ 데이터 전처리 및 3️⃣ 데이터 병합 시작 (제한 모드)")
        print(f"\n(Test mode): 전처리 전에 각 파일에서 앞 {limit_size}개만 사용합니다.")
    else:
        print("2️⃣ 데이터 전처리 및 3️⃣ 데이터 병합 시작")
    print("="*80)
    
    print(f"📋 사용할 직종 소분류 파일: job_subcategories_{job_category_file}.csv")
    print("⚡ JSON 파일 4개(이력서, 자기소개서, 직업훈련, 자격증) 병렬 처리 시작")
    preprocessing_start = time.time()
    
    merged_df = preprocess_and_merge_data(
        file_list, 
        str(data_dir_path), 
        limit_data=limit_data, 
        limit_size=limit_size,
        job_category_file=job_category_file
    )
    print(f"✅ 최종 병합 데이터: {len(merged_df)}건, {len(merged_df.columns)}개 변수")
    
    preprocessing_elapsed = time.time() - preprocessing_start
    print(f"⏱️ 전처리 및 병합 완료! 소요 시간: {preprocessing_elapsed:.2f}초")
    
    return merged_df


def learning(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str = "tabpfn",
    experiment_id: Optional[str] = None,
    logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    단일 실험에 대한 learning 함수
    단일 그래프, 트리트먼트, estimator로 한 번의 estimation(fitting) 및 refutation을 진행하고 결과를 로깅
    
    Args:
        merged_df_clean: 전처리된 데이터프레임
        graph_file: 그래프 파일 경로
        treatment: 트리트먼트 변수명
        outcome: 결과 변수명
        estimator: 추정 방법 (기본값: tabpfn)
        experiment_id: 실험 ID (선택적)
        logger: 로거 객체 (선택적)
    
    Returns:
        실험 결과 딕셔너리
    """
    if experiment_id is None:
        experiment_id = f"{Path(graph_file).stem}_{treatment}_{outcome}_{estimator}"
    
    print(f"\n{'='*80}")
    print(f"🎓 Learning 실행 - {experiment_id}")
    print(f"그래프: {Path(graph_file).name}")
    print(f"Treatment: {treatment}, Outcome: {outcome}, Estimator: {estimator}")
    print(f"{'='*80}\n")
    
    result = run_single_experiment(
        merged_df_clean=merged_df_clean,
        graph_file=graph_file,
        treatment=treatment,
        outcome=outcome,
        estimator=estimator,
        experiment_id=experiment_id,
        logger=logger
    )
    
    if result["status"] == "success":
        print(f"✅ Learning 완료: {experiment_id}")
        print(f"   ATE 값: {result.get('ate_value', 'N/A')}")
        print(f"   F1 Score: {result.get('f1_score', 'N/A')}")
        print(f"   AUC: {result.get('auc', 'N/A')}")
    else:
        print(f"❌ Learning 실패: {experiment_id}")
        if result.get("error"):
            print(f"   에러: {result['error']}")
    
    return result


def _run_experiments_batch(
    experiment_list: List[Tuple[str, str, str, str]],
    experiment_func,
    experiment_type: str,
    merged_df_clean: pd.DataFrame,
    logger: Optional[Any] = None,
    output_dir: Optional[Path] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    실험 배치 실행 공통 함수
    
    Args:
        experiment_list: 실험 조합 리스트
        experiment_func: 단일 실험 실행 함수
        experiment_type: 실험 타입 ("learning" 또는 "prediction")
        merged_df_clean: 전처리된 데이터프레임
        logger: 로거 객체
        output_dir: 출력 디렉토리
        **kwargs: experiment_func에 전달할 추가 인자
    
    Returns:
        실험 결과 리스트
    """
    emoji_map = {"learning": "🎓", "prediction": "🔮"}
    emoji = emoji_map.get(experiment_type, "🔬")
    
    print("="*80)
    print(f"{emoji} {experiment_type.capitalize()} Experiments 실행")
    print("="*80)
    
    total_experiments = len(experiment_list)
    print(f"\n📊 총 {total_experiments}개의 {experiment_type} 실험을 실행합니다.\n")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV 설정
    csv_file = None
    csv_columns = None
    if output_dir:
        results_file = output_dir / f"{experiment_type}_results_{timestamp}.json"
        csv_file = output_dir / f"{experiment_type}_results_{timestamp}.csv"
        
        if experiment_type == "learning":
            csv_columns = [
                'graph_name', 'treatment', 'estimator', 'ate_value',
                'placebo_passed', 'placebo_pvalue',
                'unobserved_passed', 'unobserved_pvalue',
                'subset_passed', 'subset_pvalue',
                'dummy_passed', 'dummy_pvalue',
                'f1_score', 'auc', 'duration_seconds'
            ]
        else:  # prediction
            csv_columns = [
                'graph_name', 'treatment', 'estimator', 'f1_score', 'auc', 'accuracy'
            ]
        
        pd.DataFrame(columns=csv_columns).to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # 실험 실행
    for idx, (graph_file, treatment, outcome, estimator) in enumerate(experiment_list, 1):
        experiment_id = f"exp_{idx:04d}_{Path(graph_file).stem}_{treatment}_{outcome}_{estimator}"
        
        print(f"\n[{idx}/{total_experiments}] {experiment_type.capitalize()} 실행 중...")
        
        # 단일 실험 실행
        result = experiment_func(
            merged_df_clean=merged_df_clean,
            graph_file=graph_file,
            treatment=treatment,
            outcome=outcome,
            estimator=estimator,
            experiment_id=experiment_id,
            logger=logger,
            **kwargs
        )
        
        results.append(result)
        
        # CSV에 결과 추가
        if output_dir and csv_file and result.get("status") == "success":
            if experiment_type == "learning":
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
            else:  # prediction
                metrics = result.get("metrics", {})
                csv_row = {
                    'graph_name': result.get('graph_name', ''),
                    'treatment': result.get('treatment', ''),
                    'estimator': result.get('estimator', ''),
                    'f1_score': metrics.get('f1_score'),
                    'auc': metrics.get('auc'),
                    'accuracy': metrics.get('accuracy')
                }
            
            # CSV 파일에 결과 추가
            try:
                existing_df = pd.read_csv(csv_file, encoding='utf-8-sig')
            except (FileNotFoundError, pd.errors.EmptyDataError):
                existing_df = pd.DataFrame(columns=csv_columns)
            
            new_row_df = pd.DataFrame([csv_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            updated_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 중간 결과 저장 (JSON)
        if output_dir:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        failed_count = sum(1 for r in results if r.get("status") == "failed")
        print(f"\n✅ 성공: {success_count}, ❌ 실패: {failed_count}")
    
    # 최종 요약
    print(f"\n{'='*80}")
    print(f"📋 {experiment_type.capitalize()} Experiments 완료")
    print(f"{'='*80}")
    print(f"총 실험 수: {total_experiments}")
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    print(f"성공: {success_count}")
    print(f"실패: {failed_count}")
    if output_dir:
        print(f"JSON 결과 파일: {results_file}")
        print(f"CSV 결과 파일: {csv_file}")
    print(f"{'='*80}\n")
    
    return results


def learning_experiments(
    merged_df_clean: pd.DataFrame,
    experiment_list: List[Tuple[str, str, str, str]],
    logger: Optional[Any] = None,
    output_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    experiment_list의 모든 조합에 대해 learning 실행
    
    Args:
        merged_df_clean: 전처리된 데이터프레임
        experiment_list: 실험 조합 리스트 [(graph_file, treatment, outcome, estimator), ...]
        logger: 로거 객체 (선택적)
        output_dir: 출력 디렉토리 (선택적)
    
    Returns:
        실험 결과 리스트
    """
    return _run_experiments_batch(
        experiment_list=experiment_list,
        experiment_func=learning,
        experiment_type="learning",
        merged_df_clean=merged_df_clean,
        logger=logger,
        output_dir=output_dir
    )


def prediction(
    merged_df_clean: pd.DataFrame,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    checkpoint_dir: Path,
    experiment_id: Optional[str] = None,
    logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    단일 실험에 대한 prediction 함수
    단일 그래프, 트리트먼트 및 학습된 모델을 불러와 prediction 진행 및 결과 지표 출력
    
    Args:
        merged_df_clean: 전처리된 데이터프레임
        graph_file: 그래프 파일 경로
        treatment: 트리트먼트 변수명
        outcome: 결과 변수명
        estimator: 추정 방법
        checkpoint_dir: checkpoint 디렉토리 경로
        experiment_id: 실험 ID (선택적)
        logger: 로거 객체 (선택적)
    
    Returns:
        예측 결과 딕셔너리
    """
    if experiment_id is None:
        experiment_id = f"{Path(graph_file).stem}_{treatment}_{outcome}_{estimator}"
    
    print(f"\n{'='*80}")
    print(f"🔮 Prediction 실행 - {experiment_id}")
    print(f"그래프: {Path(graph_file).name}")
    print(f"Treatment: {treatment}, Outcome: {outcome}, Estimator: {estimator}")
    print(f"{'='*80}\n")
    
    try:
        result = run_inference(
            merged_df_clean=merged_df_clean,
            graph_file=graph_file,
            checkpoint_dir=checkpoint_dir,
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
        
        if result.get("status") == "success":
            metrics = result.get("metrics", {})
            print(f"✅ Prediction 완료: {experiment_id}")
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"   F1 Score: {metrics.get('f1_score', 'N/A')}")
            print(f"   AUC: {metrics.get('auc', 'N/A')}")
        else:
            print(f"❌ Prediction 실패: {experiment_id}")
            if result.get("error"):
                print(f"   에러: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Prediction 실패: {experiment_id}")
        print(f"   에러: {e}")
        return {
            "status": "failed",
            "experiment_id": experiment_id,
            "error": str(e)
        }


def prediction_experiments(
    merged_df_clean: pd.DataFrame,
    experiment_list: List[Tuple[str, str, str, str]],
    checkpoint_dir: Path,
    logger: Optional[Any] = None,
    output_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    experiment_list의 모든 조합에 대해 prediction 실행 및 결과 지표를 CSV로 저장
    
    Args:
        merged_df_clean: 전처리된 데이터프레임
        experiment_list: 실험 조합 리스트 [(graph_file, treatment, outcome, estimator), ...]
        checkpoint_dir: checkpoint 디렉토리 경로
        logger: 로거 객체 (선택적)
        output_dir: 출력 디렉토리 (선택적)
    
    Returns:
        예측 결과 리스트
    """
    return _run_experiments_batch(
        experiment_list=experiment_list,
        experiment_func=prediction,
        experiment_type="prediction",
        merged_df_clean=merged_df_clean,
        logger=logger,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir
    )




def main():
    parser = argparse.ArgumentParser(description="LaborLab 2 인과추론 분석 파이프라인")
    parser.add_argument("--config", type=str, default="config.json", help="설정 파일 경로")
    
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
    limit_data = config.get("limit_data", False)
    limit_size = config.get("limit_size", 5000)
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoint")
    job_category_file = config.get("job_category_file", "KSIC")
    
    # 새로운 설정 변수
    do_preprocess = config.get("preprocess", True)
    do_learning = config.get("learning", False)
    do_prediction = config.get("prediction", False)
    do_experiment = config.get("experiment", False)
    
    # 절대 경로로 변환
    data_dir_path = script_dir / data_dir
    output_dir_path = script_dir / output_dir
    output_dir_path.mkdir(exist_ok=True)
    checkpoint_dir_path = script_dir / checkpoint_dir
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 로거 설정
    logger = None
    if not config.get("no_logs", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = setup_logging(
            log_dir=output_dir_path,
            log_filename=f"pipeline_{timestamp}.log"
        )
        if logger:
            logger.info(f"파이프라인 시작 - {timestamp}")
    
    merged_df = None
    merged_df_clean = None
    experiment_list = []
    
    # ========================================================================
    # 1. 전처리 실행
    # ========================================================================
    if do_preprocess:
        merged_df = preprocess(
            data_dir_path=data_dir_path,
            seis_data_dir=seis_data_dir,
            limit_data=limit_data,
            limit_size=limit_size,
            job_category_file=job_category_file
        )
        
        # 인과 모델을 위한 데이터 준비
        merged_df_clean = prepare_data_for_causal_model(
            merged_df=merged_df,
            config=config,
            data_dir_path=data_dir_path,
            graph_data_dir=graph_data_dir
        )
    
    # ========================================================================
    # 2. experiment_list 생성
    # ========================================================================
    if do_experiment or do_learning or do_prediction:
        experiment_list = create_experiment_list(
            config=config,
            data_dir_path=data_dir_path,
            graph_data_dir=graph_data_dir
        )
        
        if not experiment_list:
            print("❌ 유효한 실험 조합이 없습니다.")
            return
        
        print(f"\n📊 생성된 실험 조합: {len(experiment_list)}개\n")
    
    # ========================================================================
    # 3. Learning 실행
    # ========================================================================
    if do_learning:
        if merged_df_clean is None:
            print("❌ 전처리된 데이터가 없습니다. preprocess를 먼저 실행하세요.")
            return
        
        if do_experiment:
            # experiment_list의 모든 조합에 대해 learning 실행
            learning_experiments(
                merged_df_clean=merged_df_clean,
                experiment_list=experiment_list,
                logger=logger,
                output_dir=output_dir_path
            )
        else:
            # 단일 실험 실행 (config에서 첫 번째 실험 조합 사용)
            if experiment_list:
                graph_file, treatment, outcome, estimator = experiment_list[0]
                learning(
                    merged_df_clean=merged_df_clean,
                    graph_file=graph_file,
                    treatment=treatment,
                    outcome=outcome,
                    estimator=estimator,
                    logger=logger
                )
            else:
                print("❌ 실행할 실험이 없습니다.")
    
    # ========================================================================
    # 4. Prediction 실행
    # ========================================================================
    if do_prediction:
        if merged_df_clean is None:
            print("❌ 전처리된 데이터가 없습니다. preprocess를 먼저 실행하세요.")
            return
        
        if do_experiment:
            # experiment_list의 모든 조합에 대해 prediction 실행
            prediction_experiments(
                merged_df_clean=merged_df_clean,
                experiment_list=experiment_list,
                checkpoint_dir=checkpoint_dir_path,
                logger=logger,
                output_dir=output_dir_path
            )
        else:
            # 단일 실험 실행 (config에서 첫 번째 실험 조합 사용)
            if experiment_list:
                graph_file, treatment, outcome, estimator = experiment_list[0]
                prediction(
                    merged_df_clean=merged_df_clean,
                    graph_file=graph_file,
                    treatment=treatment,
                    outcome=outcome,
                    estimator=estimator,
                    checkpoint_dir=checkpoint_dir_path,
                    logger=logger
                )
            else:
                print("❌ 실행할 실험이 없습니다.")
    
    print(f"\n{'='*80}")
    print("✅ 파이프라인 완료")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
