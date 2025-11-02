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
from typing import List, Dict, Any


def load_experiment_config(config_file: str) -> Dict[str, Any]:
    """실험 설정 파일을 로드합니다."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_single_experiment(
    data_dir: str,
    graph_file: str,
    treatment: str,
    outcome: str,
    estimator: str,
    base_dir: Path,
    experiment_id: str,
    no_logs: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """단일 실험을 실행합니다."""
    print(f"\n{'='*80}")
    print(f"실험 ID: {experiment_id}")
    print(f"그래프: {Path(graph_file).name}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    print(f"Estimator: {estimator}")
    print(f"{'='*80}\n")
    
    # 명령어 구성
    cmd = [
        sys.executable,
        "-m", "src.main",
        "--data-dir", data_dir,
        "--graph", graph_file,
        "--treatment", treatment,
        "--outcome", outcome,
        "--estimator", estimator,
    ]
    
    if no_logs:
        cmd.append("--no-logs")
    if verbose:
        cmd.append("--verbose")
    
    # 실험 실행
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            cwd=base_dir,  # laborlab 디렉토리에서 실행
            capture_output=True,
            text=True,
            check=True
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "experiment_id": experiment_id,
            "status": "success",
            "duration_seconds": duration,
            "graph": graph_file,
            "treatment": treatment,
            "outcome": outcome,
            "estimator": estimator,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }
    except subprocess.CalledProcessError as e:
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
            "stdout": e.stdout,
            "stderr": e.stderr,
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
    
    # 절대 경로로 변환
    data_dir_path = base_dir / data_dir
    if not data_dir_path.is_absolute():
        data_dir_path = base_dir / data_dir
    
    # 그래프 파일 경로 처리
    graph_files = []
    for graph in graphs:
        if isinstance(graph, str):
            graph_path = base_dir / data_dir / graph
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
    
    # 실험 조합 생성
    experiment_combinations = list(itertools.product(
        graph_files,
        treatments,
        outcomes,
        estimators
    ))
    
    total_experiments = len(experiment_combinations)
    print(f"\n📊 총 {total_experiments}개의 실험을 실행합니다.")
    print(f"   - 그래프: {len(graph_files)}개")
    print(f"   - Treatment: {len(treatments)}개")
    print(f"   - Outcome: {len(outcomes)}개")
    print(f"   - Estimator: {len(estimators)}개\n")
    
    # 결과 저장
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = base_dir / "log" / f"batch_experiments_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    # 실험 실행
    for idx, (graph_file, treatment, outcome, estimator) in enumerate(experiment_combinations, 1):
        experiment_id = f"exp_{idx:04d}_{Path(graph_file).stem}_{treatment}_{outcome}_{estimator}"
        
        print(f"\n[{idx}/{total_experiments}] 실험 실행 중...")
        
        result = run_single_experiment(
            data_dir=str(data_dir_path),
            graph_file=graph_file,
            treatment=treatment,
            outcome=outcome,
            estimator=estimator,
            base_dir=base_dir,
            experiment_id=experiment_id,
            no_logs=config.get("no_logs", False),
            verbose=config.get("verbose", False)
        )
        
        results.append(result)
        
        # 중간 결과 저장
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
    print(f"결과 파일: {results_file}")
    print(f"{'='*80}\n")
    
    # 실패한 실험 목록 출력
    if failed_count > 0:
        print("❌ 실패한 실험 목록:")
        for result in results:
            if result["status"] == "failed":
                print(f"  - {result['experiment_id']}: {result.get('error', 'Unknown error')}")


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
        "verbose": False
    }
    
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 예시 설정 파일 생성: {config_file}")
    print("\n설정 파일을 수정한 후 다시 실행해주세요.")


if __name__ == "__main__":
    main()

