"""
DoWhy 라이브러리를 이용한 인과모델 구축, 추정, 검증 End-to-End 파이프라인
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

# DoWhy 라이브러리 임포트
import dowhy
from dowhy import CausalModel
import networkx as nx

# 로컬 DoWhy 라이브러리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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

def create_causal_graph():
    """
    NetworkX를 사용하여 인과 그래프를 직접 생성하는 함수
    
    Returns:
        nx.DiGraph: 인과 그래프 객체
    """
    # 방향성 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가 (실제 데이터에 있는 변수들만 사용)
    G.add_node("ACCR_CD", label="학력코드")
    G.add_node("ACQ_180_YN", label="180일이내취업여부")
    G.add_node("HOPE_WAGE_SM_AMT", label="희망임금합계금액")
    G.add_node("AVG_HOPE_WAGE_SM_AMT", label="평균희망임금합계금액")
    
    # 엣지 추가 (인과관계) - DAG 구조로 수정
    G.add_edge("ACCR_CD", "HOPE_WAGE_SM_AMT")
    G.add_edge("ACCR_CD", "AVG_HOPE_WAGE_SM_AMT")
    G.add_edge("HOPE_WAGE_SM_AMT", "ACQ_180_YN")
    G.add_edge("AVG_HOPE_WAGE_SM_AMT", "ACQ_180_YN")
    
    return G

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
    graph_name = Path(args.graph).stem
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
    logger.info(f"데이터: {args.data}, 그래프: {args.graph}")
    logger.info(f"처치: {args.treatment}, 결과: {args.outcome}, 추정방법: {args.estimator}")
    
    return logger

def parse_arguments():
    """명령행 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description="DoWhy 인과추론 분석")
    
    parser.add_argument('--data', type=str, required=True, help='데이터 파일 경로')
    parser.add_argument('--graph', type=str, required=True, help='그래프 파일 경로')
    parser.add_argument('--estimator', type=str, choices=['tabpfn', 'linear_regression', 'propensity_score', 'instrumental_variable'], 
                       default='linear_regression', help='추정 방법')
    parser.add_argument('--treatment', type=str, default='ACCR_CD', help='처치 변수명')
    parser.add_argument('--outcome', type=str, default='ACQ_180_YN', help='결과 변수명')
    parser.add_argument('--no-logs', action='store_true', help='로그 저장 비활성화')
    parser.add_argument('--verbose', action='store_true', help='상세 출력 활성화')
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    args = parse_arguments()
    logger = setup_logging(args)
    
    try:
        print(f"\n🚀 DoWhy 인과추론 분석 시작")
        print(f"📊 데이터: {args.data}, 🕸️ 그래프: {args.graph}")
        print(f"🎯 처치: {args.treatment}, 📈 결과: {args.outcome}, 🔧 추정방법: {args.estimator}")
        print("="*60)
        
        # 1. 데이터 로드 및 전처리
        print("1️⃣ 데이터 로드 및 전처리 중...")
        df = preprocess.load_and_preprocess_data(args.data)
        causal_graph = create_causal_graph()
        
        # 2. 인과모델 생성 및 분석
        print("2️⃣ 인과모델 생성 중...")
        model = CausalModel(data=df, treatment=args.treatment, outcome=args.outcome, graph=causal_graph)
        
        print("3️⃣ 인과효과 식별 중...")
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        print("4️⃣ 인과효과 추정 중...")
        estimate = estimation.estimate_causal_effect(model, identified_estimand, args.estimator, logger)
        
        print("5️⃣ 검증 테스트 실행 중...")
        validation_results = estimation.run_validation_tests(model, identified_estimand, estimate, logger)
        
        print("6️⃣ 민감도 분석 실행 중...")
        sensitivity_df = estimation.run_sensitivity_analysis(model, identified_estimand, estimate, logger)
        
        print("7️⃣ 시각화 생성 중...")
        heatmap_path = estimation.create_sensitivity_heatmap(sensitivity_df, logger) if not sensitivity_df.empty else None
        
        print("8️⃣ 최종 요약 보고서 출력 중...")
        estimation.print_summary_report(estimate, validation_results, sensitivity_df)
        
        if logger:
            logger.info("분석 완료")
        
        print(f"\n✅ 전체 분석 완료!")
        
    except Exception as e:
        if logger:
            logger.error(f"분석 중 오류 발생: {e}")
        print(f"❌ 분석 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
