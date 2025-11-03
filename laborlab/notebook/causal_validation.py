"""
DoWhy 라이브러리를 이용한 인과모델 구축, 추정, 검증 End-to-End 파이프라인

이 스크립트는 dummy_data.csv와 dummy_graph를 사용하여
학력코드(ACCR_CD)가 180일이내취업여부(ACQ_180_YN)에 미치는 인과효과를 분석합니다.
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 작동하도록 설정
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import logging
from datetime import datetime
import os

import dowhy
from dowhy import CausalModel
import networkx as nx

# TabPFN 모델 사용 (필요한 라이브러리 확인 후 임포트)
import sys
import os
# 로컬 DoWhy 라이브러리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import tabpfn
    import torch
    from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator
    TABPFN_AVAILABLE = True
    print("✓ TabPFN Estimator 사용 가능")
except ImportError as e:
    TABPFN_AVAILABLE = False
    print(f"⚠️ TabPFN Estimator 사용 불가: {e}")
    print("필요한 라이브러리 설치: pip install tabpfn torch")

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# DoWhy 특정 로거들의 INFO 메시지만 제거
import logging as dowhy_logging
# DoWhy의 반복적인 메시지들만 제거
dowhy_logging.getLogger("dowhy.causal_estimator").setLevel(dowhy_logging.WARNING)
dowhy_logging.getLogger("dowhy.causal_estimators").setLevel(dowhy_logging.WARNING)

# 한글 폰트 설정 (시각화용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# CONFIG 설정 섹션
# =============================================================================

# 데이터 및 그래프 설정
script_dir = Path(__file__).parent
DATA_CONFIG = {
    'data_file': str(script_dir / 'data' / 'dummy_data.csv'),           # 사용할 데이터 파일
    'graph_file': str(script_dir / 'data' / 'dummy_graph'),             # 사용할 그래프 파일
    'treatment': 'ACCR_CD',                  # 처치 변수 (학력코드)
    'outcome': 'ACQ_180_YN',                 # 결과 변수 (180일이내취업여부)
}

# 추정 방법 설정 (TabPFN 사용 가능 여부에 따라 결정)
if 'TABPFN_AVAILABLE' in globals() and TABPFN_AVAILABLE:
    ESTIMATION_CONFIG = {
        'method': 'backdoor.tabpfn',  # 추정 방법: TabPFN
        'test_significance': True,               # 통계적 유의성 검정 수행
        'proceed_when_unidentifiable': True,     # 식별 불가능할 때도 진행
    }
else:
    ESTIMATION_CONFIG = {
        'method': 'backdoor.linear_regression',  # 추정 방법: linear regression (fallback)
        'test_significance': True,               # 통계적 유의성 검정 수행
        'proceed_when_unidentifiable': True,     # 식별 불가능할 때도 진행
    }

# 검증 설정 -> 추후 수정필요(현재 기본값 사용)
VALIDATION_CONFIG = {
    'placebo_treatment': {
        'method': 'placebo_treatment_refuter',
        'placebo_type': 'permute',
        'num_simulations': 100
    },
    'unobserved_confounder': {
        'method': 'add_unobserved_common_cause',
        'confounders_effect_on_treatment': 'binary_flip',
        'confounders_effect_on_outcome': 'linear',
        'effect_strength_on_treatment': 0.10,
        'effect_strength_on_outcome': 0.10,
        'num_simulations': 100
    },
    'data_subset': {
        'method': 'data_subset_refuter',
        'subset_fraction': 0.8,
        'num_simulations': 200,
        'random_state': 42
    },
    'dummy_outcome': {
        'method': 'dummy_outcome',
        'num_simulations': 200
    },
    'sensitivity_analysis': {
        'effect_strength_range': (0.0, 0.5),
        'num_points': 11,
        'num_simulations': 200
    }
}

# 시각화 설정
VISUALIZATION_CONFIG = {
    'figsize': (10, 8),
    'dpi': 100,
    'save_plots': True,
    'plot_format': 'png'
}

# 로깅 설정
LOGGING_CONFIG = {
    'save_logs': True,
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_level': 20  # logging.INFO = 20
}

# =============================================================================
# 로깅 설정 함수들
# =============================================================================

def setup_logging(graph_file, treatment, config):
    """
    로깅을 설정하는 함수
    
    Args:
        graph_file (str): 그래프 파일명
        treatment (str): 처치 변수명
        config (dict): 로깅 설정
    
    Returns:
        str: 생성된 로그 파일명
    """
    if not config['save_logs']:
        return None
    
    # log 폴더 생성 (스크립트가 있는 디렉토리 기준)
    script_dir = Path(__file__).parent
    log_dir = script_dir / "log"
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 생성: 그래프명_처치변수_날짜시간.log
    graph_name = Path(graph_file).stem  # 파일 확장자 제거
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{graph_name}_{treatment}_{timestamp}.log"
    log_filepath = log_dir / log_filename
    
    # 로깅 설정
    logging.basicConfig(
        level=config['log_level'],
        format=config['log_format'],
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()  # 콘솔에도 출력
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"로깅 시작 - 로그 파일: {log_filepath}")
    logger.info(f"그래프 파일: {graph_file}")
    logger.info(f"처치 변수: {treatment}")
    logger.info(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return str(log_filepath)

def log_estimation_results(logger, estimate, method_name):
    """
    추정 결과를 로깅하는 함수
    
    Args:
        logger: 로거 객체
        estimate: 추정된 인과효과 객체
        method_name (str): 추정 방법명
    """
    logger.info("="*60)
    logger.info("인과 효과 추정 결과")
    logger.info("="*60)
    logger.info(f"추정 방법: {method_name}")
    logger.info(f"추정된 인과 효과 (ATE): {estimate.value:.6f}")
    
    if hasattr(estimate, 'p_value') and estimate.p_value is not None:
        logger.info(f"P-value: {estimate.p_value:.6f}")
        significance = "유의함" if estimate.p_value <= 0.05 else "유의하지 않음"
        logger.info(f"통계적 유의성: {significance}")
    
    # 추정치의 신뢰구간이 있다면 로깅
    if hasattr(estimate, 'confidence_intervals'):
        logger.info(f"신뢰구간: {estimate.confidence_intervals}")

def log_validation_results(logger, validation_results):
    """
    검증 결과를 로깅하는 함수
    
    Args:
        logger: 로거 객체
        validation_results (dict): 검증 결과 딕셔너리
    """
    logger.info("="*60)
    logger.info("검증 결과 요약")
    logger.info("="*60)
    
    # 가상 원인 테스트
    if validation_results.get('placebo'):
        placebo = validation_results['placebo']
        effect_change = abs(placebo.new_effect - placebo.estimated_effect)
        status = "통과" if effect_change < 0.01 else "실패"
        logger.info(f"가상 원인 테스트: {status}")
        logger.info(f"  - 기존 추정치: {placebo.estimated_effect:.6f}")
        logger.info(f"  - 가상처치 후 추정치: {placebo.new_effect:.6f}")
        logger.info(f"  - 효과 변화: {effect_change:.6f}")
    
    # 미관측 교란 테스트
    if validation_results.get('unobserved'):
        unobserved = validation_results['unobserved']
        change_rate = abs(unobserved.new_effect - unobserved.estimated_effect) / abs(unobserved.estimated_effect)
        status = "강건함" if change_rate < 0.2 else "민감함"
        logger.info(f"미관측 교란 테스트: {status}")
        logger.info(f"  - 기존 추정치: {unobserved.estimated_effect:.6f}")
        logger.info(f"  - 교란 추가 후 추정치: {unobserved.new_effect:.6f}")
        logger.info(f"  - 변화율: {change_rate:.2%}")
    
    # 부분표본 안정성 테스트
    if validation_results.get('subset'):
        subset = validation_results['subset']
        logger.info(f"부분표본 안정성 테스트:")
        logger.info(f"  - 기존 추정치: {subset.estimated_effect:.6f}")
        logger.info(f"  - 부분표본 추정치: {subset.new_effect:.6f}")
    
    # 더미 결과 테스트
    if validation_results.get('dummy'):
        dummy = validation_results['dummy']
        status = "통과" if abs(dummy.new_effect) < 0.01 else "실패"
        logger.info(f"더미 결과 테스트: {status}")
        logger.info(f"  - 더미 결과 추정치: {dummy.new_effect:.6f}")

def log_sensitivity_analysis(logger, sensitivity_df, config):
    """
    민감도 분석 결과를 로깅하는 함수
    
    Args:
        logger: 로거 객체
        sensitivity_df (pd.DataFrame): 민감도 분석 결과
        config (dict): 민감도 분석 설정
    """
    logger.info("="*60)
    logger.info("민감도 분석 결과")
    logger.info("="*60)
    
    logger.info(f"효과 강도 범위: {config['effect_strength_range'][0]} ~ {config['effect_strength_range'][1]}")
    logger.info(f"그리드 포인트 수: {config['num_points']}")
    logger.info(f"시뮬레이션 수: {config['num_simulations']}")
    logger.info(f"분석된 조합 수: {len(sensitivity_df)}")
    
    if not sensitivity_df.empty:
        logger.info(f"효과 범위: {sensitivity_df['new_effect'].min():.6f} ~ {sensitivity_df['new_effect'].max():.6f}")
        
        # 효과가 0에 가까운 지점 찾기
        min_abs_effect = sensitivity_df.loc[sensitivity_df['new_effect'].abs().idxmin()]
        logger.info(f"최소 절대 효과 지점:")
        logger.info(f"  - 처치 강도 (et): {min_abs_effect['effect_strength_on_treatment']:.2f}")
        logger.info(f"  - 결과 강도 (eo): {min_abs_effect['effect_strength_on_outcome']:.2f}")
        logger.info(f"  - 효과값: {min_abs_effect['new_effect']:.6f}")
        
        # 효과가 음수인 조합 수
        negative_effects = len(sensitivity_df[sensitivity_df['new_effect'] < 0])
        logger.info(f"음수 효과 조합 수: {negative_effects} ({negative_effects/len(sensitivity_df)*100:.1f}%)")
        
        # 효과가 0에 가까운 조합 수 (절대값 < 0.01)
        near_zero_effects = len(sensitivity_df[sensitivity_df['new_effect'].abs() < 0.01])
        logger.info(f"0에 가까운 효과 조합 수: {near_zero_effects} ({near_zero_effects/len(sensitivity_df)*100:.1f}%)")

def log_heatmap_info(logger, heatmap_path, config):
    """
    히트맵 정보를 로깅하는 함수
    
    Args:
        logger: 로거 객체
        heatmap_path (str): 히트맵 파일 경로
        config (dict): 시각화 설정
    """
    logger.info("="*60)
    logger.info("시각화 결과")
    logger.info("="*60)
    
    if heatmap_path and os.path.exists(heatmap_path):
        file_size = os.path.getsize(heatmap_path)
        logger.info(f"히트맵 파일: {heatmap_path}")
        logger.info(f"파일 크기: {file_size:,} bytes")
        logger.info(f"이미지 해상도: {config['figsize'][0]}x{config['figsize'][1]} inches")
        logger.info(f"DPI: {config['dpi']}")
    else:
        logger.warning("히트맵 파일이 생성되지 않았습니다.")

# =============================================================================
# 유틸리티 함수들
# =============================================================================

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

def preprocess_data(df):
    """
    데이터 전처리 함수
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    df_processed = df.copy()
    
    # ACCR_CD를 숫자로 인코딩
    accr_mapping = {
        '고등학교': 1,
        '(2/3년제) 대학': 2,
        '4년제 대학': 3,
        '대학원': 4,
        '중퇴': 5
    }
    
    df_processed['ACCR_CD'] = df_processed['ACCR_CD'].map(accr_mapping)
    df_processed = df_processed.dropna(subset=['ACCR_CD', 'ACQ_180_YN'])
    
    return df_processed

def load_data_and_graph(data_file, graph_file):
    """
    데이터를 로드하는 함수
    
    Args:
        data_file (str): 데이터 파일 경로
        graph_file (str): 그래프 파일 경로 (사용하지 않음)
    
    Returns:
        tuple: (데이터프레임, 그래프 객체)
    """
    df = pd.read_csv(data_file)
    df_processed = preprocess_data(df)
    causal_graph = create_causal_graph()
    
    return df_processed, causal_graph

def create_causal_model(df, causal_graph, treatment, outcome):
    """
    인과모델을 생성하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임
        causal_graph (nx.DiGraph): NetworkX 그래프 객체
        treatment (str): 처치 변수명
        outcome (str): 결과 변수명
    
    Returns:
        CausalModel: DoWhy 인과모델 객체
    """
    return CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=causal_graph
    )

def identify_effect(model, proceed_when_unidentifiable=True):
    """
    인과효과를 식별하는 함수
    
    Args:
        model (CausalModel): 인과모델 객체
        proceed_when_unidentifiable (bool): 식별 불가능할 때도 진행할지 여부
    
    Returns:
        IdentifiedEstimand: 식별된 추정량 객체
    """
    return model.identify_effect(proceed_when_unidentifiable=proceed_when_unidentifiable)

def estimate_effect(model, identified_estimand, method_name, test_significance=True, logger=None):
    """
    인과효과를 추정하는 함수
    
    Args:
        model (CausalModel): 인과모델 객체
        identified_estimand: 식별된 추정량 객체
        method_name (str): 추정 방법명
        test_significance (bool): 통계적 유의성 검정 수행 여부
        logger: 로거 객체 (선택사항)
    
    Returns:
        CausalEstimate: 추정된 인과효과 객체
    """
    estimate = model.estimate_effect(
        identified_estimand,
        method_name=method_name,
        test_significance=test_significance
    )
    
    if logger:
        log_estimation_results(logger, estimate, method_name)
    
    return estimate

def run_validation_tests(model, identified_estimand, estimate, config, logger=None):
    """
    다양한 검증 테스트를 실행하는 함수
    
    Args:
        model (CausalModel): 인과모델 객체
        identified_estimand: 식별된 추정량 객체
        estimate: 추정된 인과효과 객체
        config (dict): 검증 설정 딕셔너리
        logger: 로거 객체 (선택사항)
    
    Returns:
        dict: 검증 결과 딕셔너리
    """
    validation_results = {}
    
    # 1. 가상 원인 테스트 (Placebo Treatment)
    try:
        placebo_config = config['placebo_treatment']
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name=placebo_config['method'],
            placebo_type=placebo_config['placebo_type'],
            num_simulations=placebo_config['num_simulations']
        )
        validation_results['placebo'] = refute_placebo
    except Exception as e:
        validation_results['placebo'] = None
    
    # 2. 미관측 공통 원인 추가 테스트
    try:
        unobserved_config = config['unobserved_confounder']
        refute_unobserved = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name=unobserved_config['method'],
            confounders_effect_on_treatment=unobserved_config['confounders_effect_on_treatment'],
            confounders_effect_on_outcome=unobserved_config['confounders_effect_on_outcome'],
            effect_strength_on_treatment=unobserved_config['effect_strength_on_treatment'],
            effect_strength_on_outcome=unobserved_config['effect_strength_on_outcome'],
            num_simulations=unobserved_config['num_simulations']
        )
        validation_results['unobserved'] = refute_unobserved
    except Exception as e:
        validation_results['unobserved'] = None
    
    # 3. 부분표본 안정성 테스트
    try:
        subset_config = config['data_subset']
        refute_subset = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name=subset_config['method'],
            subset_fraction=subset_config['subset_fraction'],
            num_simulations=subset_config['num_simulations'],
            random_state=subset_config['random_state']
        )
        validation_results['subset'] = refute_subset
    except Exception as e:
        validation_results['subset'] = None
    
    # 4. 더미 결과 변수 테스트
    try:
        dummy_config = config['dummy_outcome']
        refute_dummy = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name=dummy_config['method'],
            num_simulations=dummy_config['num_simulations']
        )
        validation_results['dummy'] = refute_dummy
    except Exception as e:
        validation_results['dummy'] = None
    
    # 로깅
    if logger:
        log_validation_results(logger, validation_results)
    
    return validation_results

def run_sensitivity_analysis(model, identified_estimand, estimate, config, logger=None):
    """
    민감도 분석을 실행하는 함수
    
    Args:
        model (CausalModel): 인과모델 객체
        identified_estimand: 식별된 추정량 객체
        estimate: 추정된 인과효과 객체
        config (dict): 민감도 분석 설정
        logger: 로거 객체 (선택사항)
    
    Returns:
        pd.DataFrame: 민감도 분석 결과 데이터프레임
    """
    try:
        effect_range = config['effect_strength_range']
        num_points = config['num_points']
        num_simulations = config['num_simulations']
        
        grid = np.linspace(effect_range[0], effect_range[1], num_points)
        
        rows = []
        for i, et in enumerate(grid):
            for j, eo in enumerate(grid):
                try:
                    ref = model.refute_estimate(
                        identified_estimand, estimate,
                        method_name="add_unobserved_common_cause",
                        confounders_effect_on_treatment="binary_flip",
                        confounders_effect_on_outcome="linear",
                        effect_strength_on_treatment=et,
                        effect_strength_on_outcome=eo,
                        num_simulations=num_simulations
                    )
                    rows.append((et, eo, ref.new_effect))
                except Exception as e:
                    rows.append((et, eo, np.nan))
        
        sensitivity_df = pd.DataFrame(rows, columns=[
            "effect_strength_on_treatment", 
            "effect_strength_on_outcome", 
            "new_effect"
        ])
        
        if logger:
            log_sensitivity_analysis(logger, sensitivity_df, config)
        
        return sensitivity_df
        
    except Exception as e:
        if logger:
            logger.error(f"민감도 분석 중 오류 발생: {e}")
        return pd.DataFrame()

def create_sensitivity_heatmap(sensitivity_df, config, logger=None):
    """
    민감도 분석 결과를 히트맵으로 시각화하는 함수
    
    Args:
        sensitivity_df (pd.DataFrame): 민감도 분석 결과
        config (dict): 시각화 설정
        logger: 로거 객체 (선택사항)
    
    Returns:
        tuple: (matplotlib.figure.Figure, str) 생성된 그림 객체와 파일 경로
    """
    if sensitivity_df.empty:
        return None
    
    try:
        
        # 피벗 테이블 생성
        pivot = sensitivity_df.pivot(
            index="effect_strength_on_treatment",
            columns="effect_strength_on_outcome",
            values="new_effect"
        ).sort_index(ascending=True)
        
        # 히트맵 생성
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # 히트맵 그리기
        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            extent=[
                pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()
            ],
            cmap='RdYlBu_r'
        )
        
        # 색상막대 추가
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("New Effect (after unobserved confounding)", fontsize=12)
        
        # 0-컨투어 라인 추가
        X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
        CS = ax.contour(X, Y, pivot.values, levels=[0.0], linewidths=2, colors='black')
        ax.clabel(CS, inline=True, fmt="effect=0", fontsize=10)
        
        # 최소 절대값 지점 마커
        abs_min_idx = np.unravel_index(
            np.nanargmin(np.abs(pivot.values)), 
            pivot.values.shape
        )
        et_star = pivot.index.values[abs_min_idx[0]]
        eo_star = pivot.columns.values[abs_min_idx[1]]
        ax.plot(eo_star, et_star, marker="o", markersize=8, color='red')
        ax.annotate(
            f"Min effect at (et={et_star:.2f}, eo={eo_star:.2f})",
            (eo_star, et_star), 
            xytext=(10, 10), 
            textcoords="offset points", 
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
        )
        
        # 축 레이블 및 제목
        ax.set_xlabel("Effect Strength on Outcome (eo)", fontsize=12)
        ax.set_ylabel("Effect Strength on Treatment (et)", fontsize=12)
        ax.set_title("Sensitivity Analysis: Effect of Unobserved Confounders", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 그림 저장 (log 폴더에 저장)
        output_path = None
        if config['save_plots']:
            # log 폴더 경로 생성
            script_dir = Path(__file__).parent
            log_dir = script_dir / "log"
            log_dir.mkdir(exist_ok=True)
            
            # 파일명에 타임스탬프 추가
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensitivity_heatmap_{timestamp}.{config['plot_format']}"
            output_path = log_dir / filename
            
            plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        
        # plt.show()  # GUI 환경이 아닌 경우 주석 처리
        
        # 로깅
        if logger:
            log_heatmap_info(logger, output_path, config)
        
        return fig, output_path
        
    except Exception as e:
        if logger:
            logger.error(f"히트맵 생성 중 오류 발생: {e}")
        return None, None

def print_summary_report(estimate, validation_results, sensitivity_df):
    """
    전체 분석 결과 요약 보고서를 출력하는 함수
    
    Args:
        estimate: 추정된 인과효과 객체
        validation_results (dict): 검증 결과 딕셔너리
        sensitivity_df (pd.DataFrame): 민감도 분석 결과
    """
    print("\n" + "="*80)
    print("📋 최종 분석 결과 요약 보고서")
    print("="*80)
    
    # 기본 추정 결과
    print(f"\n🎯 주요 추정 결과:")
    print(f"  - 추정된 인과 효과 (ATE): {estimate.value:.6f}")
    if hasattr(estimate, 'p_value') and estimate.p_value is not None:
        significance = "유의함" if estimate.p_value <= 0.05 else "유의하지 않음"
        print(f"  - 통계적 유의성: {significance} (p-value: {estimate.p_value:.6f})")
    
    # 검증 결과 요약
    print(f"\n🔬 검증 결과 요약:")
    
    if validation_results.get('placebo'):
        placebo = validation_results['placebo']
        effect_change = abs(placebo.new_effect - placebo.estimated_effect)
        print(f"  - 가상 원인 테스트: {'통과' if effect_change < 0.01 else '실패'}")
    
    if validation_results.get('unobserved'):
        unobserved = validation_results['unobserved']
        change_rate = abs(unobserved.new_effect - unobserved.estimated_effect) / abs(unobserved.estimated_effect)
        print(f"  - 미관측 교란 테스트: {'강건함' if change_rate < 0.2 else '민감함'}")
    
    if validation_results.get('subset'):
        subset = validation_results['subset']
        print(f"  - 부분표본 안정성: 추정치 변화 확인됨")
    
    if validation_results.get('dummy'):
        dummy = validation_results['dummy']
        print(f"  - 더미 결과 테스트: {'통과' if abs(dummy.new_effect) < 0.01 else '실패'}")
    
    # 민감도 분석 요약
    if not sensitivity_df.empty:
        print(f"\n📈 민감도 분석 요약:")
        print(f"  - 분석된 조합 수: {len(sensitivity_df)}")
        print(f"  - 효과 범위: {sensitivity_df['new_effect'].min():.6f} ~ {sensitivity_df['new_effect'].max():.6f}")
        
        # 효과가 0에 가까운 지점 찾기
        min_abs_effect = sensitivity_df.loc[sensitivity_df['new_effect'].abs().idxmin()]
        print(f"  - 최소 절대 효과 지점: et={min_abs_effect['effect_strength_on_treatment']:.2f}, eo={min_abs_effect['effect_strength_on_outcome']:.2f}")
    
    print(f"\n✅ 전체 분석 완료!")

# =============================================================================
# 메인 실행 함수
# =============================================================================

def main():
    """
    메인 실행 함수
    """
    # 로깅 설정
    log_filename = setup_logging(
        DATA_CONFIG['graph_file'], 
        DATA_CONFIG['treatment'], 
        LOGGING_CONFIG
    )
    logger = logging.getLogger(__name__) if log_filename else None
    
    try:
        # 1. 데이터 및 그래프 로드
        df, causal_graph = load_data_and_graph(
            DATA_CONFIG['data_file'], 
            DATA_CONFIG['graph_file']
        )
        
        # 2. 인과모델 생성
        model = create_causal_model(
            df, 
            causal_graph, 
            DATA_CONFIG['treatment'], 
            DATA_CONFIG['outcome']
        )
        
        # 3. 인과효과 식별
        identified_estimand = identify_effect(
            model, 
            ESTIMATION_CONFIG['proceed_when_unidentifiable']
        )
        
        # 4. 인과효과 추정
        estimate = estimate_effect(
            model,
            identified_estimand,
            ESTIMATION_CONFIG['method'],
            ESTIMATION_CONFIG['test_significance'],
            logger
        )
        
        # 5. 검증 테스트 실행
        validation_results = run_validation_tests(
            model, 
            identified_estimand, 
            estimate, 
            VALIDATION_CONFIG,
            logger
        )
        
        # 6. 민감도 분석
        sensitivity_df = run_sensitivity_analysis(
            model, 
            identified_estimand, 
            estimate, 
            VALIDATION_CONFIG['sensitivity_analysis'],
            logger
        )
        
        # 7. 민감도 분석 히트맵 생성
        heatmap_path = None
        if not sensitivity_df.empty:
            _, heatmap_path = create_sensitivity_heatmap(sensitivity_df, VISUALIZATION_CONFIG, logger)
        
        # 8. 최종 요약 보고서 출력
        print_summary_report(estimate, validation_results, sensitivity_df)
        
        # 9. 로깅 완료 메시지
        if logger:
            logger.info("="*60)
            logger.info("분석 완료")
            logger.info("="*60)
            logger.info(f"분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if log_filename:
                logger.info(f"로그 파일: {log_filename}")
            if heatmap_path:
                logger.info(f"히트맵 파일: {heatmap_path}")
        
    except Exception as e:
        if logger:
            logger.error(f"분석 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
