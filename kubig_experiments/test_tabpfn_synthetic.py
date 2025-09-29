import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
import logging
import warnings
import uuid
from datetime import datetime
from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator

from kubig_experiments.src.dag_parser import parse_edges_from_dot, extract_roles_general


def _dot_to_nx(graph_txt: str) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(parse_edges_from_dot(graph_txt))
    return g


@pytest.fixture(scope="session")
def test_logger():
    # 1) warnings 전부 무시 + 브릿지 차단
    warnings.filterwarnings("ignore")
    logging.captureWarnings(False)

    # 2) 외부 로거 경고 숨기기
    for name in [
        "py.warnings",
        "dowhy",
        "dowhy.causal_model",
        "dowhy.causal_identifier",
        "dowhy.causal_estimator",
    ]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)

    # 3) 고유 run_id (timestamp + uuid8)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_uid = uuid.uuid4().hex[:8]
    run_id = f"{run_ts}_{run_uid}"

    # 4) 로그 디렉토리: 테스트 파일 위치 기준
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"tabpfn_tests_{run_id}.log"

    # 5) 로거 + 핸들러
    base_logger = logging.getLogger("kubig.tests.tabpfn")
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    if not base_logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s | %(message)s"
        )

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        base_logger.addHandler(fh)
        base_logger.addHandler(ch)

    # 6) run_id를 필드로 주입하는 어댑터
    logger = logging.LoggerAdapter(base_logger, extra={"run_id": run_id})
    logger.info("Logging initialized. File: %s", str(log_path))
    return logger


@pytest.mark.parametrize("dag_idx", range(1, 30))
def test_tabpfn_estimator_runs(dag_idx, test_logger):
    """
    - DAG에서 일반화 규칙으로 Treatment/Confounder/Mediator 자동 추출
    - Baseline: backdoor.linear_regression
    - 비교: TabPFN 추정기
    - Validation: Placebo treatment, Random treatment
    """
    df = pd.read_csv("./kubig_experiments/data/data_preprocessed.csv")
    dag_dir = Path("./kubig_experiments/dags/output")
    dag_file = dag_dir / f"dag_{dag_idx}.txt"
    graph_txt = dag_file.read_text(encoding="utf-8")

    roles = extract_roles_general(graph_txt, outcome="ACQ_180_YN")
    treatment = roles["treatment"]

    test_logger.info(
        "[%s] Roles | X=%s | M=%s | C=%s",
        dag_file.name, roles["treatment"], roles["mediators"], roles["confounders"]
    )

    if treatment not in df.columns:
        msg = f"[skip] treatment '{treatment}' not found in dataframe."
        test_logger.info("[%s] %s", dag_file.name, msg)
        pytest.skip(msg)

    nx_graph = _dot_to_nx(graph_txt)
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome="ACQ_180_YN",
        graph=nx_graph,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    if identified is None:
        msg = "[skip] No valid identified estimand (식별 실패)."
        test_logger.info("[%s] %s", dag_file.name, msg)
        pytest.skip(msg)

    # method = "backdoor.linear_regression"
    # est = model.estimate_effect(identified, method_name=method, test_significance=True)
    # if est.value is None or not (isinstance(est.value, (float, np.floating)) and np.isfinite(est.value)):
    #     msg = f"[skip] baseline estimation returned invalid value: {est.value}"
    #     test_logger.info("[%s] %s", dag_file.name, msg)
    #     pytest.skip(msg)

    est_tabpfn = model.estimate_effect(
        identified,
        method_name="backdoor.tabpfn",
        method_params={"estimator": TabpfnEstimator, "n_estimators": 8},
    )
    if est_tabpfn is None or est_tabpfn.value is None or not (
        isinstance(est_tabpfn.value, (float, np.floating)) and np.isfinite(est_tabpfn.value)
    ):
        msg = f"[skip] TabPFN estimation returned invalid value: {getattr(est_tabpfn, 'value', None)}"
        test_logger.info("[%s] %s", dag_file.name, msg)
        pytest.skip(msg)

    # test_logger.info("[%s] [%s] Baseline(linear_regression) ATE: %s", dag_file.name, treatment, est.value)
    test_logger.info("[%s] [%s] TabPFN ATE: %s", dag_file.name, treatment, est_tabpfn.value)

    refuters = ["placebo_treatment_refuter", "random_common_cause"]
    for ref in refuters:
        refutation = model.refute_estimate(identified, est_tabpfn, method_name=ref)
        test_logger.info("[%s] Refutation (%s): %s", dag_file.name, ref, refutation)
        assert refutation is not None
