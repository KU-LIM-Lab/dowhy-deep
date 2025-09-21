import importlib
import logging
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils import resample

import torch

from dowhy.causal_estimator import CausalEstimate, CausalEstimator, IdentifiedEstimand

logger = logging.getLogger(__name__)


class TabpfnEstimator(CausalEstimator):
    """
    
    A causal estimator that uses Tabular PFN (TabPFN) for effect estimation.
    This estimator supports backdoor identification methods.
    
    """

    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        test_significance: Union[bool, str] = False,
        evaluate_effect_strength: bool = False,
        confidence_intervals: bool = False,
        num_null_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction: int = CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level: float = CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates: Union[bool, str] = "auto",
        num_quantiles_to_discretize_cont_cols: int = CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        method_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            identified_estimand=identified_estimand,
            test_significance=test_significance,
            evaluate_effect_strength=evaluate_effect_strength,
            confidence_intervals=confidence_intervals,
            num_null_simulations=num_null_simulations,
            num_simulations=num_simulations,
            sample_size_fraction=sample_size_fraction,
            confidence_level=confidence_level,
            need_conditional_estimates=need_conditional_estimates,
            num_quantiles_to_discretize_cont_cols=num_quantiles_to_discretize_cont_cols,
            **kwargs,
        )
        self.logger.info("INFO: Using TabPFNEstimator")
        self.method_params = method_params if method_params is not None else {}
        self.tabpfn_model = None
        
        self._check_tabpfn_dependencies()
        self._device = self._get_device()

    def _check_tabpfn_dependencies(self):
        """Checks if TabPFN and PyTorch are installed."""
        try:
            importlib.import_module("tabpfn")
            importlib.import_module("torch")
        except ImportError:
            raise ImportError(
                "TabPFNEstimator requires tabpfn and torch to be installed. "
                "Please install them with: pip install tabpfn torch"
            )

    def _get_device(self):
        """Determines the appropriate PyTorch device (GPU or CPU) to use."""
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info("INFO: Using GPU for TabPFN. Make sure CUDA is enabled.")
        else:
            device = "cpu"
            self.logger.warning(
                "WARNING: No GPU found. TabPFN will run on CPU, which can be very slow."
                "For better performance, consider using a GPU instance."
            )
        return torch.device(device)

    def construct_symbolic_estimator(self, estimand):
        """Constructs a symbolic expression for the estimator."""
        expr = "E[{} | do({})] = E[{} | {}, {}]".format(
            self._target_estimand.outcome_variable[0],
            self._target_estimand.treatment_variable[0],
            self._target_estimand.outcome_variable[0],
            ", ".join(self._target_estimand.treatment_variable),
            ", ".join(self._target_estimand.get_backdoor_variables()),
        )
        return expr

    def _get_model_class(self, outcome_data):
        """Returns the appropriate TabPFN class (Classifier or Regressor) based on the outcome data type."""
        is_binary_classification = (
            outcome_data.nunique() <= 2
            and outcome_data.dtype in ['int64', 'int32', 'object', 'category', 'bool']
        )
        if is_binary_classification:
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier, "Classifier"
        else:
            from tabpfn import TabPFNRegressor
            return TabPFNRegressor, "Regressor"

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """Preprocesses data and fits the TabPFN model."""
        self.reset_encoders() # Forget any existing encoders
        
        self._data = data
        self._set_effect_modifiers(data, effect_modifier_names)

        self._treatment_name = self._target_estimand.treatment_variable
        self._outcome_name = self._target_estimand.outcome_variable
        self._observed_common_causes_names = self._target_estimand.get_backdoor_variables()

        if self._target_estimand.identifier_method != "backdoor":
            raise NotImplementedError("TabPFNEstimator only supports the backdoor identification method.")
        
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        # 1. 결과 변수를 먼저 추출하여 모델 클래스 결정
        outcome_data = self._data[self._outcome_name].iloc[:, 0]
        model_class, model_type = self._get_model_class(outcome_data)
        
        # 2. 모델에 입력될 모든 피처 컬럼 이름을 정의
        self.all_feature_names = (
            self._observed_common_causes_names + self._treatment_name + self._effect_modifier_names
        )
        
        # 3. 해당 피처들로 데이터프레임 생성
        features_df = self._data[self.all_feature_names]
        
        # 4. dowhy의 내장 _encode 메소드를 사용하여 원-핫 인코딩
        encoded_features_df = self._encode(features_df, "features")
        
        # 5. Numpy 배열로 변환
        X = encoded_features_df.to_numpy(dtype=np.float32)

        if model_type == "Classifier":
            y, _ = pd.factorize(outcome_data)
            y = y.astype(np.int64) 
        else:
            y = outcome_data.to_numpy(dtype=np.float32)

        # 6. TabPFN 모델 초기화 및 학습
        tabpfn_params = {
            "device": self._device,
        }

        if "N_ensemble_configurations" in self.method_params:
            tabpfn_params["n_estimators"] = self.method_params["N_ensemble_configurations"]

        self.tabpfn_model = model_class(**tabpfn_params)
        
        # 7. 데이터셋 제한사항 경고 (sample size, feature size, class size)
        num_samples, num_features = X.shape
        if num_samples > 10000:
            self.logger.warning(
                "WARNING: TabPFN performs best on datasets with up to 10,000 samples. "
                "Your dataset has %d samples.", num_samples
            )
        if num_features > 500:
            self.logger.warning(
                "WARNING: TabPFN performs best on datasets with up to 500 features. "
                "Your dataset has %d features.", num_features
            )
        if model_type == "Classifier":
            num_classes = len(outcome_data.unique())
            if num_classes > 10:
                raise ValueError(
                    f"Number of classes {num_classes} exceeds the maximal number of classes supported by TabPFN(10). Consider reducing the number of classes."
                )

        self.tabpfn_model.fit(X, y)
        self.logger.info(f"TabPFN {model_type} model has been fitted successfully.")

        return self

    def estimate_effect(
        self,
        data: pd.DataFrame = None,
        treatment_value: Any = 1,
        control_value: Any = 0,
        target_units=None,
        need_conditional_estimates: bool = None,
        **_,
    ):
        """
        Estimates the causal effect (ATE) using the fitted TabPFN model.
        Supports estimating effects on custom target units.
        """
        # 1. 파라미터 저장 및 검증
        self._target_units = target_units
        self._treatment_value = treatment_value
        self._control_value = control_value
        
        if data is None:
            data = self._data
            
        if need_conditional_estimates is None:
            need_conditional_estimates = self.need_conditional_estimates
        
        # 2. 대상 데이터 결정
        if target_units is None or target_units == "ate":
            data_to_predict_on = data
        elif isinstance(target_units, pd.DataFrame):
            data_to_predict_on = target_units
        elif callable(target_units):
            data_to_predict_on = data.loc[target_units]
        else:
            raise ValueError("target_units must be None, 'ate', DataFrame, or callable")

        # 3. Counterfactual 데이터 생성
        control_data = data_to_predict_on.copy()
        treatment_data = data_to_predict_on.copy()

        for treatment in self._treatment_name:
            control_data[treatment] = self._control_value
            treatment_data[treatment] = self._treatment_value
        
        # 4. 피처 인코딩 및 텐서 변환
        control_features_df = control_data[self.all_feature_names]
        treatment_features_df = treatment_data[self.all_feature_names]

        encoded_control_df = self._encode(control_features_df, "features")
        encoded_treatment_df = self._encode(treatment_features_df, "features")

        X_control = encoded_control_df.to_numpy(dtype=np.float32)
        X_treatment = encoded_treatment_df.to_numpy(dtype=np.float32)

        # 5. 모델 예측 (Classifier/Regressor)
        if hasattr(self.tabpfn_model, 'predict_proba'):
            predictions_control = self.tabpfn_model.predict_proba(X_control)[:, 1]
            predictions_treatment = self.tabpfn_model.predict_proba(X_treatment)[:, 1]
        else:
            predictions_control = self.tabpfn_model.predict(X_control)
            predictions_treatment = self.tabpfn_model.predict(X_treatment)

        # 6. 효과 계산
        cate_estimates = predictions_treatment - predictions_control
        ate = np.mean(cate_estimates)
        
        # 7. 조건부 추정 처리
        conditional_estimates = None
        if need_conditional_estimates:
            conditional_estimates = pd.Series(cate_estimates, index=data_to_predict_on.index)
            
        # 8. Confidence intervals 처리
        effect_intervals = None
        if self._confidence_intervals:
            effect_intervals = self._estimate_confidence_intervals(ate)

        # 9. CausalEstimate 객체 생성
        estimate = CausalEstimate(
            data=data,
            treatment_name=self._treatment_name,
            outcome_name=self._outcome_name,
            estimate=ate,
            control_value=self._control_value,
            treatment_value=self._treatment_value,
            conditional_estimates=conditional_estimates,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            effect_intervals=effect_intervals
        )
        
        estimate.add_estimator(self)
        return estimate    

    def _estimate_confidence_intervals(self, estimate_value, confidence_level=None, method=None, **kwargs):
        """
        Confidence intervals 구현(DoWhy CausalEstimator 클래스 참고)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        return self._estimate_confidence_intervals_with_bootstrap(
            self._data,
            estimate_value=estimate_value,
            confidence_level=confidence_level,
            num_simulations=self.num_simulations,
            sample_size_fraction=self.sample_size_fraction,
        )
    
    def _generate_bootstrap_estimates(self, data, num_bootstrap_simulations, sample_size_fraction):
        """
        Bootstrap 구현
        CausalEstimator._generate_bootstrap_estimates() 오버라이드
        """
        simulation_results = np.zeros(num_bootstrap_simulations)
        sample_size = int(sample_size_fraction * len(data))
        
        self.logger.info(f"TabPFN Bootstrap: {num_bootstrap_simulations} simulations, sample_size: {sample_size}")
        
        for index in range(num_bootstrap_simulations):
            # Bootstrap 샘플 생성
            new_data = resample(data, n_samples=sample_size)
            
            # 새 estimator 생성 및 학습
            new_estimator = self.get_new_estimator_object(
                self._target_estimand,
                test_significance=False,
                evaluate_effect_strength=False,
                confidence_intervals=False,
            )
            
            new_estimator.fit(new_data, effect_modifier_names=self._effect_modifier_names)
            
            # 각 샘플에 대해 effect 추정
            new_effect = new_estimator.estimate_effect(
                new_data,
                treatment_value=self._treatment_value,
                control_value=self._control_value,
                target_units=self._target_units,
            )
            
            simulation_results[index] = new_effect.value
        
        # BootstrapEstimates 반환
        return CausalEstimator.BootstrapEstimates(
            simulation_results,
            {"num_simulations": num_bootstrap_simulations, "sample_size_fraction": sample_size_fraction}
        )
    