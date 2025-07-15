"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import logging
import pathlib
import pickle
import pprint
import textwrap
from typing import Iterable

import pandas as pd

from mlkaps.modeling.encoding import encode_dataframe
from mlkaps.modeling.model_wrapper import ModelWrapper
from mlkaps.modeling.optuna_model_tuner import OptunaModelTuner, OptunaRecorder
from mlkaps.sampling.experiment import Objective


class ModelingError(Exception):
    """Exception raised for errors in the modeling process."""

    pass


class SurrogateFactory:
    """Factory class to easily build surrogate models according to the configuration."""

    def __init__(
        self,
        sampled_data: pd.DataFrame,
        *,
        parameters: dict,
        modeling_method: str = None,
        output_directory: pathlib.Path = None,
        model_parameters: dict = None,
        model_name: str = None,
        time_budget: int = None,
        n_trials: int = None,
        record: bool = True,
    ):
        """Initialize the SurrogateFactory.

        Args:
            sampled_data (pd.DataFrame): The training data for the surrogate models.
            parameters (dict): Dictionary mapping parameter names to their parameter value objects.
            modeling_method (str, optional): The modeling method to use (e.g., 'lightgbm', 'xgboost', 'optuna').
            output_directory (pathlib.Path, optional): Directory for saving optuna recordings and model outputs.
            model_parameters (dict, optional): Parameters to pass to the model constructor.
            model_name (str, optional): Name of the model to use (for optuna tuning).
            time_budget (int, optional): Time budget in seconds for optuna tuning.
            n_trials (int, optional): Number of trials for optuna tuning.
            record (bool, optional): Whether to record optuna tuning sessions. Defaults to True.

        Raises:
            ValueError: If parameters dict is None.
        """

        # Validate required parameters
        if parameters is None:
            raise ValueError("parameters dict is required")

        self.parameters = parameters
        self.modeling_method = modeling_method
        self.output_directory = output_directory
        self.model_parameters = model_parameters or {}
        self.model_name = model_name
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.record = record

        # Ensure the data is correctly encoded to the right type
        self.sampled_data = encode_dataframe(parameters, sampled_data)

    def _build_optuna_tuner(self, X: pd.DataFrame, y: Iterable):
        """Build an optuna tuner according to the passed configuration.

        Args:
            X (pd.DataFrame): The input of the model.
            y (Iterable): The target/objective of the model.

        Returns:
            tuple: A tuple containing:
                - tuner: The optuna tuner instance.
                - time_budget (int): Time budget for tuning.
                - n_trials (int): Number of trials for tuning.

        Raises:
            ModelingError: If no tuner could be found for the model type specified in the configuration.
        """

        # First try fetch the correct tuner
        if self.model_name is None:
            raise ModelingError("model_name is required for optuna tuning")

        tuner = OptunaModelTuner.known_tuners.get(self.model_name, None)
        if tuner is None:
            raise ModelingError(f"Could not find optuna tuner with name '{self.model_name}'")

        # Get the tuning budget
        time_budget = self.time_budget
        n_trials = self.n_trials

        if time_budget is None and n_trials is None:
            logging.warning("No budget was set for optuna, defaulting to 10 minutes per tuning session")
            time_budget = 10 * 60

        tuner = tuner(X, y)

        # Check if we should record the tuning session
        if self.record and self.output_directory is not None:
            tuner = OptunaRecorder(tuner, self.output_directory / f"optuna_records_for_{self.model_name}")

        return tuner, time_budget, n_trials

    def _build_model_using_optuna(self, X: pd.DataFrame, y: Iterable) -> ModelWrapper:
        """Build a tuned model using optuna.

        Args:
            X (pd.DataFrame): The input of the model.
            y (Iterable): The target/objective of the model.

        Returns:
            ModelWrapper: A tuned and fitted model.
        """

        tuner, time_budget, n_trials = self._build_optuna_tuner(X, y)

        model, params = tuner.run(time_budget=time_budget, n_trials=n_trials)

        msg = textwrap.indent(pprint.pformat(params), "\t")
        logging.info(f"Finished building model with optuna, parameters are\n{msg}")

        return model

    def _build_model_using_parameters(self, model_name: str, X: pd.DataFrame, y: Iterable) -> ModelWrapper:
        """Build a model using the default hyperparameters or one present in the configuration.

        Args:
            model_name (str): The name of the model to use.
            X (pd.DataFrame): The input of the model.
            y (Iterable): The target/objective of the model.

        Returns:
            ModelWrapper: The fitted model.

        Raises:
            ModelingError: If no model was found with the given name.
        """

        model = ModelWrapper.known_models.get(model_name, None)

        if model is None:
            raise ModelingError(f"Could not find model wrapper with name '{model_name}'")

        model = model(**self.model_parameters)
        model.fit(X, y)
        return model

    def build(self, objective: Objective, inputs: Iterable[str] = None, model_name: str = None) -> ModelWrapper:
        """Build a new model for the given objective.

        Args:
            objective (Objective): The objective/label to fit the model on.
            inputs (Iterable[str], optional): A list of features to fit the model on. Defaults to None.
            model_name (str, optional): The name of the model type to use. Defaults to None.

        Returns:
            ModelWrapper: A fitted model.
        """

        if model_name is None:
            model_name = self.modeling_method

        if inputs is None:
            inputs = list(self.parameters.keys())

        X = self.sampled_data[inputs]
        y = self.sampled_data[objective.name]

        if model_name == "optuna":
            surrogate = self._build_model_using_optuna(X, y)
        else:
            surrogate = self._build_model_using_parameters(model_name, X, y)

        return surrogate


def build_main_surrogates(
    sampled_data: pd.DataFrame,
    *,
    parameters: dict,
    objectives: list,
    modeling_method: str,
    output_directory: pathlib.Path,
    model_parameters: dict = None,
    model_name: str = None,
    time_budget: int = None,
    n_trials: int = None,
    record: bool = True,
) -> dict:
    """Factory function for building one surrogate per objective.

    Args:
        sampled_data (pd.DataFrame): The training data for the surrogate models.
        parameters (dict): Dictionary mapping parameter names to their parameter value objects.
        objectives (list): List of objective names to build surrogates for.
        modeling_method (str): The modeling method to use (e.g., 'lightgbm', 'xgboost', 'optuna').
        output_directory (pathlib.Path): Directory for saving model outputs.
        model_parameters (dict, optional): Parameters to pass to the model constructor.
        model_name (str, optional): Name of the model to use (for optuna tuning).
        time_budget (int, optional): Time budget in seconds for optuna tuning.
        n_trials (int, optional): Number of trials for optuna tuning.
        record (bool, optional): Whether to record optuna tuning sessions. Defaults to True.

    Returns:
        dict: Dictionary mapping objective names to their surrogate models.
    """
    surrogate_models = {}
    factory = SurrogateFactory(
        sampled_data,
        parameters=parameters,
        modeling_method=modeling_method,
        output_directory=output_directory,
        model_parameters=model_parameters,
        model_name=model_name,
        time_budget=time_budget,
        n_trials=n_trials,
        record=record,
    )

    for obj in objectives:
        surrogate_models[obj] = factory.build(obj)
        with open(output_directory / (obj.name + "_model.pkl"), "wb") as f:
            pickle.dump(surrogate_models[obj.name], f)

    return surrogate_models
