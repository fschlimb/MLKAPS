"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import numpy as np
import pandas as pd
import pymoo.termination
from tqdm import tqdm
import os
import logging
import time
import pathlib


import pymoo.core.result
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableGA,
    MixedVariableSampling,
    MixedVariableMating,
    MixedVariableDuplicateElimination,
)
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Real, Binary, Integer
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination

from mlkaps.configuration import ExperimentConfig
from mlkaps.modeling.encoding import encode_dataframe
from mlkaps.sampling.sampler_factory import SamplerFactory

from mlkaps.sampling import RandomSampler
from mlkaps.sampling.sampler import ValueSet
from mlkaps.optimization.optimizer_checkpoint import OptimizerCheckpoint


def _get_parameter_types(parameters: dict) -> dict:
    """
    Extract parameter types from a parameters dictionary using get_dtype() on each parameter value container.

    Args:
        parameters (dict): Dictionary mapping parameter names to their parameter value objects (ValueContainer instances).

    Returns:
        dict: Dictionary mapping parameter names to their types as strings.
    """
    types = {}
    for name, param in parameters.items():
        dtype = param.get_dtype()
        # Map the dtype to the expected string values
        if dtype == "int":
            types[name] = "int"
        elif dtype == "float":
            types[name] = "float"
        elif dtype == str:
            types[name] = "categorical"
        elif dtype == bool:
            types[name] = "bool"
        else:
            # For backward compatibility, assume it's already a string
            types[name] = str(dtype)
    return types


def _parse_genetic_optimization_config(config_dict: dict):
    """
    Extract the genetic optimization parameters from the configuration dictionary.

    Args:
        config_dict (dict): Configuration dictionary containing optimization parameters.

    Returns:
        dict: Dictionary containing parsed optimization parameters.

    Raises:
        Exception: If optimization_parameters section is missing from the configuration dict.
    """
    if "optimization_parameters" not in config_dict:
        raise Exception("Missing optimization_parameters section in the configuration dict")

    parameter_section = config_dict["optimization_parameters"]

    selection_method = parameter_section.get("selection_method", "normalized_selection")

    # Load algorithm specific parameters
    optimization_parameters = parameter_section.get("evolution", {})

    # Build the termination criterion as a pymoo collection of termination object
    termination = parameter_section.get("termination", {})
    terminations = [get_termination(k, v) for k, v in termination.items()]
    termination_criterion = TerminationCollection(*terminations)

    # Initialize normalization coefficients
    normalization_coefficients = {}

    if "selection_parameters" in parameter_section:
        selection_parameters = parameter_section["selection_parameters"]
        coefficients = selection_parameters.get("coefficients", {})
        normalization_coefficients.update(coefficients)

    return {
        "selection_method": selection_method,
        "optimization_parameters": optimization_parameters,
        "termination_criterion": termination_criterion,
        "normalization_coefficients": normalization_coefficients,
    }


def create_genetic_optimizer_from_config(
    config_dict: dict, exp_config: ExperimentConfig, surrogate_models: dict, optimizer_checkpoint: OptimizerCheckpoint
):
    """
    Create a GeneticOptimizer from a configuration dictionary.

    Args:
        config_dict (dict): Configuration dictionary.
        exp_config (ExperimentConfig): Experiment configuration.
        surrogate_models (dict): Dictionary of surrogate models for each objective.
        optimizer_checkpoint (OptimizerCheckpoint): Checkpoint manager for optimization state.

    Returns:
        GeneticOptimizer: Configured genetic optimizer instance.

    Raises:
        Exception: If optimization method is not 'genetic' or if coefficient is set for undefined objective.
    """
    optim_section = config_dict["OPTIMIZATION"]
    optimization_method = optim_section["optimization_method"]

    # Ensure that the configuration is for a genetic optimizer
    if optimization_method != "genetic":
        raise Exception(
            "Tried to parse a genetic optimizer configuration, but the configuration dict "
            f"specifies a '{optimization_method}' optimization method"
        )

    # Parse optimization-specific parameters
    parsed_config = _parse_genetic_optimization_config(optim_section)

    # Create sampler
    sampler = SamplerFactory(exp_config).from_config(optim_section["sampling"]["sampler"]["sampling_method"])

    # Filter feature values based on input parameters
    filtered_features = {
        k: v for k, v in exp_config["parameters"]["features_values"].items() if k in exp_config.input_parameters
    }
    sampler.set_variables(filtered_features)

    sample_count = optim_section["sampling"]["sample_count"]
    early_stopping = optim_section["optimization_parameters"].get("early_stopping", False)

    # Validate objectives and set up normalization coefficients
    objectives = exp_config.objectives
    normalization_coefficients = parsed_config["normalization_coefficients"]

    # Handle objective-specific normalization coefficients
    for obj in normalization_coefficients:
        if obj not in objectives:
            raise Exception(f'Coefficient set for undefined objective "{obj}"')

    # Default every objective to the same weight if not specified
    for obj in objectives:
        if obj not in normalization_coefficients:
            normalization_coefficients[obj] = 1

    # Create the optimizer with keyword arguments
    return GeneticOptimizer(
        objectives=objectives,
        parameters=exp_config["parameters"]["features_values"],
        input_names=exp_config.input_parameters,
        optimization_parameters=parsed_config["optimization_parameters"],
        termination_criterion=parsed_config["termination_criterion"],
        normalization_coefficients=normalization_coefficients,
        selection_method=parsed_config["selection_method"],
        do_early_stopping=early_stopping,
        output_directory=exp_config.output_directory,
        sampler=sampler,
        samples_count=sample_count,
        surrogate_models=surrogate_models,
        optimizer_checkpoint=optimizer_checkpoint,
    )


class DesignParametersProblem(Problem):
    """
    Custom pymoo problem that uses the models generated in the modeling phase to estimate
    the objective values for a set of design parameters. Those objectives are used to evaluate a
    population of models to find the best parameters for a given kernel inputs.
    """

    def __init__(
        self, *, objectives: list, parameters: dict, input_names: list, surrogate_models: dict, model_type: str = "lightgbm"
    ):
        """
        Initialize the DesignParametersProblem.

        Args:
            objectives (list): List of objective names to optimize.
            parameters (dict): Dictionary mapping parameter names to their parameter value objects.
            input_names (list): List of input parameter names that are fixed during optimization.
            surrogate_models (dict): Dictionary of surrogate models for each objective.
            model_type (str): The type of model to use. Defaults to "lightgbm".

        Raises:
            ValueError: If more than 2 objectives are provided (only 1D and 2D optimization problems are currently supported).
        """

        # Kernels inputs will be defined later on
        self.kernel_inputs = None
        self.input_columns = None

        self.objectives = objectives
        self.parameters = parameters
        self.input_names = input_names
        self.surrogate_models = surrogate_models
        self.model_type = model_type

        # Extract the name and type of the optimization feature
        # Get parameter types from the parameters using get_dtype()
        parameters_type = _get_parameter_types(parameters)
        self.optimization_parameters = {k: t for k, t in parameters_type.items() if k not in self.input_names}

        self.objectives_count = len(self.objectives)

        if self.objectives_count > 2:
            raise ValueError("Only 1D and 2D optimization problems are currently supported")

        mixed_vars = self._define_vars()
        super().__init__(vars=mixed_vars, n_obj=self.objectives_count)

    def _define_vars(self):
        """
        Define the variables for the mixed precision optimization problem.

        Returns:
            dict: Dictionary mapping variable names to their pymoo variable definitions.

        Raises:
            ValueError: If an unexpected variable type is encountered.
        """
        # To define a mixed precision problem, we need to define each
        # variable, and their respective bound
        parameters = self.parameters

        mixed_vars = {}
        for name, parameter_type in self.optimization_parameters.items():
            match parameter_type:
                case "float":
                    pymoo_var = Real(bounds=parameters[name].get_sampling_bounds())
                case "int":
                    pymoo_var = Integer(bounds=parameters[name].get_sampling_bounds())
                case "bool":
                    pymoo_var = Binary()
                case "categorical":
                    assert isinstance(parameters[name], ValueSet)
                    pymoo_var = Choice(options=parameters[name].sample_linear_space())
                case _:
                    raise ValueError(f"Unexpected variable type for '{name}' ('{parameter_type}')")
            mixed_vars[name] = pymoo_var
        return mixed_vars

    def set_kernel_input(self, kernel_inputs):
        """
        Set the input coordinate for the problem, to which the design parameters will be applied.

        Args:
            kernel_inputs (pd.DataFrame | pd.Series | dict): The input coordinates that will be used
                to build the model input. Can be a DataFrame, Series, or dictionary.
        """
        self.kernel_inputs = kernel_inputs
        if isinstance(self.kernel_inputs, pd.DataFrame):
            self.input_columns = kernel_inputs.columns
        elif isinstance(self.kernel_inputs, pd.Series):
            self.input_columns = self.kernel_inputs.index
        elif isinstance(self.kernel_inputs, dict):
            # If the input is dict, we can directly build a Dataframe from it
            self.input_columns = None

    def _build_model_input(self, x):
        """
        Build model input by combining design parameters with kernel inputs.

        Args:
            x (numpy.ndarray): Array of design parameter values from the optimizer.

        Returns:
            pd.DataFrame: Combined DataFrame containing both design parameters and
                kernel inputs, tiled to match the number of samples.
        """
        # Pymoo returns a ndarray of dict
        # We must convert it to a simple list for pandas to automatically builds the DataFrame
        x = list(x)
        sample_count = len(x)
        # Create a dataframe containing the samples
        model_input = pd.DataFrame(x)

        # Tile the user inputs to match the number of samples
        repeats_user_inputs = pd.DataFrame(np.tile(self.kernel_inputs, (sample_count, 1)), columns=self.input_columns)

        # Concat both dataframes
        model_input = pd.concat([model_input, repeats_user_inputs], axis=1)

        return model_input

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objectives of the given population.

        Args:
            x (numpy.ndarray): A 2D array of shape (n_features, n_samples) containing
                the design parameter values for each sample.
            out (dict): Dictionary to store the evaluation results for each objective.
            *args: Extra positional arguments, unused in this implementation.
            **kwargs: Extra keyword arguments, unused in this implementation.
        """

        # Ensure the user has set the kernel inputs
        if self.kernel_inputs is None:
            raise Exception("Kernel inputs were not set prior to calling evaluate !")

        model_inputs = self._build_model_input(x)

        predictions = []

        # Build all the predictions
        for obj in self.objectives:
            prediction = self.surrogate_models[obj.name].predict(model_inputs)

            # If one of the objective is a mazimization objective, then reverse it
            if obj.direction == "maximize":
                prediction *= -1
            predictions.append(prediction)

        if len(self.objectives) == 1:
            out["F"] = predictions[0]
        elif len(self.objectives) == 2:
            out["F"] = np.column_stack(predictions)
        else:
            raise Exception(f"Unsupported number of objectives ({len(self.objectives)})")


class _GeneticOptimizationMethod:
    """
    Base class for all genetic optimization methods.
    Optimization methods apply an optimization algorithm for one input point, and differ
    in the algorithm used/ logic for selecting the final solutions
    """

    def run(self, kernel_input: pd.Series):
        raise NotImplementedError()


class _NormalizedOptimizationMethod(_GeneticOptimizationMethod):
    """
    Optimization method that uses a normalized weighted sum of the objectives to select the best solution.
    """

    def __init__(
        self,
        *,
        objectives: list,
        parameters: dict,
        input_names: list,
        optimization_parameters: dict,
        termination_criterion,
        normalization_coefficients: dict,
        surrogate_models: dict,
    ):
        """
        Initialize the normalized optimization method.

        Args:
            objectives (list): List of objective function names.
            parameters (dict): Dictionary of parameter values for sampling.
            input_names (list): List of input parameter names.
            optimization_parameters (dict): Additional optimization parameters.
            termination_criterion: Termination criterion for optimization.
            normalization_coefficients (dict): Dictionary mapping objective names to their
                normalization coefficients.
            surrogate_models (dict): Dictionary of surrogate models for each objective.
        """
        self.objectives = objectives
        self.parameters = parameters
        self.input_names = input_names
        self.optimization_parameters = optimization_parameters
        self.termination_criterion = termination_criterion
        self.normalization_coefficients = normalization_coefficients
        self.surrogate_models = surrogate_models

    def _normalized_selection(self, raw_parameters, raw_objectives):
        """
        Parse a list of results and select the one with the best normalized weighted sum.

        The weights correspond to user-defined coefficients for each objective.

        Args:
            raw_parameters: A list of encoded kernel design parameters.
            raw_objectives: A list of corresponding objective values.

        Returns:
            tuple: A tuple containing:
                - optimal_parameters: The optimal design parameters.
                - optimal_values: The corresponding objective values.
        """

        # FIXME: We should generalize this to any number of objectives
        if len(self.objectives) != 2:
            raise Exception("Normalized selection requires exactly 2 objectives !")

        # Normalize the objectives, and find the best parameter set

        normalized_objectives = raw_objectives.copy()
        # Normalize both objectives
        normalized_objectives[:, 0] = np.interp(
            raw_objectives[:, 0],
            (raw_objectives[:, 0].min(), raw_objectives[:, 0].max()),
            (0, 1),
        )

        normalized_objectives[:, 1] = np.interp(
            raw_objectives[:, 1],
            (raw_objectives[:, 1].min(), raw_objectives[:, 1].max()),
            (0, 1),
        )

        coefficients = self.normalization_coefficients
        first_coefficient = list(coefficients.values())[0]
        second_coefficient = list(coefficients.values())[1]

        # Find the parameter set with the best normalized weighted sum
        best_normalized_index = np.argmin(
            first_coefficient * normalized_objectives[:, 0] + second_coefficient * normalized_objectives[:, 1]
        )

        optimal_parameters = pd.Series(
            raw_parameters[best_normalized_index], index=[k for k in self.parameters if k not in self.input_names]
        )

        # We return the raw objectives, not the normalized ones
        optimal_values = raw_objectives[best_normalized_index]

        return optimal_parameters, optimal_values

    def _run_nsga2(self, problem) -> pymoo.core.result.Result:
        """
        Run the NSGA2 algorithm on the given problem.

        Extracts required parameters from the configuration.

        Args:
            problem: The problem to run the algorithm on.

        Returns:
            pymoo.core.result.Result: The return value of the NSGA2 algorithm.
        """

        # Create the algorithm object
        # NSGA2 For mixed variables
        algorithm = NSGA2(
            **self.optimization_parameters,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        # Run the algorithm
        res = minimize(problem, algorithm, termination=self.termination_criterion, seed=1)
        return res

    def run(self, kernel_input):
        """
        Run the NSGA2 algorithm and select the kernel design parameters using user-defined coefficients.

        Args:
            kernel_input: The kernel inputs corresponding to the local sampling point.

        Returns:
            tuple: A tuple containing:
                - optimal_parameters: The optimal design parameter.
                - optimal_objectives_values: The corresponding objective values.
        """
        # FIXME: Normalized optimization can support as many as objectives as the user wants
        if len(self.objectives) != 2:
            raise Exception("Normalized optimization currently only supports 2D optimization problems")

        problem = DesignParametersProblem(
            objectives=self.objectives,
            parameters=self.parameters,
            input_names=self.input_names,
            surrogate_models=self.surrogate_models,
        )
        problem.set_kernel_input(kernel_input)

        res = self._run_nsga2(problem)
        optimal_parameters, optimal_objectives_values = self._normalized_selection(res.X, res.F)

        return optimal_parameters, optimal_objectives_values


class _MonoObjectiveOptimizationMethod(_GeneticOptimizationMethod):
    """
    An optimization method for mono-objective problems, where the best solution is selected
    based on the minimum objective value.
    """

    def __init__(
        self,
        *,
        objectives: list,
        parameters: dict,
        input_names: list,
        optimization_parameters: dict,
        termination_criterion,
        do_early_stopping: bool,
        output_directory: pathlib.Path,
        surrogate_models: dict,
        record_history: bool = True,
    ):
        """
        Initialize the mono-objective optimization method.

        Args:
            objectives (list): List of objective function names.
            parameters (dict): Dictionary of parameter values for sampling.
            input_names (list): List of input parameter names.
            optimization_parameters (dict): Additional optimization parameters.
            termination_criterion: Termination criterion for optimization.
            do_early_stopping (bool): Whether to enable early stopping.
            output_directory (pathlib.Path): Directory where output files will be saved.
            surrogate_models (dict): Dictionary of surrogate models for each objective.
            record_history (bool, optional): Whether to record optimization history. Defaults to True.
        """
        self.objectives = objectives
        self.parameters = parameters
        self.input_names = input_names
        self.optimization_parameters = optimization_parameters
        self.termination_criterion = termination_criterion
        self.do_early_stopping = do_early_stopping
        self.output_directory = output_directory
        self.surrogate_models = surrogate_models
        self.do_record = record_history

        if do_early_stopping:
            criterion = self._build_early_stopping_criterion(surrogate_models)
            self.termination = TerminationCollection(termination_criterion, criterion)
        else:
            self.termination = termination_criterion

    def _build_early_stopping_criterion(self, surrogate_models) -> RobustTermination:
        """
        Build a stopping criterion with a heuristic for the convergence threshold.

        Executes 10k random solutions and takes a fraction of the minimum value as a threshold.

        Args:
            surrogate_models (dict): The models to compute the threshold with.

        Returns:
            RobustTermination: A convergence stopping criterion.
        """

        begin = time.time()

        sampler = RandomSampler(variables=self.parameters)

        samples = sampler.sample(10000)

        predictions = None
        for m in surrogate_models.values():
            pred = m.predict(samples)
            if predictions is None:
                predictions = pred
            else:
                np.column_stack([pred, predictions])

        # Take the nearest power of 10 below the minimum prediction in absolute value
        epsilon = 1e-10  # Small value to avoid log10(0)
        thresh = 10 ** (np.floor(np.log10(np.min(abs(predictions)) + epsilon)) - 1)

        end = time.time()

        logging.info(f"Early stopping enabled, threshold inferred to be {thresh} (Overhead: {np.round(end - begin, 3)}s)")
        return RobustTermination(MultiObjectiveSpaceTermination(tol=thresh, n_skip=5), period=20)

    def run(self, kernel_input: pd.Series):
        """
        Optimized kernel optimization for single objective experiments.

        Args:
            kernel_input (pd.Series): The kernel inputs corresponding to the local sampling point.

        Returns:
            tuple: A tuple containing:
                - X: The optimal design parameters.
                - F: The corresponding objective values.
        """

        if len(self.objectives) != 1:
            raise Exception("Mono objective optimization was used with multiples objectives")

        problem = DesignParametersProblem(
            objectives=self.objectives,
            parameters=self.parameters,
            input_names=self.input_names,
            surrogate_models=self.surrogate_models,
        )
        problem.set_kernel_input(kernel_input)

        algorithm = MixedVariableGA(**self.optimization_parameters)

        res = minimize(
            problem,
            algorithm,
            termination=self.termination,
            seed=1,
            save_history=self.do_record,
        )

        if self.do_record:
            self._record_history(res, kernel_input)

        best_configuration = pd.concat([pd.Series(res.X), kernel_input])

        optimal_index = np.argmin(res.F)
        return best_configuration, res.F[optimal_index]

    def _record_history(self, ga_res, kernel_input):
        """
        Record the optimization history for convergence analysis.

        Args:
            ga_res: The genetic algorithm result object containing the optimization history.
            kernel_input: The kernel input parameters for the current optimization run.
        """
        history = ga_res.history

        dbs = []

        input_df = pd.DataFrame([kernel_input], columns=kernel_input.index).reset_index(drop=True)

        # Record the best solution at each iteration
        for iteration, hist in enumerate(history):
            best_sol_index = np.argmin(hist.pop.get("F"))

            # Build a DataFrame containing the best solution
            db = pd.DataFrame([hist.pop.get("X")[best_sol_index]])
            db["performance"] = hist.pop.get("F")[best_sol_index]
            db["iteration"] = iteration

            dbs.append(pd.concat((db, input_df), axis=1))

        db = pd.concat(dbs, axis=0).reset_index(drop=True)

        # Check for existing records to append to
        output_path = self.output_directory / "ga_convergence_study/records.csv"
        if output_path.exists():
            db_old = pd.read_csv(output_path)
            db = pd.concat([db_old, db], axis=0).reset_index(drop=True)
        else:
            os.makedirs(output_path.parent, exist_ok=True)

        db.to_csv(output_path, index=False)


class GeneticOptimizer(object):
    """
    A genetic optimizer that creates a list of samples inside the sampling space and finds the
    local optimal design parameters, gathered as a dataframe.
    """

    def __init__(
        self,
        *,
        objectives: list,
        parameters: dict,
        input_names: list,
        optimization_parameters: dict,
        termination_criterion,
        normalization_coefficients: dict,
        selection_method: str,
        do_early_stopping: bool,
        output_directory: pathlib.Path,
        sampler,
        samples_count: int,
        surrogate_models: dict,
        optimizer_checkpoint: OptimizerCheckpoint,
    ):
        """
        Construct a new genetic optimizer with explicit parameters.

        Args:
            objectives (list): List of objective function names.
            parameters (dict): Dictionary of parameter values for sampling.
            input_names (list): List of input parameter names.
            optimization_parameters (dict): Additional optimization parameters.
            termination_criterion: Termination criterion for optimization.
            normalization_coefficients (dict): Dictionary mapping objective names to their
                normalization coefficients.
            selection_method (str): Selection method for optimization.
            do_early_stopping (bool): Whether to enable early stopping.
            output_directory (pathlib.Path): Directory where output files will be saved.
            sampler: The sampler to use to generate the optimization points.
            samples_count (int): The number of optimization points to generate.
            surrogate_models (dict): A dict of surrogates for each objective in the experiment,
                defined as {objective_name: surrogate_model}.
            optimizer_checkpoint (OptimizerCheckpoint): Checkpoint manager for optimization state.
        """
        self.objectives = objectives
        self.parameters = parameters
        self.input_names = input_names
        self.optimization_parameters = optimization_parameters
        self.termination_criterion = termination_criterion
        self.normalization_coefficients = normalization_coefficients
        self.selection_method = selection_method
        self.do_early_stopping = do_early_stopping
        self.output_directory = output_directory
        self.sampler = sampler
        self.samples_count = samples_count
        self.surrogate_models = surrogate_models
        self.optimizer_checkpoint = optimizer_checkpoint
        self.output_path = output_directory / "optim.csv"

    def _make_optimization_method(self):
        """
        Find the optimization method to use for the current configuration.

        Returns:
            _GeneticOptimizationMethod: An optimization method instance based on the configuration.
        """

        selection_method = self.selection_method
        objectives = self.objectives
        # If we only have one objective, we can use the mono-objective optimization method
        if len(objectives) == 1 or selection_method == "mono":
            return _MonoObjectiveOptimizationMethod(
                objectives=self.objectives,
                parameters=self.parameters,
                input_names=self.input_names,
                optimization_parameters=self.optimization_parameters,
                termination_criterion=self.termination_criterion,
                do_early_stopping=self.do_early_stopping,
                output_directory=self.output_directory,
                surrogate_models=self.surrogate_models,
            )
        elif self.selection_method == "normalized":
            return _NormalizedOptimizationMethod(
                objectives=self.objectives,
                parameters=self.parameters,
                input_names=self.input_names,
                optimization_parameters=self.optimization_parameters,
                termination_criterion=self.termination_criterion,
                normalization_coefficients=self.normalization_coefficients,
                surrogate_models=self.surrogate_models,
            )
        else:
            raise ValueError(f"Unknown selection method ('{selection_method}')")

    def _optimize_point(
        self,
        optimization_method: _GeneticOptimizationMethod,
        input_features: pd.Series,
    ):
        """
        Optimize design parameters for a single input point.

        Args:
            optimization_method (_GeneticOptimizationMethod): The optimization method to use.
            input_features (pd.Series): The input features for the optimization point.

        Returns:
            tuple: A tuple containing:
                - best_design_params: The best design parameters found.
                - objective_values: The corresponding objective values.
        """
        best_design_params, objective_values = optimization_method.run(input_features)
        return best_design_params, objective_values

    def _optimize_all_samples(self, optimization_method, optimization_points: pd.DataFrame):
        """
        Iterate over all the given samples, and find the best design parameters for each sample.

        Results are returned as a DataFrame.

        Args:
            optimization_method: The function to use to find the best design parameters for each sample.
            optimization_points (pd.DataFrame): A dataframe containing the samples to optimize,
                in the shape (nb_features, nb_samples).

        Returns:
            pd.DataFrame: A dataframe containing the results of the optimization, in the shape
                (nb_features, nb_samples).
        """

        # if we have saved results, load them and reduce the number of optimization points to process.
        results = self.optimizer_checkpoint.maybe_load_results(optimization_points)
        if results is not None:
            optimization_points = optimization_points.tail(len(optimization_points) - len(results))
        else:
            results = pd.DataFrame()

        # Run the optimization method on every point
        for _, user_inputs in tqdm(
            optimization_points.iterrows(),
            total=len(optimization_points),
            desc="Running optimization phase",
        ):
            best_config, _ = self._optimize_point(optimization_method, user_inputs)
            # results.append(best_config)
            best_config = pd.DataFrame([best_config])
            # Get parameter types for encoding
            parameters_type = _get_parameter_types(self.parameters)
            best_config = encode_dataframe(parameters_type, best_config)

            # log best_config to the checkpoint. Should we change here to process a batch, of configs?
            best_config = self.optimizer_checkpoint.save(best_config)
            results = pd.concat([results, best_config], ignore_index=True)

        return results

    def _optimize(self):
        """
        Run the optimization process on all sampled points.

        Returns:
            pd.DataFrame: Encoded DataFrame containing the best design parameters for each sample.
        """

        # First define the optimization points
        samples = self.sampler.sample(self.samples_count)

        # Build the optimization method
        optimization_method = self._make_optimization_method()

        # Run the optimization method on every point
        results = self._optimize_all_samples(optimization_method, samples)

        # Get parameter types for encoding
        parameters_type = _get_parameter_types(self.parameters)
        return encode_dataframe(parameters_type, results)

    def run(self):
        """
        Generate a grid of samples for the kernel inputs, and find the best parameters of every sample.

        Further down, those parameters can be clustered to form a map of the best design parameters
        for each kernel input.

        Returns:
            pd.DataFrame: A dataframe with the best design parameters for each kernel input.
        """

        results = self._optimize()

        # test the checkpoint is correct
        saved_results = pd.read_csv(self.output_path)
        self.optimizer_checkpoint.consistency_check(saved_results)
        return results
