"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pathlib
import pandas as pd
import logging
import pprint
import textwrap

from mlkaps.configuration import ExperimentConfig
from mlkaps.sampling.sampler_factory import SamplerFactory
from mlkaps.sampling.adaptive.ga_adaptive import GAAdaptiveSampler
from mlkaps.sampling.experiment import Objective
from mlkaps.sampling import ValueRange, ValueSequence
from mlkaps.sampling.adaptive import (
    StoppingCriterionFactory,
    AdaptiveSamplingOrchestrator,
)

from .mono_kernel_executor import MonoKernelExecutor
from .function_harness import MonoFunctionHarness, FunctionPath
from .subprocess_harness import MonoSubprocessHarness
from .failed_run_resolver import DiscardResolver, ConstantResolver
from .samples_checkpoint import SamplesCheckpoint


def _get_key_or_error(cdict: dict, key: str, error_msg: str | None = None, default=None, fatal=True):
    """
    Extract a key from a dictionary with error handling.

    :param cdict: The dictionary to extract the key from
    :type cdict: dict
    :param key: The key to extract
    :type key: str
    :param error_msg: Custom error message to use if key is missing
    :type error_msg: str | None
    :param default: Default value to return if key is missing and not fatal
    :type default: Any
    :param fatal: Whether to raise an exception if key is missing
    :type fatal: bool
    :return: The value associated with the key or the default value
    :rtype: Any
    :raises ValueError: If key is missing and fatal is True
    """
    if key not in cdict:
        error_msg = error_msg or f"Missing parameter '{key}'"
        res = default
        if fatal:
            msg = textwrap.indent(f"Dictionnary:\n{pprint.pformat(cdict)}\n", "\t=> ")
            msg = f"{error_msg}\n{msg}"
            raise ValueError(msg)
        else:
            logging.warning(error_msg)
    else:
        res = cdict[key]
    return res


def _make_path_absolute(path: pathlib.Path | str, reference: pathlib.Path) -> pathlib.Path:
    """
    Convert a path to absolute, using a reference path if the input path is relative.

    :param path: The path to convert to absolute
    :type path: pathlib.Path | str
    :param reference: The reference path to use for relative paths
    :type reference: pathlib.Path
    :return: The absolute path
    :rtype: pathlib.Path
    """
    path = pathlib.Path(path)

    if path.is_absolute():
        return path

    return reference / path


class _StaticSamplerInterfaceWrapper:
    """
    Sampler that uses a static spatial-sampler to generate a list of samples that will be executed
    in a one-shot manner.

    This wrapper provides a unified interface for static sampling methods like LHS (Latin Hypercube Sampling)
    and random sampling. It generates all samples at once and then executes them in batches.
    """

    def __init__(
        self,
        *,
        kernel_sampler,
        # Configuration parameters that were previously in config
        feature_values: dict,
        output_directory: pathlib.Path,
        # Parameters that were previously in config_dict
        sampler_type: str = "lhs",
        nsamples: int,
        sampler_parameters: dict = None,
        # Checkpoint functionality
        samples_checkpoint: SamplesCheckpoint = None,
    ):
        """
        Initialize the static sampler wrapper.

        :param kernel_sampler: Callable that executes kernel samples
        :type kernel_sampler: Callable
        :param feature_values: Dictionary of feature values for sampling
        :type feature_values: dict
        :param output_directory: Directory where output files will be saved
        :type output_directory: pathlib.Path
        :param sampler_type: Type of sampler to use (default: "lhs")
        :type sampler_type: str
        :param nsamples: Number of samples to generate
        :type nsamples: int
        :param sampler_parameters: Additional parameters for the sampler
        :type sampler_parameters: dict | None
        :param samples_checkpoint: Checkpoint handler for saving/loading samples
        :type samples_checkpoint: SamplesCheckpoint | None
        :raises ValueError: If kernel_sampler is not callable
        """
        if not callable(kernel_sampler):
            raise ValueError("The kernel sampler must be a callable object")

        self.kernel_sampler = kernel_sampler
        self.output_directory = output_directory
        self.feature_values = feature_values
        self.sampler_type = sampler_type
        self.nsamples = nsamples
        self.sampler_parameters = sampler_parameters or {}
        self.samples_checkpoint = samples_checkpoint

    def __call__(self) -> pd.DataFrame:
        """
        Execute the static sampling process.

        :return: DataFrame containing samples and their evaluation results
        :rtype: pd.DataFrame
        """
        sampler, n_samples = self._build_sampler()
        return self._sample(sampler, n_samples)

    def _build_sampler(self):
        """
        Build the underlying sampler instance using SamplerFactory.

        :return: Tuple of (sampler instance, number of samples)
        :rtype: tuple
        """

        # We need to create a minimal config-like object for SamplerFactory
        # This is a temporary solution until SamplerFactory is also refactored
        class ConfigProxy:
            def __init__(self, feature_values):
                self.feature_values = feature_values

        config_proxy = ConfigProxy(self.feature_values)
        sampler = SamplerFactory(config_proxy).from_config(self.sampler_type, self.sampler_parameters)

        if hasattr(sampler, "set_variables"):
            sampler.set_variables(self.feature_values)

        return sampler, self.nsamples

    def _sample(self, sampler, n_samples):
        """
        Execute the sampling process in batches and save results incrementally.
        Handles checkpoint loading/saving for resuming interrupted sampling.

        :param sampler: The sampler instance to use for generating samples
        :type sampler: Any
        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: DataFrame containing all samples and their evaluation results
        :rtype: pd.DataFrame
        """
        # Check for existing samples from checkpoint
        samples_reloaded = None
        if self.samples_checkpoint:
            samples_reloaded = self.samples_checkpoint.maybe_load_samples()
            if samples_reloaded is not None:
                n_samples = n_samples - len(samples_reloaded)

        if n_samples <= 0:
            return samples_reloaded

        new_samples = sampler.sample(n_samples)
        new_samples = self.kernel_sampler(new_samples)

        # Combine with reloaded samples if any
        if samples_reloaded is not None:
            results = pd.concat([samples_reloaded, new_samples])
            results.reset_index(drop=True, inplace=True)
            return results
        else:
            return new_samples


class _GAAdaptiveInterfaceWrapper:
    """
    Interface wrapper for GA (Genetic Algorithm) adaptive sampling.

    This wrapper provides a unified interface for the GA-based adaptive sampling method.
    It uses genetic algorithms to intelligently select sampling points based on previous
    evaluations, combining exploration and exploitation strategies.
    """

    def __init__(
        self,
        *,
        kernel_sampler,
        # Configuration parameters that were previously in config
        output_directory: pathlib.Path,
        objectives: list,
        feature_values: dict,
        input_parameters: list,
        # Required GA parameters
        n_samples: int,
        bootstrap_ratio: float,
        initial_ga_ratio: float,
        final_ga_ratio: float,
        # Optional GA parameters with defaults
        n_iterations: int = None,
        samples_per_iteration: int = None,
        do_early_stopping: bool = True,
        use_optuna: bool = False,
        # Checkpoint functionality
        samples_checkpoint: SamplesCheckpoint = None,
    ):
        """
        Initialize the GA adaptive sampler wrapper.

        :param kernel_sampler: Callable that executes kernel samples
        :type kernel_sampler: Callable
        :param output_directory: Directory where output files will be saved
        :type output_directory: pathlib.Path
        :param objectives: List of objective function names to optimize
        :type objectives: list
        :param feature_values: Dictionary of feature values for sampling
        :type feature_values: dict
        :param input_parameters: List of input parameter names
        :type input_parameters: list
        :param n_samples: Total number of samples to generate
        :type n_samples: int
        :param bootstrap_ratio: Ratio of samples used for initial bootstrapping
        :type bootstrap_ratio: float
        :param initial_ga_ratio: Initial ratio of GA-selected vs random samples
        :type initial_ga_ratio: float
        :param final_ga_ratio: Final ratio of GA-selected vs random samples
        :type final_ga_ratio: float
        :param n_iterations: Number of GA iterations (optional)
        :type n_iterations: int | None
        :param samples_per_iteration: Number of samples per iteration (optional)
        :type samples_per_iteration: int | None
        :param do_early_stopping: Whether to enable early stopping in GA
        :type do_early_stopping: bool
        :param use_optuna: Whether to use Optuna for hyperparameter tuning
        :type use_optuna: bool
        :param samples_checkpoint: Checkpoint handler for saving/loading samples
        :type samples_checkpoint: SamplesCheckpoint | None
        :raises ValueError: If kernel_sampler is not callable
        """
        if not callable(kernel_sampler):
            raise ValueError("The kernel sampler must be a callable object")

        self.kernel_sampler = kernel_sampler
        self.output_directory = output_directory
        self.objectives = objectives
        self.feature_values = feature_values
        self.input_parameters = input_parameters
        self.n_samples = n_samples
        self.bootstrap_ratio = bootstrap_ratio
        self.initial_ga_ratio = initial_ga_ratio
        self.final_ga_ratio = final_ga_ratio
        self.n_iterations = n_iterations
        self.samples_per_iteration = samples_per_iteration
        self.do_early_stopping = do_early_stopping
        self.use_optuna = use_optuna
        self.samples_checkpoint = samples_checkpoint

    def __call__(self) -> pd.DataFrame:
        """
        Execute the GA adaptive sampling process.

        :return: DataFrame containing samples and their evaluation results
        :rtype: pd.DataFrame
        """
        sampler = self._build_sampler()
        return sampler()

    def _build_sampler(self):
        """
        Build the GA adaptive sampler instance with calculated parameters.

        :return: Configured GAAdaptiveSampler instance
        :rtype: GAAdaptiveSampler
        """
        samples_per_iteration = self._get_samples_per_iter(self.n_samples, self.bootstrap_ratio)

        sampler = GAAdaptiveSampler(
            execution_function=self.kernel_sampler,
            n_samples=self.n_samples,
            samples_per_iteration=samples_per_iteration,
            bootstrap_ratio=self.bootstrap_ratio,
            initial_ga_ratio=self.initial_ga_ratio,
            final_ga_ratio=self.final_ga_ratio,
            output_directory=self.output_directory,
            objectives=self.objectives,
            feature_values=self.feature_values,
            input_parameters=self.input_parameters,
            do_early_stopping=self.do_early_stopping,
            use_optuna=self.use_optuna,
            samples_checkpoint=self.samples_checkpoint,
        )
        return sampler

    def _get_samples_per_iter(self, n_samples, bootstrap_ratio):
        """
        Calculate the number of samples to take per GA iteration.

        :param n_samples: Total number of samples
        :type n_samples: int
        :param bootstrap_ratio: Ratio of samples used for bootstrapping
        :type bootstrap_ratio: float
        :return: Number of samples per iteration
        :rtype: int
        :raises ValueError: If neither n_iterations nor samples_per_iteration is provided
        """
        # Compute the number of samples taken with GA at each iteration
        # Done either via a direct fixed number of samples or a fixed number of iterations
        if all([self.n_iterations is None, self.samples_per_iteration is None]):
            raise ValueError("Either 'n_iterations' or 'samples_per_iteration' must be provided")
        elif self.n_iterations is not None:
            # if a number of iteration is given, compute the number of samples per iteration
            # by dividing the number of samples by the number of iterations
            samples_per_iteration = n_samples * (1 - bootstrap_ratio) / self.n_iterations
        elif self.samples_per_iteration is None:
            raise ValueError("Neither 'n_iterations' or 'samples_per_iteration' were defined")
        else:
            samples_per_iteration = self.samples_per_iteration
        return samples_per_iteration


class _AdaptiveSamplerInterfaceWrapper:
    """
    Interface wrapper for adaptive sampling methods (HVS, multi-level HVS).

    This wrapper provides a unified interface for adaptive sampling methods that use
    orchestrators and stopping criteria to iteratively refine the sampling process
    based on accumulated data.
    """

    def __init__(
        self,
        *,
        kernel_sampler,
        # Configuration parameters that were previously in config
        feature_values: dict,
        input_parameters: list,
        design_parameters: list,
        output_directory: pathlib.Path,
        # Parameters from config_dict
        sampler_type: str = "hvs",
        stopping_criteria: dict = None,
        orchestrator_parameters: dict = None,
        method_parameters: dict = None,
        # Checkpoint functionality
        samples_checkpoint: SamplesCheckpoint = None,
    ):
        """
        Initialize the adaptive sampler wrapper.

        :param kernel_sampler: Callable that executes kernel samples
        :type kernel_sampler: Callable
        :param feature_values: Dictionary of feature values for sampling
        :type feature_values: dict
        :param input_parameters: List of input parameter names
        :type input_parameters: list
        :param design_parameters: List of design parameter names
        :type design_parameters: list
        :param output_directory: Directory where output files will be saved
        :type output_directory: pathlib.Path
        :param sampler_type: Type of adaptive sampler to use (default: "hvs")
        :type sampler_type: str
        :param stopping_criteria: Dictionary defining stopping criteria (optional)
        :type stopping_criteria: dict | None
        :param orchestrator_parameters: Parameters for the sampling orchestrator (optional)
        :type orchestrator_parameters: dict | None
        :param method_parameters: Parameters for the sampling method (optional)
        :type method_parameters: dict | None
        :param samples_checkpoint: Checkpoint handler for saving/loading samples
        :type samples_checkpoint: SamplesCheckpoint | None
        :raises ValueError: If kernel_sampler is not callable
        """
        if not callable(kernel_sampler):
            raise ValueError("The kernel sampler must be a callable object")

        self.kernel_sampler = kernel_sampler
        self.feature_values = feature_values
        self.input_parameters = input_parameters
        self.design_parameters = design_parameters
        self.output_directory = output_directory
        self.sampler_type = sampler_type
        self.stopping_criteria = stopping_criteria
        self.orchestrator_parameters = orchestrator_parameters or {}
        self.method_parameters = method_parameters or {}
        self.samples_checkpoint = samples_checkpoint

    def __call__(self) -> pd.DataFrame:
        """
        Execute the adaptive sampling process using orchestrator.

        :return: DataFrame containing samples and their evaluation results
        :rtype: pd.DataFrame
        """
        orchestrator = self._build_sampler()
        samples = orchestrator.run()
        samples.reset_index(drop=True, inplace=True)
        return samples

    def _build_sampler(self):
        """
        Build the adaptive sampling orchestrator with configured sampler and stopping criteria.

        :return: Configured AdaptiveSamplingOrchestrator instance
        :rtype: AdaptiveSamplingOrchestrator
        """
        stopping_criteria = None
        if self.stopping_criteria:
            stopping_criteria = StoppingCriterionFactory.create_all_from_dict(self.stopping_criteria)

        # We need to create a minimal config-like object for SamplerFactory
        class ConfigProxy:
            def __init__(self, feature_values):
                self.feature_values = feature_values

        config_proxy = ConfigProxy(self.self.feature_values)
        sampler = SamplerFactory(config_proxy).from_config(self.sampler_type, self.method_parameters)

        if hasattr(sampler, "set_variables"):
            sampler.set_variables(self.self.feature_values)

        if hasattr(sampler, "set_per_level_features"):
            levels = [self.input_parameters, self.design_parameters]
            sampler.set_per_level_features(levels)

        orchestrator = AdaptiveSamplingOrchestrator(
            features=self.feature_values,
            execution_function=self.kernel_sampler,
            adaptive_sampler=sampler,
            output_directory=self.output_directory,
            samples_checkpoint=self.samples_checkpoint,
            stopping_criteria=stopping_criteria,
            **self.orchestrator_parameters,
        )

        return orchestrator


class ExecutorFactory:
    """
    Factory class for creating kernel execution instances.

    This factory creates configured MonoKernelExecutor instances with appropriate
    runners (function or subprocess) and failure resolution strategies.
    """

    def __init__(
        self,
        *,
        # Configuration parameters that were previously in config
        objectives: list,
        objectives_bounds: dict = None,
        working_directory: pathlib.Path,
        # Parameters from config_dict
        runner: str,
        runner_parameters: dict = None,
        failure_resolver: str = "discard",
        failure_resolver_parameters: dict = None,
        # Checkpoint functionality
        samples_checkpoint: SamplesCheckpoint = None,
    ):
        """
        Initialize the executor factory.

        :param objectives: List of objective function names
        :type objectives: list
        :param objectives_bounds: Dictionary defining bounds for objectives (optional)
        :type objectives_bounds: dict | None
        :param working_directory: Working directory for relative path resolution
        :type working_directory: pathlib.Path
        :param runner: Type of runner to use ("function" or "executable")
        :type runner: str
        :param runner_parameters: Parameters for the runner (optional)
        :type runner_parameters: dict | None
        :param failure_resolver: Type of failure resolver ("discard" or "constant")
        :type failure_resolver: str
        :param failure_resolver_parameters: Parameters for failure resolver (optional)
        :type failure_resolver_parameters: dict | None
        :param samples_checkpoint: Checkpoint handler for saving/loading samples
        :type samples_checkpoint: SamplesCheckpoint | None
        """
        self.objectives = objectives
        self.objectives_bounds = objectives_bounds
        self.working_directory = working_directory
        self.runner = runner
        self.runner_parameters = runner_parameters or {}
        self.failure_resolver = failure_resolver
        self.failure_resolver_parameters = failure_resolver_parameters or {}
        self.samples_checkpoint = samples_checkpoint

    def __call__(self):
        """
        Create and return a configured MonoKernelExecutor instance.

        :return: Configured kernel executor
        :rtype: MonoKernelExecutor
        """
        failure_resolver = self._build_failure_resolver()
        runner = self._build_runner()

        kernel_sampler = MonoKernelExecutor(runner, failure_resolver, self.samples_checkpoint)
        return kernel_sampler

    def _build_runner(self):
        """
        Build the appropriate runner instance based on runner type.

        :return: Configured runner instance
        :rtype: MonoFunctionHarness | MonoSubprocessHarness
        :raises ValueError: If runner type is unknown
        """
        mapping = {
            "function": self._build_function_runner,
            "executable": self._build_subprocess_runner,
        }

        if self.runner not in mapping:
            raise ValueError(f"Unknown runner type '{self.runner}'")

        return mapping[self.runner](self.runner_parameters)

    def _get_timeout(self, param):
        """
        Extract timeout parameter from configuration with default handling.

        :param param: Parameter dictionary containing timeout configuration
        :type param: dict
        :return: Timeout value in seconds or None for no timeout
        :rtype: int | None
        """
        timeout = _get_key_or_error(
            param,
            "timeout",
            "No timeout defined for the kernel sampling module, defaulting to 30s\n"
            "Set 'timeout' to 'None' to disable timeout.",
            default=30,
            fatal=False,
        )

        if timeout == "None":
            timeout = None

        return timeout

    def _build_subprocess_runner(self, param):
        """
        Build a subprocess runner for executable-based kernels.

        :param param: Parameters for the subprocess runner
        :type param: dict
        :return: Configured subprocess runner
        :rtype: MonoSubprocessHarness
        """
        kernel = _get_key_or_error(param, "kernel", "Subprocess runner requires a 'kernel' parameter")
        kernel = _make_path_absolute(kernel, self.working_directory)
        timeout = self._get_timeout(param)
        parameters_order = _get_key_or_error(
            param,
            "parameters_order",
            "Subprocess runner requires a 'parameters_order' parameter",
        )

        res = MonoSubprocessHarness(self.objectives, self.objectives_bounds, kernel, parameters_order, timeout)
        return res

    def _build_function_runner(self, param):
        """
        Build a function runner for Python function-based kernels.

        :param param: Parameters for the function runner
        :type param: dict
        :return: Configured function runner
        :rtype: MonoFunctionHarness
        """
        function = _get_key_or_error(param, "function", "Function runner requires a 'function' parameter")

        function = FunctionPath(function)
        # If the path is relative, then reference it to the working directory
        if function.is_source() and function.is_relative():
            function.path = _make_path_absolute(function.path, self.working_directory)

        timeout = self._get_timeout(param)

        res = MonoFunctionHarness(function, self.objectives, timeout)
        return res

    def _build_failure_resolver(self):
        """
        Build the appropriate failure resolver instance.

        :return: Configured failure resolver instance
        :rtype: DiscardResolver | ConstantResolver
        :raises ValueError: If failure resolver type is unknown
        """
        mapping = {
            "discard": DiscardResolver,
            "constant": ConstantResolver,
        }

        if self.failure_resolver not in mapping:
            raise ValueError(f"Unknown failure resolver type '{self.failure_resolver}'")

        return mapping[self.failure_resolver](**self.failure_resolver_parameters)


class SamplingSystemFactory:
    """
    Factory class for creating complete sampling system instances.

    This factory creates configured sampling system instances that combine
    kernel executors with appropriate sampling strategies (static, GA adaptive,
    or other adaptive methods).
    """

    def __init__(
        self,
        *,
        # Configuration parameters that were previously in config
        output_directory: pathlib.Path,
        objectives: list,
        objectives_bounds: dict = None,
        working_directory: pathlib.Path,
        feature_values: dict = None,
        input_parameters: list = None,
        design_parameters: list = None,
        # Parameters from config_dict["SAMPLING"]
        sampler: str,
        runner: str,
        runner_parameters: dict = None,
        failure_resolver: str = "discard",
        failure_resolver_parameters: dict = None,
        sampler_parameters: dict = None,
        # Checkpoint functionality
        samples_checkpoint: SamplesCheckpoint = None,
    ):
        """
        Initialize the sampling system factory.

        :param output_directory: Directory where output files will be saved
        :type output_directory: pathlib.Path
        :param objectives: List of objective function names
        :type objectives: list
        :param objectives_bounds: Dictionary defining bounds for objectives (optional)
        :type objectives_bounds: dict | None
        :param working_directory: Working directory for relative path resolution
        :type working_directory: pathlib.Path
        :param feature_values: Dictionary of feature values for sampling (optional)
        :type feature_values: dict | None
        :param input_parameters: List of input parameter names (optional)
        :type input_parameters: list | None
        :param design_parameters: List of design parameter names (optional)
        :type design_parameters: list | None
        :param sampler: Type of sampler to use
        :type sampler: str
        :param runner: Type of runner to use ("function" or "executable")
        :type runner: str
        :param runner_parameters: Parameters for the runner (optional)
        :type runner_parameters: dict | None
        :param failure_resolver: Type of failure resolver ("discard" or "constant")
        :type failure_resolver: str
        :param failure_resolver_parameters: Parameters for failure resolver (optional)
        :type failure_resolver_parameters: dict | None
        :param sampler_parameters: Parameters for the sampler (optional)
        :type sampler_parameters: dict | None
        :param samples_checkpoint: Checkpoint handler for saving/loading samples
        :type samples_checkpoint: SamplesCheckpoint | None
        """
        self.output_directory = output_directory
        self.objectives = objectives
        self.objectives_bounds = objectives_bounds
        self.working_directory = working_directory
        self.feature_values = feature_values
        self.input_parameters = input_parameters
        self.design_parameters = design_parameters
        self.sampler = sampler
        self.runner = runner
        self.runner_parameters = runner_parameters or {}
        self.failure_resolver = failure_resolver
        self.failure_resolver_parameters = failure_resolver_parameters or {}
        self.sampler_parameters = sampler_parameters or {}
        self.samples_checkpoint = samples_checkpoint

    def __call__(self):
        """
        Create and return a configured sampling system instance.

        :return: Configured sampling wrapper instance
        :rtype: _StaticSamplerInterfaceWrapper | _GAAdaptiveInterfaceWrapper | _AdaptiveSamplerInterfaceWrapper
        """
        kernel_sampler = ExecutorFactory(
            objectives=self.objectives,
            objectives_bounds=self.objectives_bounds,
            working_directory=self.working_directory,
            runner=self.runner,
            runner_parameters=self.runner_parameters,
            failure_resolver=self.failure_resolver,
            failure_resolver_parameters=self.failure_resolver_parameters,
            samples_checkpoint=self.samples_checkpoint,
        )()
        sampler = self._build_sampler(kernel_sampler)

        return sampler

    def _build_sampler(self, kernel_sampler):
        """
        Build the appropriate sampler wrapper based on sampler type.

        :param kernel_sampler: Configured kernel executor instance
        :type kernel_sampler: MonoKernelExecutor
        :return: Configured sampler wrapper instance
        :rtype: _StaticSamplerInterfaceWrapper | _GAAdaptiveInterfaceWrapper | _AdaptiveSamplerInterfaceWrapper
        :raises ValueError: If sampler type is unknown
        """
        # Create wrapper with appropriate parameters based on sampler type
        if self.sampler == "ga_adaptive":
            # Extract required GA parameters from sampler_parameters
            return _GAAdaptiveInterfaceWrapper(
                kernel_sampler=kernel_sampler,
                output_directory=self.output_directory,
                objectives=self.objectives,
                feature_values=self.feature_values,
                input_parameters=self.input_parameters,
                samples_checkpoint=self.samples_checkpoint,
                **self.sampler_parameters,
            )
        elif self.sampler == ("hvs", "multilevel_hvs"):
            # Adaptive sampler parameters
            return _AdaptiveSamplerInterfaceWrapper(
                kernel_sampler=kernel_sampler,
                feature_values=self.feature_values,
                input_parameters=self.input_parameters,
                design_parameters=self.design_parameters,
                output_directory=self.output_directory,
                sampler_type=self.sampler,
                samples_checkpoint=self.samples_checkpoint,
                **self.sampler_parameters,
            )
        elif self.sampler == ("lhs", "random"):
            # Static sampler parameters
            return _StaticSamplerInterfaceWrapper(
                kernel_sampler=kernel_sampler,
                feature_values=self.feature_values,
                output_directory=self.output_directory,
                sampler_type=self.sampler,
                samples_checkpoint=self.samples_checkpoint,
                **self.sampler_parameters,
            )
        else:
            raise ValueError(f"Unknown sampler type '{self.sampler}'")


def _build_kernel_sampler(config: ExperimentConfig, config_dict: dict, samples_checkpoint: SamplesCheckpoint):
    """
    Build an appropriate sampled depending on whether the sampling method is one-shot or adaptive

    Parameters
    ----------
    config
        The configuration object of the kernel sampling module
    config_dict
        The configuration dictionary
    samples_checkpoint
        The samples checkpoint handler

    Returns
    -------
    sampler:
        An appropriate sampler for use with the sampling method
    """

    sampling_config = config_dict["SAMPLING"]

    sampler_factory = SamplingSystemFactory(
        # Extract from config
        output_directory=config.output_directory,
        objectives=config.objectives,
        objectives_bounds=getattr(config, "objectives_bounds", None),
        working_directory=config.working_directory,
        feature_values=getattr(config, "feature_values", None),
        input_parameters=getattr(config, "input_parameters", None),
        design_parameters=getattr(config, "design_parameters", None),
        # Extract from sampling_config
        sampler=sampling_config["sampler"],
        runner=sampling_config["runner"],
        runner_parameters=sampling_config.get("runner_parameters", {}),
        failure_resolver=sampling_config.get("failure_resolver", "discard"),
        failure_resolver_parameters=sampling_config.get("failure_resolver_parameters", {}),
        sampler_parameters=sampling_config.get("sampler_parameters", {}),
        # Add checkpoint support
        samples_checkpoint=samples_checkpoint,
    )
    sampler = sampler_factory()

    return sampler


def sample_kernel() -> pd.DataFrame:
    # config: ExperimentConfig, config_dict: dict, samples_checkpoint: SamplesCheckpoint
    """
    Run the kernel sampling module on the user kernel

    Parameters
    ----------
    config
        A configuration object for the kernel sampling to execute
    config_dict
        The configuration dictionary
    samples_checkpoint
        The samples checkpoint handler

    Returns
    -------
    res:
        A labelled dataset of sampled points,  The dataset is also logged in the kernel_sample csv file.

    """
    # sampler = _build_kernel_sampler(config, config_dict, samples_checkpoint)
    # res = sampler()
    # samples_checkpoint.consistency_check(res)
    # return res

    def kernel(args: dict):
        return {"performance": 4711}

    inputs = {"input1": ValueRange(0, 10), "input2": ValueSequence(10, 100, 10, type=int)}
    parameters = {"a": ValueRange(0, 10), "b": ValueRange(10, 100)}
    all_params = {**inputs, **parameters}
    objective = Objective("performance", "maximize", 800)

    resolver = DiscardResolver()
    out_dir = "test_output_dir/"
    checkpoint = SamplesCheckpoint(output_directory=out_dir, parameters=all_params, objectives=[objective])
    runner = MonoFunctionHarness(function=kernel, objectives=[objective], timeout=77)
    executor = MonoKernelExecutor(runner=runner, resolver=resolver, samples_checkpoint=checkpoint)

    sampler = GAAdaptiveSampler(
        execution_function=executor,
        objectives=objective,
        parameters=all_params,
        input_names=inputs.keys(),
        n_samples=30,
        samples_per_iteration=10,
        output_directory=out_dir,
        samples_checkpoint=checkpoint,
        # ...,
    )

    samples = sampler.run()
    print(f"Generated samples:\n{samples}")
