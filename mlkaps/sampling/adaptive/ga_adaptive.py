"""
Copyright (C) 2020-2025 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause

Definition of the GA-Adaptive sampling process, based on genetic algorithms
"""

import pathlib
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableDuplicateElimination,
    MixedVariableSampling,
)
from pymoo.optimize import minimize
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination import get_termination

from tqdm import tqdm
from typing import Callable

from mlkaps.modeling import OptunaTunerLightgbm, SurrogateFactory
from . import HVSampler
from .. import SamplerError
from .. import RandomSampler, LhsSampler
import logging
import time
from mlkaps.sample_collection.samples_checkpoint import SamplesCheckpoint

from mlkaps.modeling.encoding import encode_dataframe
from mlkaps.sampling.experiment import Objective
from mlkaps.optimization.genetic_optimizer import DesignParametersProblem


class GAAdaptiveSampler:
    """Adaptive sampler based on genetic algorithms.

    The sampler uses a genetic algorithm to pick interesting points in the design space,
    and combines this with a HVS sampler to explore the design space.
    """

    def __init__(
        self,
        *,
        # Required parameters - no sensible defaults
        execution_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_directory: pathlib.Path,
        samples_checkpoint: SamplesCheckpoint,
        n_samples: int,
        objectives: list[Objective] | Objective,
        parameters: dict,
        input_names: list[str] | None = None,
        # Required GA parameters with sensible defaults
        samples_per_iteration: int,
        bootstrap_ratio: float = 0.2,
        initial_ga_ratio: float = 0.2,
        final_ga_ratio: float = 0.8,
        # Optional parameters with defaults
        do_early_stopping: bool = True,
        use_optuna: bool = False,
    ):
        """Initialize the GA-Adaptive sampler.

        Args:
            execution_function (Callable[[pd.DataFrame], pd.DataFrame]): A callback for the execution
                samples that will evaluates the samples.
            output_directory (pathlib.Path): Directory where output files will be saved.
            samples_checkpoint (SamplesCheckpoint): Checkpoint handler for samples.
            n_samples (int): The total number of samples to take.
            objectives (list[Objective] | Objective): List of objective function names or single objective.
            parameters (dict): Dictionary mapping parameter names to their ValueContainers.
            input_names (list[str] | None, optional): List of names of the input parameters. Defaults to None.
            samples_per_iteration (int): The number of samples to take per iterations of the GA loop.
            bootstrap_ratio (float, optional): The ratio (value between 0-1) of the total number of samples to take
                in the bootstrapping phase. Defaults to 0.2.
            initial_ga_ratio (float, optional): The ratio (value between 0-1) of points taken with the GA Algorithm
                at the first iteration of the algorithm. The ratio at iteration x is the linear interpolation
                between this value and final_ga_ratio. Defaults to 0.2.
            final_ga_ratio (float, optional): The ratio of points taken with the GA Algorithm
                at the last iteration of the algorithm. See initial_ga_ratio. Defaults to 0.8.
            do_early_stopping (bool, optional): Whether to enable early stopping in GA. Defaults to True.
            use_optuna (bool, optional): Whether to use Optuna for hyperparameter tuning. Defaults to False.
        """

        # Store configuration parameters directly
        self.objectives = objectives if isinstance(objectives, list) else [objectives]
        self.parameters = parameters
        self.input_names = input_names

        self.execution_function = execution_function

        self.samples_checkpoint = samples_checkpoint

        self.n_samples = n_samples
        self.samples_per_iteration = int(samples_per_iteration)

        self.bootstrap_ratio = bootstrap_ratio
        self.initial_ga_ratio = initial_ga_ratio
        self.final_ga_ratio = final_ga_ratio

        self.do_early_stopping = do_early_stopping
        self.use_optuna = use_optuna
        self._verify_input()

        # HVS sampler for exploration
        self.hvs_sampler = HVSampler(
            variables=self.parameters,
            error_metric="cov",
        )

        # FIXME: dirty quick-restart
        self.output_path = pathlib.Path(output_directory) / "kernel_sampling/samples.csv"

        self.models = {}
        self.iteration = 0

    def _verify_input(self):
        """Ensure that the parameters used to build this model are valid.

        Raises:
            SamplerError: If the sampler was built with incorrect parameters.
        """

        if self.samples_per_iteration > self.n_samples or self.samples_per_iteration < 1:
            raise SamplerError(
                f"samples_per_iteration({self.samples_per_iteration}) must be between 1 and n_samples({self.n_samples})"
            )

        if self.bootstrap_ratio > 1 or self.bootstrap_ratio < 0:
            raise SamplerError("bootstrap_ratio must be between 0 and 1")

        if self.initial_ga_ratio > 1 or self.initial_ga_ratio < 0:
            raise SamplerError("initial_ga_ratio must be between 0 and 1")

        if self.final_ga_ratio > 1 or self.final_ga_ratio < 0:
            raise SamplerError("final_ga_ratio must be between 0 and 1")

        if self.final_ga_ratio < self.initial_ga_ratio:
            raise SamplerError("final_ga_ratio must be greater than initial_ga_ratio")

    def run(self) -> pd.DataFrame:
        """Run the sampling process using the parameters used in the constructor.

        First bootstrap using LHS, then run the main sampling loop using a combination
        of LHS and genetic algorithm.

        Returns:
            pd.DataFrame: A list of samples and their respective values.

        Raises:
            SamplerError: If an error occurs during the sampling process.
        """

        with tqdm(total=self.n_samples, leave=None) as pbar:
            # FIXME: The execution function should be updated to have a proper interface for such cases
            # Attempt to set the progress bar on the execution function
            if hasattr(self.execution_function, "progress_bar"):
                self.execution_function.progress_bar = pbar

            try:
                logging.info("GA-Adaptive started")
                n_bootstrap = int(self.n_samples * self.bootstrap_ratio)
                samples = self.samples_checkpoint.maybe_load_samples()
                if samples is not None:
                    pbar.update(len(samples))
                    n_bootstrap = n_bootstrap - len(samples)

                logging.info("Bootstrapping with LHS")

                if n_bootstrap > 0:
                    lhs_samples = self._lhs_bootstrap(n_bootstrap, pbar)
                    samples = pd.concat([samples, lhs_samples])

                logging.info("Bootstrapping finished, starting GA-Adaptive loop")
                samples = self._resampling_loop(samples, pbar)
                return samples
            except Exception as exc:
                # Wrap any exception in a SamplerError
                print(f"Sampler failed with exception: {exc}")
                raise SamplerError("GA-Adaptive sampling failed!") from exc

    def _maybe_load_samples(self):
        """Attempt to load existing samples from output file for quick restart.

        Returns:
            pd.DataFrame | None: Previously saved samples DataFrame or None if file doesn't exist.
        """
        if not self.output_path.exists():
            return None

        logging.info(f"Found samples at '{self.output_path}', quick-restarting")
        logging.warning(
            f"GA-Adaptive will quick-restart by default, this is currently not configurable\n"
            f"Please delete '{self.output_path} to skip quick-restart."
        )
        loaded_samples = pd.read_csv(self.output_path)
        return loaded_samples

    def _lhs_bootstrap(self, n_samples, pbar):
        """Perform initial bootstrapping using Latin Hypercube Sampling.

        Args:
            n_samples (int): Number of bootstrap samples to generate.
            pbar (tqdm): Progress bar instance for tracking progress.

        Returns:
            pd.DataFrame | None: DataFrame with bootstrap samples and their evaluations.
        """
        if n_samples <= 0:
            return None

        # Bootstrap the sampling with an LHS
        pbar.set_description("GA-Adaptive: bootstrapping with LHS")

        sampler = LhsSampler(variables=self.parameters)
        lhs_samples = sampler.sample(n_samples)

        return self._sample_kernel(lhs_samples)

    def _resampling_loop(self, samples, pbar) -> pd.DataFrame:
        """Main adaptive sampling loop that iteratively selects points using GA and random sampling.

        Args:
            samples (pd.DataFrame): Initial samples (from bootstrap phase).
            pbar (tqdm): Progress bar instance for tracking progress.

        Returns:
            pd.DataFrame: Complete DataFrame with all samples and evaluations.
        """
        pbar.set_description("GA-Adaptive-Random")

        final_ratio_delta = self.final_ga_ratio - self.initial_ga_ratio
        while len(samples) < self.n_samples:
            # Ensure we don't overshoot the total number of samples
            leftover_samples = min(self.samples_per_iteration, self.n_samples - len(samples))

            # We want to start with a high ratio of HVS picked points, and gradually decrease it
            # in favor of the GA optimized points
            # However, we don't want the number of GA points to be 0 at the start, start with 20%
            curr_ratio = self.initial_ga_ratio + final_ratio_delta * (len(samples) / self.n_samples)

            n_ga_points = int(np.round(curr_ratio * leftover_samples))
            new_points = self._pick_ga_points(samples, n_ga_points)

            # We randomly pick the remaining points for the exploration component
            # Of the sampler
            if n_ga_points == 0:
                delta = leftover_samples
            else:
                delta = max(0, leftover_samples - len(new_points))
            # hvs_samples = self._pick_hvs_samples(delta, samples)

            sampler = RandomSampler(variables=self.parameters)
            random_samples = sampler.sample(delta)
            random_samples = pd.concat([samples, self._sample_kernel(random_samples)])

            # Concat GA points with random_samples
            ga_points = self._sample_kernel(new_points)
            samples = pd.concat(
                [
                    random_samples,
                    ga_points,
                ]
            )
            samples.reset_index(drop=True, inplace=True)
            self.samples_checkpoint.consistency_check(samples)

            self.iteration += 1

        return samples

    def _sample_kernel(self, new_points: pd.DataFrame) -> pd.DataFrame | None:
        """Execute the new_points using the execution function, returns None if no points was passed.

        Args:
            new_points (pd.DataFrame): New points to execute.

        Returns:
            pd.DataFrame | None: The list of samples decorated with their values.
                One should not expect this function return to match the inputs, as samples
                may have failed and have been removed from the resulting DataFrame.
        """

        if new_points is None or len(new_points) == 0:
            return None
        return self.execution_function(new_points)

    def _pick_random_optimization_points(self, n_points: int) -> pd.DataFrame:
        """Randomly pick new points to run the GA on.

        Args:
            n_points (int): The number of optimization points to select.

        Returns:
            pd.DataFrame: A list of optimization points.
        """

        sampler = RandomSampler(
            variables={k: v for k, v in self.parameters.items() if k in self.input_names},
        )
        return sampler.sample(n_points)

    def _pick_hvs_samples(self, n_samples: int, data: pd.DataFrame) -> pd.DataFrame:
        """Run the HVS subsampler to find new points for the next sampling iteration.

        Args:
            n_samples (int): The number of samples to take using HVS.
            data (pd.DataFrame): All the samples collected so far.

        Returns:
            pd.DataFrame: A list of samples taken with HVS.
        """
        new_samples = self.hvs_sampler.sample(n_samples, data, self.execution_function)
        return new_samples

    def _pick_ga_points(self, samples: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Compute new samples using GA.

        First, we randomly select new optimization points.
        We then build new surrogate models based on the current samples.
        We then run NSGA2 on those points, using the surrogate models as oracles.
        The samples returned are the optimal points founds by the GA.

        Args:
            samples (pd.DataFrame): A list of currently sampled points.
            n_samples (int): The number of samples to take using GA.

        Returns:
            pd.DataFrame: A list of new samples for evaluation.
        """

        if n_samples == 0:
            return None

        # First, pick new optimization points
        optimization_points = self._pick_random_optimization_points(n_samples)

        # print head of samples and dtype
        # print(samples.head())
        # print(samples.dtypes)
        # Fit models to the currently sampled points
        try:
            models = self._fit_models(samples)
        except ValueError as e:
            logging.error(f"Error fitting models: {e}")
            raise SamplerError("Failed to fit models due to a value error") from e

        # Create the GA object
        problem = DesignParametersProblem(
            objectives=self.objectives,
            parameters=self.parameters,
            input_names=self.input_names,
            surrogate_models=models,
        )
        algorithm = NSGA2(
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

        if self.do_early_stopping:
            termination = self._build_early_stopping_criterion(self.models)
            termination = TerminationCollection(termination, get_termination("time", "0:0:10"))
        else:
            termination = get_termination("time", "0:0:10")

        # Run GA on every optimization points
        sampling_list = []
        for _, point in tqdm(
            optimization_points.iterrows(),
            total=len(optimization_points),
            desc="Running Genetic Algorithm",
            leave=None,
        ):
            local_optimum, _ = self._run_ga_on_point(point, algorithm, problem, termination=termination)

            # Append the local solution to the list of points to be sampled
            sampling_list.append(local_optimum)

        # Aggregate all the results in a DataFrame
        sampling_list = pd.DataFrame(sampling_list)
        sampling_list = pd.concat([optimization_points, sampling_list], axis=1)

        return sampling_list

    def _build_early_stopping_criterion(self, surrogate_models) -> RobustTermination:
        """Build a stopping criterion with an heuristic for the convergence threshold.

        Execute 1M random solutions, and take a fraction of the minimum value as a threshold.

        Args:
            surrogate_models (dict): The models to compute the threshold with.

        Returns:
            RobustTermination: A convergence stopping criterion.
        """

        begin = time.time()

        sampler = RandomSampler(variables=self.parameters)

        samples = sampler.sample(1000000)

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

    def _run_ga_on_point(self, point, algorithm, problem, termination):
        """Execute the given genetic algorithm on one optimization point.

        Args:
            point (pandas.Series): The optimization point to run the GA on.
            algorithm (pymoo.algorithms.moo.nsga2.NSGA2): The genetic algorithm to execute.
            problem (mlkaps.optimization.genetic_optimizer.DesignParametersProblem): The pymoo problem
                corresponding to the optimization job.
            termination: The termination criterion for the algorithm.

        Returns:
            tuple: A tuple containing (optimal_configuration, point).
        """

        problem.set_kernel_input(point)

        # Run the minimization task with a short timeout to add some uncertainty
        local_optimum = minimize(problem, algorithm, termination=termination)
        if isinstance(local_optimum.X, dict):
            local_optimum = local_optimum.X
        else:
            # Randomly pick one of the solutions if there are multiple, so we avoid
            # having the same point multiple times, and it allows us to discover new
            # potential optimums
            local_optimum = local_optimum.X[np.random.choice(local_optimum.X.shape[0])]
        return local_optimum, point

    def _build_model(self, obj, samples):
        """Build a surrogate model for the given objective using the current samples.

        Args:
            obj (str): Name of the objective to build a model for.
            samples (pd.DataFrame): Current samples used for training the model.

        Returns:
            Any: Trained surrogate model.
        """

        if self.use_optuna:
            tuner = OptunaTunerLightgbm(samples.drop(self.objectives, axis=1), samples[obj])
            model, _ = tuner.run(time_budget=2 * 60, n_trials=128)
        else:
            print("Building oracle model for ga-adaptive step")

            factory = SurrogateFactory(
                samples, parameters=self.parameters, modeling_method="lightgbm", model_parameters={}, output_directory=None
            )
            model = factory.build(obj)
        return model

    def _fit_models(self, samples):
        """Create new LightGBM models and fit them to the current samples.

        Args:
            samples (pd.DataFrame): The list of samples to train the new models on.

        Returns:
            dict: A dictionary containing one model per objective.
        """

        for obj in self.objectives:
            objname = obj.name
            first_iter = objname not in self.models

            if first_iter or (self.iteration % 4) == 0:
                self.models[objname] = self._build_model(obj, samples)
            else:
                new_X, new_y = (
                    samples.drop([o.name for o in self.objectives], axis=1),
                    samples[objname],
                )
                new_X = encode_dataframe(self.parameters, new_X)
                self.models[objname].fit(new_X, new_y)

        return self.models
