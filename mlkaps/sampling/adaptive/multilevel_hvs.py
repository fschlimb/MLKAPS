"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

from collections.abc import Callable

import pandas as pd

from .adaptive_sampler import AdaptiveSampler
from .hvs import HVSampler


class MultilevelHVS(AdaptiveSampler):
    """
    Multilevel version of the Hierarchical Variance Sampling (HVS) Algorithm.

    This sampler recursively partitions the data based on multiple levels of features.
    This is equivalent to decision tree partitioning with constraints on the splitting criterion
    ordering.

    When the final level is reached, the HVS algorithm is used to sample the data based on
    the final partitions.
    """

    def __init__(
        self,
        *,
        features_levels: list[list] = None,
        variables: dict | None = None,
    ):
        """
        Create a new MultilevelHVS sampler.

        Args:
            features_levels (list[list] | None, optional): A list of lists defining the hierarchical
                feature levels for multilevel partitioning. Each inner list contains feature names
                for that partitioning level. If None, must be set later using set_per_level_features
                method. Defaults to None.
            variables (dict | None, optional): A dictionary containing the ValueContainers of the
                variables to sample. The keys must be the name of the variables.
        """
        self.hvs = HVSampler(variables=variables, error_metric="cov")

        super().__init__(variables)
        self.features_levels = features_levels
        self.partitions = []

    def reset(self):
        """
        Reset the sampler state.

        This method is currently not implemented and returns without performing any action.
        """
        return

    def dump(self, output_directory):
        """
        Save the sampler state to a directory.

        This method is currently not implemented and returns without performing any action.

        Args:
            output_directory: The directory where the sampler state should be saved.
        """
        return

    def set_per_level_features(self, leveled_features: list[list]):
        """
        Set the hierarchical feature levels for multilevel partitioning.

        Args:
            leveled_features (list[list]): A list of lists defining the hierarchical
                feature levels for multilevel partitioning. Each inner list contains
                feature names for that partitioning level.
        """
        self.features_levels = leveled_features

    def set_variables(self, variables, mask=None):
        """
        Set the variables to be sampled.

        Args:
            variables: A dictionary containing the ValueContainers of the variables to sample.
            mask (optional): A mask to filter the variables. Defaults to None.
        """
        super().set_variables(variables, mask)
        self.hvs.set_variables(variables, mask)

    def sample(
        self,
        n_samples: int,
        data: pd.DataFrame | None,
        execution_func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Sample new data points using the multilevel HVS algorithm.

        This method performs multilevel partitioning of the data based on the configured
        feature levels and samples new points using the HVS algorithm.

        Args:
            n_samples (int): The number of samples to generate.
            data (pd.DataFrame | None): The existing data to use for partitioning. If None
                or empty, bootstrap sampling is performed.
            execution_func (Callable[[pd.DataFrame], pd.DataFrame]): A function that takes
                a DataFrame of samples and returns a DataFrame with evaluated objectives.

        Returns:
            pd.DataFrame: The updated data containing both existing and new samples.

        Raises:
            Exception: If features_levels is not set.
        """

        if self.features_levels is None:
            raise Exception("Features levels not set")

        # Handle empty data (bootstrap)
        if data is None or len(data) == 0:
            return self.hvs.sample(n_samples, data, execution_func)
        # The objectives are the columns that are not labelled as features
        objectives = [k for k in data.columns if k not in self.variables.keys()]
        n_samples_per_objective = max(1, n_samples // len(objectives))

        # We sample separately for each objective
        new_samples = None
        self.partitions = []
        for objective in objectives:
            # Partition the data for current objective
            objective_samples = self._partition(objective, data, self.features_levels, n_samples_per_objective)
            new_samples = pd.concat([new_samples, objective_samples], axis=0, ignore_index=True)

        # Run the execution function on the new samples
        labelled_samples = execution_func(new_samples)
        data = pd.concat([data, labelled_samples], axis=0, ignore_index=True)

        return data

    def _partition(
        self,
        objective,
        data: pd.DataFrame,
        leveled_features: list[list],
        n_samples: int,
        axes=None,
    ) -> pd.DataFrame:
        """
        Recursively partition the data based on hierarchical feature levels.

        This method applies multilevel partitioning by recursively processing each level
        of features and delegating to either final or intermediate partitioning methods.

        Args:
            objective: The objective column name to optimize during partitioning.
            data (pd.DataFrame): The data to partition.
            leveled_features (list[list]): The remaining feature levels to process.
            n_samples (int): The number of samples to generate.
            axes (optional): The axis limitations from previous levels. Defaults to None.

        Returns:
            pd.DataFrame: The sampled data points from the partitioning process.
        """

        # Run HVS based on the features in the current level
        features = {k: v for k, v in self.variables.items() if k in leveled_features[0]}
        next_features = leveled_features[1:]

        # Even if we're cutting on some different axis, we need to propagate all the axis
        # limitations coming from the previous levels
        # For example, if the current partition has A = [0, 2.5]
        # But we're cutting on B, we still need to respect the range for A
        if axes is None:
            axes = self.variables.copy()

        new_samples = None
        # We reached the last level, partition based on current features, and samples using HVS
        if len(next_features) == 0:
            new_samples = self._partition_final(axes, data, features, n_samples, objective)
        else:
            # We are not in the last level, partition based on current features, and apply the
            # next level on each partition
            new_samples = self._partition_intermediate(axes, data, features, n_samples, next_features, objective)

        return new_samples

    def _partition_final(self, axes, data, features, n_samples, objective):
        """
        Perform final partitioning when the last level is reached.

        This method applies HVS partitioning using the current features and samples
        from the resulting partitions.

        Args:
            axes: The axis limitations from previous levels.
            data: The data to partition.
            features: The features to use for partitioning.
            n_samples: The number of samples to generate.
            objective: The objective column name to optimize during partitioning.

        Returns:
            The sampled data points from the final partitions.
        """

        # First, partition using current features
        partitions, _ = self.hvs.partition(data, objective, n_samples, split_on=features, min_samples_per_leaf=15)

        # The current partitions are not necessarily covering all the axes limitations
        # We merge the axes limitations of previous levels with the current partitions
        self._merge_axes(axes, partitions)

        self.partitions.extend([p[1] for p in partitions])
        return self.hvs.sample_partitions(partitions)

    def _merge_axes(self, axes, partitions):
        """
        Merge axis limitations from previous levels with current partitions.

        This method ensures that partitions respect the axis limitations from previous
        levels by merging them with the current partition axes.

        Args:
            axes: The axis limitations from previous levels.
            partitions: The current partitions to merge axes with.
        """
        # Merge the axes defined in the partition with the axes defined in the previous levels
        for partition in partitions:
            partition = partition[1]
            for axis in axes:
                if axis not in partition.axes.keys():
                    partition.axes[axis] = axes[axis]

    def _partition_intermediate(self, axes, data, features, n_samples, next_features, objective):
        """
        Perform intermediate partitioning when more levels remain.

        This method applies HVS partitioning using the current features and then
        recursively applies the next level on each partition.

        Args:
            axes: The axis limitations from previous levels.
            data: The data to partition.
            features: The features to use for partitioning.
            n_samples: The number of samples to generate.
            next_features: The remaining feature levels to process.
            objective: The objective column name to optimize during partitioning.

        Returns:
            The sampled data points from all intermediate partitions.
        """
        # First, partition using current features
        # The minimum number of samples per leaf must be high enough so that we can split on the
        # next level

        partitions, _ = self.hvs.partition(data, objective, n_samples, split_on=features, min_samples_per_leaf=90)

        # The current partitions are not necessarily covering all the axes limitations
        # We merge the axes limitations of previous levels with the current partitions
        self._merge_axes(axes, partitions)

        res = None
        # Then, apply the next level on all partitions
        for p in partitions:

            n_samples = p[0]
            partition = p[1]

            # No need to continue if no samples is allocated to the partition, or if the
            # current partition is empty
            if n_samples == 0 or partition.samples is None or len(partition.samples) == 0:
                continue

            results = self._partition(
                objective,
                partition.samples,
                next_features,
                n_samples,
                axes=partition.axes,
            )
            res = pd.concat([res, results], axis=0, ignore_index=True)
        return res
