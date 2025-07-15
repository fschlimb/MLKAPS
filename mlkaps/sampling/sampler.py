"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause

Define the base class for all samplers.
"""

import numpy as np
import math


class ValueContainer:
    """Base class for value containers.

    Value containers are used, for example, to store parameters and by samplers.
    """

    def __init__(self, type):
        """
        Initialize a value container.

        Args:
            type (type): The type of values contained (int, float, str, bool).
        """
        assert type in [int, float, str, bool], f"Unknown type: {type}"
        self.type = type

    def split(self, threshold):
        """
        Split the container into two containers.

        The first container contains all values less than or equal to the threshold.
        The second container contains all values greater than the threshold.

        Args:
            threshold (float | int): The threshold value for splitting.

        Returns:
            tuple: Two containers after splitting.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def sample_linear_space(self, n_samples=-1):
        """
        Return a list of n_samples many values from the container.

        If n_samples is greater than the number of values in the container, return all values in the container.

        Args:
            n_samples (int): Number of samples to return.

        Returns:
            list | np.ndarray: The sampled values.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def is_continuous(self):
        """
        Return True if the container holds continuous data, False otherwise.

        Sets and Sequences for example are not continuous while a range is.

        Returns:
            bool: True if continuous, False otherwise.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_size(self):
        """
        Return the size of the container.

        Returns:
            int | float: The size of the container.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_sampling_bounds(self):
        """
        Return the lower and upper bound of the container.

        Returns:
            list: The lower and upper bounds as [min, max].
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def map_to_numeric(self, data):
        """
        Map the data to float values.

        Args:
            data (np.ndarray | list): Data to map.

        Returns:
            np.ndarray: The mapped numeric data.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def map_from_numeric(self, indices):
        """
        Map numeric representation to values of the container.

        Args:
            indices (np.ndarray | list): Numeric indices to map from.

        Returns:
            np.ndarray: The mapped values.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_dtype(self):
        """
        Convert a type string to a numpy dtype.

        Returns:
            str | type: The numpy dtype corresponding to the container's type.
        """
        if self.type == int:
            return "int"
        elif self.type == float:
            return "float"
        elif self.type == str:
            return str
        elif self.type == bool:
            return bool
        else:
            raise ValueError(f"Unknown variable type: {self.type}")


class ValueSet(ValueContainer):
    """Container for a set of values.

    The values in the set are not checked for anything.
    They are assumed to be valid and of the same type.
    """

    def __init__(self, values, type=float):
        """
        Initialize a value set.

        Args:
            values (list | np.ndarray): The set of values.
            type (type): The type of values.
        """
        # call the parent constructor
        super().__init__(type)
        self.values = np.sort(values)

    def split(self, threshold):
        """
        Split the sequence into two sequences.

        The first sequence contains the first threshold many elements.
        The second sequence contains all values starting with index >= threshold.

        Args:
            threshold (int): Index at which to split.

        Returns:
            tuple: Two ValueSet objects after splitting.
        """
        assert threshold >= 0 and threshold <= len(self.values), "Threshold must be in the index range of the set"
        threshold = round(threshold)
        return ValueSet(self.values[:threshold], self.type), ValueSet(self.values[threshold:], self.type)

    def is_continuous(self):
        """
        Check if the value set is continuous.

        Returns:
            bool: Always False since sets are discrete.
        """
        return False

    def get_size(self):
        """
        Return the size of the set.

        The size is defined by the number of elements in the set.

        Returns:
            int: The number of elements in the set.
        """
        return len(self.values)

    def get_sampling_bounds(self):
        """
        Return the lower and upper bounds of the set.

        The bounds of sets are defined by their index space.

        Returns:
            list: The bounds as [0, len(values) - 1].
        """
        return [0, len(self.values) - 1]

    def sample_linear_space(self, n_samples=-1):
        """
        Return a list of n_samples many values from the set.

        If n_samples is greater than the number of values in the set, return all values in the set.

        Args:
            n_samples (int): Number of samples to return.

        Returns:
            np.ndarray | None: The sampled values, or None if n_samples is 0.
        """
        if n_samples == 0:
            return None
        if n_samples < 0:
            return self.values.copy()

        n = self.get_size()
        return self.map_from_numeric(np.linspace(0, n - 1, n_samples, endpoint=True))

    def map_to_numeric(self, data):
        """
        Map the data to float values.

        The data is expected to be an array of values from the set.
        The mapping is done by creating a map of the values to their index in the set.

        Args:
            data (np.ndarray | list): Data to map.

        Returns:
            np.ndarray: The mapped numeric indices.
        """
        data = data.copy()
        feature_map = {k: j for j, k in enumerate(self.values)}
        return np.vectorize(feature_map.get)(data)

    def map_from_numeric(self, indices):
        """
        Map numeric representation to values of the container.

        Get samples by picking the values at the given indices.
        The indices are expected to be in the range of the container.
        float indices are rounded to int. Non-numeric indices are not supported.

        Args:
            indices (np.ndarray | list): Numeric indices to map from.

        Returns:
            np.ndarray: The mapped values.
        """
        indices = np.asarray(indices, dtype="float").round().astype(np.intp)
        # creating the full sequence might not be optimal for sequences; can be improved if needed
        return self.values[indices]


class ValueSequence(ValueContainer):
    """Container for a sequence of numeric values.

    The sequence is defined by a start, stop and progression.
    It includes the start value and excludes the stop value.
    The progression mode can be "arithmetic" or "geometric".
    """

    def __init__(self, start, stop, progression, mode="arithmetic", type=float):
        """
        Initialize a value sequence.

        Args:
            start (int | float): Start value of the sequence.
            stop (int | float): Stop value of the sequence.
            progression (int | float): Step or ratio for the sequence.
            mode (str): Progression mode ("arithmetic" or "geometric").
            type (type): Type of the values.
        """
        # call the parent constructor
        super().__init__(type)
        if type not in [int, float]:
            raise ValueError(f"Unsupported type: {type}")
        if mode not in ["arithmetic", "geometric"]:
            raise ValueError(f"Unknown mode: {mode}")
        self.start = type(start)
        self.stop = type(stop)
        self.progression = type(progression)
        self.mode = mode
        assert mode != "geometric" or progression > 1, "Geometric progression must be greater than 1"

    def split(self, threshold):
        """
        Split the sequence into two sequences.

        - The first sequence contains the first elements in the sequence which are < threshold.
        - The second sequence contains all trailing values >= threshold.

        Args:
            threshold (int | float): Value at which to split.

        Returns:
            tuple: Two ValueSequence objects after splitting.
        """
        assert (
            threshold >= self.get_sampling_bounds()[0] and threshold <= self.get_sampling_bounds()[1] + 1
        ), "Threshold must be in the index range of the sequence"
        return (
            ValueSequence(self.start, threshold, self.progression, self.mode, self.type),
            ValueSequence(threshold, self.stop, self.progression, self.mode, self.type),
        )

    def is_continuous(self):
        """
        Check if the value sequence is continuous.

        Returns:
            bool: Always False since sequences are discrete.
        """
        return False

    def get_size(self):
        """
        Return the size of the sequence.

        The size is defined by the number of elements in the sequence.

        Returns:
            int: The number of elements in the sequence.
        """
        if self.mode == "arithmetic":
            return int((self.stop - self.start + self.progression - 1) // self.progression)
        assert self.mode == "geometric"
        eps = 1e-12  # np.finfo(np.float32).eps
        return int(math.log((self.stop * self.progression - eps) / self.start, self.progression))

    def get_sampling_bounds(self):
        """
        Return the lower and upper bounds of the sequence.

        Returns:
            list: The bounds as [start, last].
        """
        if self.mode == "arithmetic":
            last = self.start + self.progression * (self.get_size() - 1)
        else:
            last = self.start * (self.progression ** (self.get_size() - 1))
        return [self.start, last]

    def sample_linear_space(self, n_samples=-1):
        """
        Return a list of n_samples many values from the sequence.

        If n_samples is greater than the number of values in the sequence, return all values in the sequence.

        Args:
            n_samples (int): Number of samples to return.

        Returns:
            np.ndarray | list | None: The sampled values, or None if n_samples is 0.
        """
        if n_samples == 0:
            return None
        if n_samples == 1:
            return [self.start]

        n = self.get_size()
        if n_samples < 0:
            n_samples = n
        indices = np.round(np.linspace(0, n - 1, n_samples, endpoint=True)).astype("int")
        if self.mode == "arithmetic":

            def gen():
                for x in range(n_samples):
                    yield self.start + indices[x] * self.progression

        else:
            # geometric
            def gen():
                for x in range(n_samples):
                    yield self.start * (self.progression ** indices[x])

        return np.fromiter(gen(), dtype=self.get_dtype(), count=n_samples)

    def map_to_numeric(self, data):
        """
        Map the data to float values.

        The data is expected to be an array of values from the set.
        The mapping is done by creating a map of the values to their index in the set.

        Args:
            data (np.ndarray | list): Data to map.

        Returns:
            np.ndarray: The mapped numeric data.
        """
        if self.type == int:
            return np.round(data).astype("int")
        return data

    def map_from_numeric(self, data):
        """
        Map numeric representation to values of the container.

        "Quantize" input data to values in the sequence defined by start, stop and progression.

        Args:
            data (np.ndarray | list): Numeric data to map from.

        Returns:
            np.ndarray: The mapped sequence values.
        """
        # for each element in data, find the clostest element in the sequence defined by start, stop and progression
        if self.mode == "arithmetic":
            data = np.clip(data, self.start, self.stop)
            data = np.round((data - self.start) / self.progression).astype("int")
            data = data * self.progression + self.start
        else:
            # geometric
            data = np.clip(data, *self.get_sampling_bounds())
            for i in range(len(data)):
                if data[i] > self.start:
                    # is there a more elegant way to do this?
                    pos = math.log(data[i] / self.start, self.progression)
                    low = self.start * (self.progression ** math.floor(pos))
                    high = self.start * (self.progression ** math.ceil(pos))
                    data[i] = low if data[i] - low < high - data[i] else high
        return data.astype(self.get_dtype())


class ValueRange(ValueContainer):
    """Container for a range of values.

    The range is defined by a start and stop value.
    The range is inclusive of the start value. The stop value can be either inclusive or exclusive.
    The range is always defined by floats.
    """

    def __init__(self, start, stop, include_high_bound=True, type=float):
        """
        Initialize a value range.

        Args:
            start (float): Start value of the range.
            stop (float): Stop value of the range.
            include_high_bound (bool): Whether to include the upper bound.
            type (type): Type of the values (should be float).
        """
        # call the parent constructor
        super().__init__(type)
        assert type == float, "Only float type is supported for continuous ranges"
        assert start <= stop, "Start must be <= stop"
        self.start = type(start)
        self.stop = type(stop)
        self.include_high_bound = include_high_bound

    def split(self, threshold):
        """
        Split the range into two ranges.

        The first range contains all values less than or equal to the threshold.
        The second range contains all values greater than the threshold.

        Args:
            threshold (float): Value at which to split.

        Returns:
            tuple: Two ValueRange objects after splitting.
        """
        # is it ok that this will have threshold in both ranges?
        assert threshold >= self.start and threshold <= self.stop, "threshold must be within the range"
        return (
            ValueRange(self.start, threshold, self.include_high_bound, self.type),
            ValueRange(threshold, self.stop, self.include_high_bound, self.type),
        )

    def is_continuous(self):
        """
        Check if the value range is continuous.

        Returns:
            bool: Always True since ranges are continuous.
        """
        return True

    def get_size(self):
        """
        Return the size of the range.

        The size is defined by the distance between the start and stop values.

        Returns:
            float: The size of the range.
        """
        return self.stop - self.start

    def get_sampling_bounds(self):
        """
        Return the lower and upper bounds of the range.

        Returns:
            list: The bounds as [start, stop].
        """
        assert self.include_high_bound, "Upper bound must be inclusive"
        return [self.start, self.stop]

    def sample_linear_space(self, n_samples):
        """
        Return a list of n_samples many values from the range.

        If n_samples is greater than the number of values in the range, return all values in the range.

        Args:
            n_samples (int): Number of samples to return.

        Returns:
            np.ndarray | list: The sampled values.
        """
        assert n_samples >= 0, "Cannot sample full continuous space, n_samples must be >= 0"
        if n_samples == 0:
            return []
        if n_samples == 1:
            return [self.start]

        dtype = self.get_dtype()
        return np.linspace(self.start, self.stop, num=n_samples, endpoint=self.include_high_bound, dtype=dtype)

    def map_to_numeric(self, data):
        """
        Map the data to float values.

        Nothing to be done here, the data is already in the correct format.

        Args:
            data (np.ndarray | list): Data to map.

        Returns:
            np.ndarray: The mapped numeric data.
        """
        return data.astype(self.get_dtype())

    def map_from_numeric(self, data):
        """
        Map numeric representation to values of the container.

        Nothing to be done here, the data is already in the correct format.

        Args:
            data (np.ndarray | list): Numeric data to map from.

        Returns:
            np.ndarray: The mapped values.
        """
        return np.asarray(data, dtype=self.get_dtype())


def _mask_variables(variables: dict, mask: list) -> dict:
    """
    Helper function to filter out variables that are not in the mask.

    Args:
        variables (dict): The variables to filter.
        mask (list): The list of variables to keep.

    Returns:
        dict: Masked dictionary.
    """
    if mask is None:
        return variables
    return {key: value for key, value in variables.items() if key in mask}


class SamplerError(Exception):
    """
    Generic exception to raise when a sampler fails
    """


class Sampler:
    """
    Base class for all samplers.
    """

    def __init__(self, variables=None):
        """
        Initialize the sampler.

        Args:
            variables (dict | None): A dictionary associating the name of each variable to its ValueContainer.
        """
        self.variables = None

        # Set the variables using a setter to ensure that overriding classes can perform
        # additional checks if needed
        self.set_variables(variables)

    def _raise_if_variables_not_set(self):
        """
        Raise an exception if the variables are not set or if the variables are empty.

        Raises:
            SamplerError: If the variables are not set or if the variables are empty.
        """

        if self.variables is None:
            raise SamplerError("The sampler variables (values and/or types) were not set!")
        if len(self.variables) == 0:
            raise SamplerError("The passed variables were empty!")

    def set_variables(self, variables: dict):
        """
        Set the variables to be sampled.

        Args:
            variables (dict): A dictionary associating the name of each variable to its ValueContainer.
        """

        self.variables = variables
