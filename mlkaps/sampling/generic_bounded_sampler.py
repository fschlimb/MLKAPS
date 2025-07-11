"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause

Contain the definition for static samplers that only need an array containing the bounds of each variable.
"""

import numpy as np
import pandas as pd
from smt.sampling_methods import LHS, Random

from .sampler import SamplerError
from .static_sampler import StaticSampler
from .variable_mapping import map_float_to_variables


def convert_variables_bounds_to_numeric(variables):
    """
    Convert a dictionary of variables bounds to a dictionary of numeric bounds:
    - Categorical variables are converted to [0, n_values-1]
    - For numeric variables, the bounds are set to [min, max] where min and max are the
      lowest and highest possible values for the variable.

    :param variables: A dictionary associating the name of each variable to its ValueContainer.
    :type variables: dict

    :return: A dictionary associating the name of each variable to its bounds.
    :rtype: dict
    """

    # Generate a list of bounds for each parameter
    bounds = {}
    for variable, values in variables.items():
        bounds[variable] = values.get_sampling_bounds()
    return bounds


class GenericBoundedSampler(StaticSampler):
    """
    A sampler that works with any sampling techniques based on an array of bounds.
    Conceived with smt.sampling_methods in mind (LHS, Random, etc.)
    """

    def __init__(
        self,
        *,
        generic_sampler_type,
        variables: dict | None = None,
    ):
        """
        Build a new generic sampler using the generic_sampler_type sampling method

        :param generic_sampler_type:
            A sampling object that uses smt API;
            - Must have a constructor with a xlimits argument, defining a list of list,
            corresponding to the bounds of each variable
            - Must have a __call__ method, returning 2d list of samples
        :type generic_sampler_type: type
        :param variables:
            A dictionary associating the name of each variable to its ValueContainer.
        :type variable_values: dict | None, optional
        """

        self.sampler_type = generic_sampler_type
        # Calling the parent constructor will call set_variables()
        # We must define the bounds to be empty before
        self.bounds = None
        super().__init__(variables)

    def _generate_bounds(self):
        """
        Generate the bounds of the sampling process

        :return: A dictionnary containing the bounds for each variables
        :rtype: dict(str, list)
        """

        if self.variables is None:
            return None
        return convert_variables_bounds_to_numeric(self.variables)

    def set_variables(self, variables):
        """
        Set the variables used in the sampling process.

        :param variables: Contain the possible ValueContainer for each variable.
        :type variables: dict
        """

        super().set_variables(variables)
        self.bounds = self._generate_bounds()

    def _generate_samples_from_bounds(self, n_samples: int):
        """
        Execute the sampler on the bounded variable space

        :raise SamplerError: raise an exception if the variables were not set before usage, or if the sampler failed

        :param n_samples: The number of samples to take
        :type n_samples: int

        :return: A list of samples
        :rtype: pandas.DataFrame
        """

        self._raise_if_variables_not_set()

        # Dict are not guaranteed to be ordered
        # This may cause an issue where the samples do not match the order of the variables
        # As a safety measure, we sort the bounds dict by keys to guarantee the
        # order
        ordered_key = sorted(self.bounds.keys())
        fixed_order_bounds = np.array([self.bounds[i] for i in ordered_key])

        try:
            sampler = self.sampler_type(xlimits=fixed_order_bounds)
            # The sampler returns a dict of array containing every value for each
            # parameter
            random_samples = sampler(n_samples)
            # Create a new dataframe to ensure ordering
            ordered_random_samples = pd.DataFrame(random_samples, columns=ordered_key)
        except Exception as exc:
            print(f"Sampler failed with exception: {exc}")
            raise SamplerError from exc

        return ordered_random_samples

    def sample(self, n_samples: int) -> pd.DataFrame | None:

        if n_samples == 0:
            return None

        if n_samples < 0:
            raise SamplerError(f"Cannot sample negative samples count ({n_samples}) !")

        random_samples = self._generate_samples_from_bounds(n_samples)

        # Map the generated samples with numeric features back to the original variables types
        translated_columns = map_float_to_variables(random_samples, self.variables)

        columns = sorted(self.bounds.keys())

        # Now that we translated each column, we can create the final dataframe
        translated_samples = pd.DataFrame(translated_columns, columns=columns)
        return translated_samples


class LhsSampler(GenericBoundedSampler):
    """
    Sampler based on Latin Hypercube Sampling.
    """

    def __init__(self, *, variables: dict | None = None):
        """
        Create a new LHS (Latin Hypercube Sampling) sampler.

        :param variable_types:
            A dictionary associating the name of each variable to its ValueContainer.
        :type variable_types: dict | None, optional
        """
        super().__init__(
            generic_sampler_type=LHS,
            variables=variables,
        )


class RandomSampler(GenericBoundedSampler):
    """
    Sampler based on random uniform sampling
    """

    def __init__(self, *, variables: dict | None = None):
        """
        Create a new Random uniform sampler.

        :param variable_types:
            A dictionary associating the name of each variable to its ValueContainer.
        :type variable_types: dict | None, optional
        """
        super().__init__(
            generic_sampler_type=Random,
            variables=variables,
        )
