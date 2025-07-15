"""
Copyright (C) 2020-2025 Intel Corporation
Copyright (C) 2022-2025 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pathlib
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import logging
import csv


class SamplesCheckpoint:
    def __init__(self, *, output_directory, parameters: dict, objectives: list):
        """
        Initialize the sample checkpoint for kernel sampling.

        :param output_directory: Path to the output directory where sample files will be stored.
            Can be a string or pathlib.Path.
        :type output_directory: str or pathlib.Path
        :param parameters: Dictionary containing kernel inputs and design parameters.
        :type parameters: dict
        :param objectives: List of objectives.
        :type objectives: list

        The output file will be created in the 'kernel_sampling' subdirectory of the given output directory.
        If the file exists, new samples will be appended; otherwise, it will be created upon first write.
        """
        # Do not create the file here.  If it does exist, we will append new samples.
        if isinstance(output_directory, str):
            output_directory = pathlib.Path(output_directory)

        self.output_directory = output_directory / "kernel_sampling"
        self.output_path = output_directory / "samples.csv"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.parameters = parameters
        self.column_names = list(parameters.keys()) + [x.name for x in objectives]  # sorted needed?
        self.restarted = False

    def delete_file(self):
        # Delete the checkpoint file and its directory
        if self.output_path.exists():
            self.output_path.unlink()  # Remove the file
        if self.output_directory.exists():
            self.output_directory.rmdir()  # Remove the directory (only if it's empty)

    def restart(self):
        # support quick restart
        logging.info("Quick-restart: Loading the samples from previous run")
        if not self.output_path.exists():
            raise FileNotFoundError(self.output_path)
        self.restarted = True
        restart_samples = pd.read_csv(self.output_path)
        return restart_samples

    def maybe_load_samples(self, verbose: bool = True):
        # load samples if they are there.  This is not for a top-level quick restart

        # do not reload if we loaded them for a requested restart
        if self.restarted:
            return None

        # no reload if the file does not exist.
        if not self.output_path.exists():
            return None

        if verbose:
            logging.info(f"Found samples at '{self.output_path}', quick-restarting")
            logging.warning(
                f"Sampling will quick-restart by default, this is currently not \
                configurable\nPlease delete '{self.output_path} to skip quick-restart."
            )
        loaded_samples = pd.read_csv(self.output_path)
        # check if the loaded_samples seems correct
        self._sanity_check(loaded_samples)

        return loaded_samples

    def _sanity_check(self, samples: pd.DataFrame):
        # check if the samples look OK
        try:
            # expected number of columns
            _, ncols = samples.shape
            assert ncols == len(self.column_names), f"samples have {len(self.column_names)} columns, expected {ncols}"

            # expected column names
            sample_names = list(samples.head(1))
            assert (
                sample_names == self.column_names
            ), f"Sample names {sample_names} do not match expected column names {self.column_names}"

            # expected types  (note we only have types for parameters, not for the objective)
            for pname, pval in self.parameters.items():
                sample_type = samples.dtypes[pname]
                expected_type = pval.get_dtype()
                assert self._compatible_types(
                    sample_type, expected_type
                ), f"Sample type {sample_type} of parameter {pname} is not compatible with {expected_type}"

            # check all the elements of each column have the same type
            for column in samples.columns:
                column_types = samples[column].map(type).unique()
                assert len(column_types) == 1, f"Column '{column}' contains multiple types: {column_types}"

            # Assert that there are no NaN values in the samples DataFrame
            assert not samples.isnull().values.any(), f"The samples from {self.output_path} DataFrame contains NaN values."

        except AssertionError as e:
            logging.warning(f"Saved samples at '{self.output_path}' do not seem correct for this experiment")
            logging.warning(str(e))
            raise e

    def _compatible_types(self, sample_type, expected_type):
        # sample_type is from the dataframe
        # expected_type is from ML-KAPS parameter
        if sample_type == "float64" and expected_type == "float":
            return True
        if sample_type == "int64" and expected_type == "int":
            return True
        if sample_type == "string" and expected_type == "categorical":
            return True
        if sample_type == "object" and expected_type == "categorical":
            return True
        if sample_type == "bool" and expected_type == "bool":
            return True
        return False

    def consistency_check(self, samples: pd.DataFrame):
        # check if samples.csv matches the samples in memory
        saved_samples = pd.read_csv(self.output_path)
        assert (
            samples.shape == saved_samples.shape
        ), f"Samples have shape {samples.shape} and saved_samples have shape {saved_samples.shape}"
        if len(samples) == 0:
            return
        try:
            assert_frame_equal(samples, saved_samples, rtol=1e-5)
        except AssertionError as e:
            raise e

    def save_batch(self, batch: pd.DataFrame):

        # Sort the samples DataFrame by column names
        _, ncols = batch.shape
        assert ncols == len(self.column_names), f"batch has {len(self.column_names)} columns, expected {ncols}"
        batch = batch[self.column_names]

        # Append batch to output file
        # We make sure that the column order is the same as what has already been
        # written to the csv file. We write the colum nheaders only when we create the file,
        # not for subsequent appends.
        # *** This should be the only place we write samples to the samples.csv file ***
        if self.output_path.exists() and self.output_path.is_file():
            with self.output_path.open("r") as file:
                csv_reader = csv.reader(file)
                column_headers = next(csv_reader)
                # batch = batch[column_headers]
                assert column_headers == self.column_names, "columns do not match"
                batch.to_csv(self.output_path, mode="a", header=False, index=False)
        else:
            # this is the first write to the .csv file
            batch.to_csv(self.output_path, mode="a", header=True, index=False)

        # Ensure data is flushed and written to disk
        # if this proves to be expensive, we can do it every 10th batch.
        with open(self.output_path, "a") as f:
            f.flush()
            os.fsync(f.fileno())
        return batch
