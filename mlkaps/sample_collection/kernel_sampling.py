"""
Copyright (C) 2020-2024 Intel Corporation
Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
Copyright (C) 2024-  MLKAPS contributors
SPDX-License-Identifier: BSD-3-Clause
"""

import pandas as pd

from mlkaps.sampling.adaptive.ga_adaptive import GAAdaptiveSampler
from mlkaps.sampling.experiment import Objective
from mlkaps.sampling import ValueRange, ValueSequence

from .mono_kernel_executor import MonoKernelExecutor
from .function_harness import MonoFunctionHarness
from .failed_run_resolver import DiscardResolver
from .samples_checkpoint import SamplesCheckpoint


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
