# Copyright 2025 Joe Worsham

import tensorflow as tf

from matplotlib import pyplot as plt
from typing import List, Tuple


def add(fig, axes, xt: tf.Tensor, tt: tf.Tensor, style: str=None, label: str=None):
    dim_vals = tf.unstack(xt, axis=-1)
    for dim_val, ax in zip(dim_vals, axes):
        if len(dim_val.shape) == 1:
            dim_val = dim_val[None, :]
        
        assert len(dim_val.shape) == 2,\
            "Time plotter expects a single trajectory or a single batch of trajectories!"

        ax.plot(tt.numpy(), dim_val[0, ...].numpy(), style, label=label)
        for traj_val in dim_val[1:, ...]:
            ax.plot(tt.numpy(), traj_val.numpy(), style)


def plot(xt: tf.Tensor, title: str, idx1: int, idx2: int, dims: List[str], figsize: Tuple[float, float]=None):
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=figsize if figsize is not None else (6, 6))
    
    plt.title(title)
    ax.set_xlabel(dims[idx1])
    ax.set_ylabel(dims[idx2])

    val1 = xt[..., idx1]
    val2 = xt[..., idx2]

    if len(val1.shape) == 1:
        val1 = val1[None, :]

    if len(val2.shape) == 1:
        val2 = val2[None, :]
        
    assert len(val1.shape) == 2 and len(val2.shape) == 2,\
        "Phase plotter expects a single trajectory or a single batch of trajectories!"
    
    for traj_val1, traj_val2, in zip(val1, val2):
        ax.plot(traj_val1, traj_val2)
