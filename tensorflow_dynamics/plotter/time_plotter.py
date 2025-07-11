# Copyright 2025 Joe Worsham

import tensorflow as tf

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from typing import List, Tuple


def create(title: str, dims: List[str], nrows: int, ncols: int, figsize: Tuple[float, float]=None):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize if figsize is not None else (6, 6))
    axes = [item for sublist in axes for item in sublist]

    for ax, dim in zip(axes, dims):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(dim)

    return fig, axes

def add(axes, xt: tf.Tensor, tt: tf.Tensor, style: str='-', label: str=None):
    dim_vals = tf.unstack(xt, axis=-1)
    for dim_val, ax in zip(dim_vals, axes):
        if len(dim_val.shape) == 1:
            dim_val = dim_val[None, :]
        
        assert len(dim_val.shape) == 2,\
            "Time plotter expects a single trajectory or a single batch of trajectories!"

        ax.plot(tt.numpy(), dim_val[0, ...].numpy(), style, label=label)
        for traj_val in dim_val[1:, ...]:
            ax.plot(tt.numpy(), traj_val.numpy(), style)

        if label is not None:
            ax.legend()


def add_range(axes, xt: tf.Tensor, tt: tf.Tensor, xt_std: tf.Tensor, color: str="gray", alpha: float=0.4, label: str=None):
    dim_vals = tf.unstack(xt, axis=-1)
    dim_stds = tf.unstack(xt_std, axis=-1)
    for dim_val, dim_std, ax in zip(dim_vals, dim_stds, axes):
        if len(dim_val.shape) == 1:
            dim_val = dim_val[None, :]
        
        assert len(dim_val.shape) == 2,\
            "Time plotter expects a single trajectory or a single batch of trajectories!"

        dv = dim_val[0, ...].numpy()
        dstd = dim_std[0, ...].numpy()
        ax.fill_between(tt.numpy(), dv - dstd, dv + dstd, color=color, alpha=alpha, label=label)
        for traj_val, traj_std in zip(dim_val[1:, ...], dim_std[1:, ...]):
            dv = traj_val.numpy()
            dstd = traj_std.numpy()
            ax.fill_between(tt.numpy(), dv - dstd, dv + dstd, color=color, alpha=alpha)

        if label is not None:
            ax.legend()


def plot(xt: tf.Tensor, tt: tf.Tensor, title: str, dims: List[str], nrows: int, ncols: int,
         figsize: Tuple[float, float]=None):
    fig, axes = create(title, dims, nrows, ncols, figsize)
    
    dim_vals = tf.unstack(xt, axis=-1)
    for dim_val, ax, dim in zip(dim_vals, axes, dims):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(dim)

        if len(dim_val.shape) == 1:
            dim_val = dim_val[None, :]
        
        assert len(dim_val.shape) == 2,\
            "Time plotter expects a single trajectory or a single batch of trajectories!"

        for traj_val in dim_val:
            ax.plot(tt, traj_val)
                                 

