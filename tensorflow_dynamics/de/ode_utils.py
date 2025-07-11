# Copyright 2025 Joe Worsham
# based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
# and https://github.com/titu1994/tfdiffeq

import tensorflow as tf

from typing import Callable, Tuple


def compute_error_ratio(err_est: Tuple[tf.Tensor], y_0: Tuple[tf.Tensor],
                        y_1: Tuple[tf.Tensor], atol: tf.Tensor,
                        rtol: tf.Tensor) -> Tuple[tf.Tensor]:
    """Compute the mean-squared error ratio of the
    estimated error and the allowed error tolerance.

    Args:
        err_est (Tuple[tf.Tensor]): The estimated stage error. Shape: Tuple of [..., n].
        y_0 (Tuple[tf.Tensor]): The state at the beginning of this step. Shape: Tuple of [..., n].
        y_1 (Tuple[tf.Tensor]): The computed state at the end of this step. Shape: Tuple of [..., n].
        atol (tf.Tensor): The provided absolute error tolerance. Shape: [].
        rtol (tf.Tensor): The provided relative error tolerance. Shape: [].

    Returns:
        Tuple[tf.Tensor]: Error ratio for each batched entry. Shape: Tuple of [...].
    """
    ratios = []
    for err_est_n, y_0_n, y_1_n in zip(err_est, y_0, y_1):
        err_tol = atol + rtol * tf.maximum(tf.abs(y_0_n), tf.abs(y_1_n))  # -> [..., n]
        err_ratio = err_est_n / err_tol  # -> [..., n]
        ratios.append(tf.reduce_mean(err_ratio**2, axis=-1))
    return tuple(ratios)


def optimal_step_size(prev_step: tf.Tensor, mse_ratio: Tuple[tf.Tensor],
                      safety: float=0.9, ifactor: float=10.0,
                      dfactor: float=0.2, order: int=5) -> tf.Tensor:
    """Compute the optimal, next step size for an adaptive solver.
    Note: this is a batched operation and each batched trajectory
    will receive its own step.

    Args:
        prev_step (tf.Tensor): The previously applied step. Shape: [...].
        mse_ratio (Tuple[tf.Tensor]): Computed error ratio for each trajectory. Shape: Tuple of [...].
        safety (float, optional): Padding to apply to new scaling factor. Defaults to 0.9.
        ifactor (float, optional): 1/ifactor defines the lower bound step. Defaults to 10.0.
        dfactor (float, optional): 1/dfactor defines the upper bound step. Defaults to 0.2.
        order (int, optional): The order of the solution to take. Defaults to 5.

    Returns:
        tf.Tensor: The next, optimal step size for the solver. Shape: [].
    """
    new_steps = []
    for mse_ratio_n in mse_ratio:
        err_ratio = tf.sqrt(mse_ratio_n)
        exponent = tf.constant(1. / order, dtype=tf.float64)
        numerator = tf.ones_like(prev_step)
        dfactor = tf.where(mse_ratio_n < 1, numerator, numerator / dfactor)
        ifactor = numerator / ifactor
        new_factor = err_ratio ** exponent / safety
        factor = tf.maximum(ifactor, tf.minimum(new_factor, dfactor))
        new_steps.append(tf.reduce_min(prev_step / tf.where(factor == 0.0, ifactor, factor)))
    return tf.reduce_min(new_steps)  # []


def select_initial_step(func: Callable, y_0: Tuple[tf.Tensor],
                        t_0: tf.Tensor, order: tf.Tensor,
                        rtol: tf.Tensor, atol: tf.Tensor) -> tf.Tensor:
    """Empirically select a good initial step.

    The algorithm is described in [1].

    References:

    [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
    Equations I: Nonstiff Problems", Sec. II.4.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting value for this step. Shape: Tuple of [..., n].
        t_0 (tf.Tensor): The starting time for this step. Shape: [...].
        order (tf.Tensor): The order of the solver being used. Shape: [].
        rtol (tf.Tensor): Desired relative tolerance. Shape: [...] or [].
        atol (tf.Tensor): Desired absolute tolerance. Shape: [...] or [].

    Returns:
        tf.Tensor: Absolute value of the suggested initial step. Shape: [].
    """
    dy_dt = func(t_0, y_0)

    d_0s = []
    d_1s = []
    h_0s = []
    scales = []
    for dy_dt_n, y_0_n in zip(dy_dt, y_0):
        scale = atol + tf.abs(y_0_n) * rtol  # [..., n]

        d0 = tf.norm(y_0_n / scale, axis=-1)  # [...]
        d1 = tf.norm(dy_dt_n / scale, axis=-1)  # [...]

        h0 = tf.where(tf.logical_or(d0 < 1e-5, d1 < 1e-5),
                    tf.ones_like(d0, dtype=tf.float64) * 1e-6,
                    1e-2 * (d0 / d1))  # [...]

        d_0s.append(d0)
        d_1s.append(d1)
        h_0s.append(h0)
        scales.append(scale)

    h0 = tf.reduce_min(tf.stack([tf.reduce_min(h0_n) for h0_n in h_0s], axis=0))
    y1 = tuple(y_0_n + h0 * dy_dt_n for y_0_n, dy_dt_n in zip(y_0, dy_dt))  # tuple of [..., n]
    f1 = func(t_0 + h0, y1)  # tuple of [..., n]

    steps = []
    for d0_n, d1_n, dy_dt_n, f1_n, scale in zip(d_0s, d_1s, dy_dt, f1, scales):
        d2 = tf.norm((f1_n - dy_dt_n) / scale) / h0  # [...]

        h1 = tf.where(tf.logical_and(d1_n <= 1e-15, d2 <= 1e-15),
                      tf.maximum(tf.ones_like(d0_n, dtype=tf.float64) * 1e-6, h0 * 1e-3),
                      (1e-2 / (d1_n + d2))**(1. / (order + 1.)))  # [...]

        steps.append(tf.reduce_min(tf.minimum(100 * h0, h1)))
    return tf.reduce_min(steps)
