# Copyright 2025 Joe Worsham
# based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
# and https://github.com/titu1994/tfdiffeq

import tensorflow as tf

from dataclasses import dataclass
from typing import Callable, List, Tuple

@dataclass
class ButcherTableau:
    """Runge-Kutta Butcher Tableau
    for defining common, explicit RK
    solvers. More info can be found at
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    rk_matrix: tf.Tensor  # [s, s]
    weights: tf.Tensor  # [s]
    nodes: tf.Tensor  # [s]
    placeholders: List[tf.Tensor]  # [s]
    weights_err: tf.Tensor=None  # [s]


def second_order_two_stage_tableau(alpha: float) -> ButcherTableau:
    """Create a second-order, two-stage tablue parameterized
    by the provided alpha.

    More info can be found at
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    Args:
        alpha (float): 2nd order mixing parameter.

    Returns:
        ButcherTableau: Associated Butcher tableau.
    """
    d = 1. / (2.* alpha)
    return ButcherTableau(
        rk_matrix=tf.constant(
            [[0., 0.],
            [alpha, 0.]], dtype=tf.float64),
        weights=tf.constant([1. - d, d], dtype=tf.float64),
        nodes=tf.constant([0., alpha], dtype=tf.float64),
        placeholders=[None,] * 2
    )


# fixed RK methods
EULER = ButcherTableau(
    rk_matrix=tf.constant([[0.,]], dtype=tf.float64),
    weights=tf.constant([1.,], dtype=tf.float64),
    nodes=tf.constant([0.,], dtype=tf.float64),
    placeholders=[None,]
)

RK4 = ButcherTableau(
    rk_matrix=tf.constant(
        [[0., 0., 0., 0.],
         [0.5, 0., 0., 0.],
         [0., 0.5, 0., 0.],
         [0., 0., 1., 0.]], dtype=tf.float64),
    weights=tf.constant([1./6., 1./3., 1./3., 1./6.], dtype=tf.float64),
    nodes=tf.constant([0., 0.5, 0.5, 1.0], dtype=tf.float64),
    placeholders=[None,] * 4
)

RK38 = ButcherTableau(
    rk_matrix=tf.constant(
        [[0., 0., 0., 0.],
         [1./3., 0., 0., 0.],
         [-1./3., 1., 0., 0.],
         [1., -1., 1., 0.]], dtype=tf.float64),
    weights=tf.constant([1./8., 3./8., 3./8., 1./8.], dtype=tf.float64),
    nodes=tf.constant([0., 1./3., 2./3., 1.0], dtype=tf.float64),
    placeholders=[None,] * 4
)

MIDPOINT = second_order_two_stage_tableau(0.5)
HEUN = second_order_two_stage_tableau(1.0)
RALSTON = second_order_two_stage_tableau(2./3.)

# adaptive RK methods
DOPRI5 = ButcherTableau(
    rk_matrix=tf.constant(
        [[0., 0., 0., 0., 0., 0., 0.],
         [1./5., 0., 0., 0., 0., 0., 0.],
         [3./40., 9./40., 0., 0., 0., 0., 0.],
         [44./45., -56./15., 32./9., 0., 0., 0., 0.],
         [19372./6561., -25360./2187., 64448./6561., -212./729., 0., 0., 0.],
         [9017./3168., -355./33., 46732./5247., 49./176., -5103./18656., 0., 0.],
         [35./384., 0., 500./1113., 125./192., -2187./6784., 11./84., 0.]],
         dtype=tf.float64),
    weights=tf.constant([35./384., 0., 500./1113., 125./192., -2187./6784., 11./84., 0.], dtype=tf.float64),
    nodes=tf.constant([0., 1./5., 3./10., 4./5., 8./9., 1., 1.], dtype=tf.float64),
    placeholders=[None,] * 7,
    weights_err=tf.constant([35./384. - 5179./57600., 0., 500./1113. - 7571./16695., 125./192. - 393./640., -2187./6784. + 92097./339200., 11./84. - 187./2100., -1./40.], dtype=tf.float64)
)


def fixed_rk_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
                  dt: tf.Tensor, tableau: ButcherTableau) ->\
                    Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Common Runge-Kutta fixed step function following
    the weights provided in an RK Butcher Tableau.

    Note that the input is expected to be a tuple to support
    ragged states being integrated through time. t_0 and dt are
    expected to scalar tensors.

    More info can be found at
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.
        tableau (ButcherTableau): The tableua describing the computation for the step.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    s = tableau.nodes.shape[0]

    # solve k1 out of loop
    tableau.placeholders[0] = func(t_0, y_0)

    # solve all other stages based on previous stages
    for i in tf.range(1, s):
        ti = t_0 + tableau.nodes[i]*dt
        
        k = tuple(tf.stack([k_n[j] for k_n in tableau.placeholders[:i]], axis=-1) for j in range(len(y_0)))
        k = tuple(tableau.rk_matrix[i, :i] * k_n for k_n in k)
        k = tuple(tf.reduce_sum(k_n, axis=-1) for k_n in k)
        yi = tuple(y_0_n + dt[:, None] * k_n for y_0_n, k_n in zip(y_0, k))
        tableau.placeholders[i] = func(ti, yi)

    k = tuple(tf.stack([k_n[j] for k_n in tableau.placeholders], axis=-1) for j in range(len(y_0)))  # tuple of [..., n, s]
    k = tuple(tableau.weights * k_n for k_n in k)
    k = tuple(tf.reduce_sum(k_n, axis=-1) for k_n in k)  # tuple of [..., n]

    y_1 = tuple(y_0_n + dt[:, None] * k_n for y_0_n, k_n in zip(y_0, k))
    y_err = tuple(tf.zeros_like(y_0_n) for y_0_n in y_0)

    return y_1, y_err


def adaptive_rk_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
                     dt: tf.Tensor, tableau: ButcherTableau) ->\
                        Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Runge-Kutta step function with error following
    the weights provided in an RK Butcher Tableau.

    This is typically used with adaptive step sizes.

    Note that the input is expected to be a tuple to support
    ragged states being integrated through time. t_0 and dt are
    expected to scalar tensors.

    More info can be found at
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor) -> tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.
        tableau (ButcherTableau): The tableau describing the computation for the step.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: The computed errors of this step evaluation.
    """
    s = tableau.nodes.shape[0]

    # solve k1 out of loop
    tableau.placeholders[0] = func(t_0, y_0)

    # solve all other stages based on previous stages
    for i in tf.range(1, s):
        ti = t_0 + tableau.nodes[i]*dt
        k = tuple(tf.stack([k_n[j] for k_n in tableau.placeholders[:i]], axis=-1) for j in range(len(y_0)))
        k = tuple(tableau.rk_matrix[i, :i] * k_n for k_n in k)
        k = tuple(tf.reduce_sum(k_n, axis=-1) for k_n in k)
        yi = tuple(y_0_n + dt * k_n for y_0_n, k_n in zip(y_0, k))
        tableau.placeholders[i] = func(ti, yi)

    k = tuple(tf.stack([k_n[j] for k_n in tableau.placeholders], axis=-1) for j in range(len(y_0)))  # tuple of [..., n, s]
    k_val = tuple(tableau.weights * k_n for k_n in k)
    k_val = tuple(tf.reduce_sum(k_n, axis=-1) for k_n in k_val)  # tuple of [..., n]

    k_err = tuple(tableau.weights_err * k_n for k_n in k)
    k_err = tuple(tf.reduce_sum(k_n, axis=-1) for k_n in k_err)  # tuple of [..., n]

    y_1 = tuple(y_0_n + dt * k_n for y_0_n, k_n in zip(y_0, k_val))
    y_err = tuple(dt * k_n for k_n in k_err)
    return y_1, y_err


def euler_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
               dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the Euler RK step.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    return fixed_rk_step(func, y_0, t_0, dt, EULER)


def rk4_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
             dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the RK4 RK step.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    return fixed_rk_step(func, y_0, t_0, dt, RK4)


def rk38_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
              dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the RK38 RK step.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    return fixed_rk_step(func, y_0, t_0, dt, RK38)


def midpoint_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
                  dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the midpoint RK step.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    return fixed_rk_step(func, y_0, t_0, dt, MIDPOINT)


def heun_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
              dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the Heun RK step.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    return fixed_rk_step(func, y_0, t_0, dt, HEUN)


def ralston_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
                 dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the Ralston RK step.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    return fixed_rk_step(func, y_0, t_0, dt, RALSTON)


def dopri5_step(func: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
                dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the Dormand-Prince 4/5 order RK adaptive step.

    Args:
        func (Callable): The ODE to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: The 4/5 errors of this step.
    """
    return adaptive_rk_step(func, y_0, t_0, dt, DOPRI5)


def euler_maruyama_step(drift: Callable, diffusion: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
                        dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the Euler-Maruyama SDE step.
    https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method

    Args:
        drift (Callable): The drift to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        diffusion (Callable): The diffusion to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    drift = tuple(dx_n * dt for dx_n in drift(t_0, y_0))
    d_omega = tuple(tf.random.normal(
        shape=y_0_n.shape,
        mean=tf.zeros_like(y_0_n, dtype=tf.float64),
        stddev=tf.ones_like(y_0_n, dtype=tf.float64)*tf.sqrt(dt),
        dtype=tf.float64) for y_0_n in y_0)
    diffuse = tuple(dw_n * d_omega_n for dw_n, d_omega_n in zip(diffusion(t_0, y_0), d_omega))
    y_1 = tuple(y_0_n + drift_n + diffuse_n for y_0_n, drift_n, diffuse_n in zip(y_0, drift, diffuse))
    y_err = tuple(tf.zeros_like(y_0_n) for y_0_n in y_0)
    return y_1, y_err


def rk_sde_step(drift: Callable, diffusion: Callable, y_0: Tuple[tf.Tensor], t_0: tf.Tensor,
                        dt: tf.Tensor) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
    """Implementation of the Runge-Kutta SDE step.
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)

    Args:
        drift (Callable): The drift to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        diffusion (Callable): The diffusion to solve with signature (t: tf.Tensor, y: tf.Tensor)->tf.Tensor.
        y_0 (Tuple[tf.Tensor]): The starting values for this step.
        t_0 (tf.Tensor): The starting time for this step.
        dt (tf.Tensor): The change in time to solve for.

    Returns:
        Tuple[tf.Tensor]: The next values of y at t_0+dt.
        Tuple[tf.Tensor]: Zeros to indicate that no error was computed.
    """
    d_omega = tuple(tf.random.normal(
        shape=y_0_n.shape,
        mean=tf.zeros_like(y_0_n, dtype=tf.float64),
        stddev=tf.ones_like(y_0_n, dtype=tf.float64)*tf.sqrt(dt),
        dtype=tf.float64) for y_0_n in y_0)
    drift = tuple(dx_n * dt for dx_n in drift(t_0, y_0))
    diffuse = diffusion(t_0, y_0)
    upsilon = tuple(y_0_n + drift_n + diffuse_n * dt**0.5 for y_0_n, drift_n, diffuse_n in zip(y_0, drift, diffuse))
    b_upsilon = diffusion(t_0, upsilon)
    rk = tuple(0.5 * (b_upsilon_n - diffuse_n) * (d_omega_n**2 - dt) * dt**-0.5 for b_upsilon_n, diffuse_n, d_omega_n in zip(b_upsilon, diffuse, d_omega))
    y_1 = tuple(y_0_n + drift_n + diffuse_n * d_omega_n + rk_n for y_0_n, drift_n, diffuse_n, d_omega_n, rk_n in zip(y_0,  drift, diffuse, d_omega, rk))
    y_err = tuple(tf.zeros_like(y_0_n) for y_0_n in y_0)
    return y_1, y_err
