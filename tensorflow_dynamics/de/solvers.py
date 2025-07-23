# Copyright 2025 Joe Worsham
# based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
# and https://github.com/titu1994/tfdiffeq

import tensorflow as tf

from tensorflow_dynamics.de.ode_utils import (
    compute_error_ratio, optimal_step_size, select_initial_step
)
from tensorflow_dynamics.de.rk_common import (
    euler_step, rk4_step, rk38_step,
    midpoint_step, heun_step, ralston_step,
    dopri5_step,
    euler_maruyama_step, rk_sde_step
)

from typing import Callable, Tuple, Union


def _solve(func: Callable, y_0: Tuple[tf.Tensor],
           h: tf.Tensor, t_eval: tf.Tensor, step: Callable,
           solver_update: Callable) -> Tuple[Tuple[tf.Tensor], tf.Tensor]:
    """Run a dynamic ODE IVP solver on the provided problem
    with the provided step logic. The step logic is where
    solutions like RK4 can be provided.

    This implementation can handle a variable number of batch
    dimensions, indicated in the shapes below as [...].

    Note that the input may be a tuple to support
    ragged states being integrated through time. t_0 and dt are
    expected to scalar tensors.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Tuple[tf.Tensor]): The initial value at time 0. Shape: Tuple of [..., n] or [..., n].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        step (Callable): The step function used to solve the ODE.
        solver_update (Callable): The fixed or adaptive RK update.

    Returns:
        Tuple[tf.Tensor]: A solution to the ode. Shape: (tuple of) [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    # the GPU-based, TF-compliant solution will use
    # two arrays with the maximum number of entries set to the
    # maximum number of steps
    ys = tuple(tf.TensorArray(dtype=tf.float64, size=2, dynamic_size=True) for _ in y_0)

    # write the initial values to the array
    ys = tuple(ys_n.write(0, y_0n) for ys_n, y_0n in zip(ys, y_0))

    # define a while loop to take every step and write the results
    def cond(_, t, t_f, h):
        t_h = t + h
        return tf.reduce_any(tf.where(h > 0, t_h < t_f, t_h > t_f))

    def body(y, t, t_f, h):
        y, t, h = solver_update(y, t, h)
        return (y, t, t_f, h)

    # run the while loop to step through the solution
    prev_y = y_0
    prev_t = t_eval[..., 0]

    # iterate to each evaluation point and then to the final time
    eval_idx = tf.constant(1, dtype=tf.int32)
    for eval_pt in tf.unstack(t_eval, axis=-1)[1:]:
        # integrate to next evaluation point
        prev_y, prev_t, _, h = tf.while_loop(cond, body, [prev_y, prev_t, eval_pt, h])

        # finish step to evaluation point
        prev_y, _ = step(func, prev_y, prev_t, eval_pt - prev_t)
        prev_t = eval_pt
        ys = tuple(ys_n.write(eval_idx, prev_yn) for ys_n, prev_yn in zip(ys, prev_y))
        eval_idx += 1

    # gather and reshape results
    y_solution = tuple(ys_n.stack() for ys_n in ys)  # -> tuple of [T, ..., n]
    for ys_n in ys:
        ys_n.close()

    # re-order to get t into the inner spot
    reshaped_ys = []
    for y_0n, y_sol_n in zip(y_0, y_solution):
        batch_rank = len(y_0n.shape[:-1])
        perm = tf.concat([tf.range(batch_rank)+1, [0, batch_rank+1]], axis=-1)
        y_sol_n = tf.transpose(y_sol_n, perm=perm)  # -> [..., T, n]
        reshaped_ys.append(y_sol_n)
    y_solution = tuple(reshaped_ys)  # tuple of -> [..., T, n]

    return y_solution, t_eval


def solve_fixed(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                h: tf.Tensor, t_eval: tf.Tensor,
                step: Callable) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Run a fixed ODE IVP solver on the provided problem
    with the provided step logic. The step logic is where
    solutions like RK4 can be provided.

    This implementation can handle a variable number of batch
    dimensions, indicated in the shapes below as [...].

    Note that the input may be a tuple to support
    ragged states being integrated through time. t_0 and dt are
    expected to scalar tensors.

    Note: if t_0 > t_f, the step size will automatically be
    inverted. The value of h should always be positive.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: (tuple of) [..., n].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        step (Callable): The step function used to solve the ODE.

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: (tuple of) [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    is_tuple = True
    if not isinstance(y_0, tuple):
        is_tuple = False
        y_0 = (y_0,)

        def tuple_func(t, y):
            return (func(t, y[0]),)

        solve_func = tuple_func
    else:
        solve_func = func

    h = h if h is not None else select_initial_step(solve_func, y_0, t_eval[..., 0], 4, 1e-6, 1e-9)  # scalar
    h = tf.where(t_eval[..., -1] > t_eval[..., 0], h, -h)

    def body(y, t, h):
        y_next, _ = step(solve_func, y, t, h)
        t_next = t + h
        valid = tf.where(h > 0, t_next < t_eval[..., -1], t_next > t_eval[..., -1])
        y_next = tuple(tf.where(valid[:, None], y_next_n, y_n) for y_next_n, y_n in zip(y_next, y))
        t_next = tf.where(valid, t_next, t)
        return y_next, t_next, h

    y_eval, t_eval = _solve(solve_func, y_0, h, t_eval, step, body)

    # remove the tuple structure if it was not passed in
    if not is_tuple:
        y_eval = y_eval[0]

    return y_eval, t_eval


def solve_adaptive(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                   t_eval: tf.Tensor, step: Callable,
                   order: int, rtol: float=1e-3, atol: float=1e-6,
                   safety: float=0.9, ifactor: float=10.0, dfactor: float=0.2) ->\
                    Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Run an adaptive ODE IVP solver on the provided problem
    with the provided step logic. The step logic is where
    solutions like Dormand Prince can be provided.

    This implementation can handle a variable number of batch
    dimensions, indicated in the shapes below as [...].

    Note that the input may be a tuple to support
    ragged states being integrated through time. t_0 and dt are
    expected to scalar tensors.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (tf.Tensor): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        step (Callable): The step function used to solve the ODE.
        order (int): The power order of this integration.
        rtol (float): The provided relative error tolerance. Defaults to 1e-3.
        atol (float): The provided absolute error tolerance. Defaults to 1e-6.
        safety (float, optional): _description_. Defaults to 0.9.
        ifactor (float, optional): _description_. Defaults to 10.0.
        dfactor (float, optional): _description_. Defaults to 0.2.

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: (tuple of) [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    is_tuple = True
    if not isinstance(y_0, tuple):
        is_tuple = False
        y_0 = (y_0,)

        def tuple_func(t, y):
            return (func(t, y[0]),)

        solve_func = tuple_func
    else:
        solve_func = func

    # the initial step size can be determined by the first function call
    h = select_initial_step(solve_func, y_0, t_eval[..., 0], order, rtol, atol)  # scalar
    h = tf.where(t_eval[..., -1] > t_eval[..., 0], h, -h)

    def body(y, t, h):
        y_next, y_err = step(solve_func, y, t, h)
        t_next = t + h
        
        mse_ratio = compute_error_ratio(y_err, y, y_next, atol, rtol)  # tuple of [...]
        accept = tuple(tf.logical_and(mse_ratio_n <= 1, tf.where(h > 0, t_next < t_eval[..., -1], t_next > t_eval[..., -1])) for mse_ratio_n in mse_ratio)  # tuple of [...]
        accept_flat = tf.reduce_all(tf.stack([tf.reduce_all(accept_n) for accept_n in accept], axis=0))

        y_next = tuple(tf.where(accept_flat, y_next_n, y_n) for y_next_n, y_n in zip(y_next, y))
        t_next = tf.where(accept_flat, t_next, t)
        h_next = optimal_step_size(h, mse_ratio, safety, ifactor, dfactor, order)
        return y_next, t_next, h_next

    y_eval, t_eval = _solve(solve_func, y_0, h, t_eval, step, body)

    # remove the tuple structure if it was not passed in
    if not is_tuple:
        y_eval = y_eval[0]

    return y_eval, t_eval


def euler_solver(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                 t_eval: tf.Tensor, h: tf.Tensor=None,
                 **unused_kwargs) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched Euer solver.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [...].

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_fixed(func, y_0, h, t_eval, euler_step)


def rk4_solver(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
               t_eval: tf.Tensor, h: tf.Tensor=None,
               **unused_kwargs) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched RK4 solver.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [...].

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_fixed(func, y_0, h, t_eval, rk4_step)


def rk38_solver(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                t_eval: tf.Tensor, h: tf.Tensor=None,
                **unused_kwargs) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched RK38 solver.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [...].

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_fixed(func, y_0, h, t_eval, rk38_step)


def midpoint_solver(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                    t_eval: tf.Tensor, h: tf.Tensor=None,
                    **unused_kwargs) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched midpoint solver.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [...].

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_fixed(func, y_0, h, t_eval, midpoint_step)


def heun_solver(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                t_eval: tf.Tensor, h: tf.Tensor=None,
                **unused_kwargs) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched Heun solver.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [...].

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_fixed(func, y_0, h, t_eval, heun_step)


def ralston_solver(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                   t_eval: tf.Tensor, h: tf.Tensor=None,
                   **unused_kwargs) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched Ralston solver.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [...].

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_fixed(func, y_0, h, t_eval, ralston_step)


def dopri5_solver(func: Callable, y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                  t_eval: tf.Tensor, rtol: float=1e-6,
                  atol: float=1e-9, safety: float=0.9, ifactor: float=10.0,
                  dfactor: float=0.2, **unused_kwargs) ->\
                    Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched RK4 solver.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        kwargs: Configuration for adaptive solver including rtol, atol, safety, order, ifactor and dfactor.

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_adaptive(func, y_0, t_eval, dopri5_step,
                          order=5, rtol=rtol, atol=atol, safety=safety,
                          ifactor=ifactor, dfactor=dfactor)


def _create_fixed_sde_step(sde_step: Callable, diffusion: Callable,
                           y_0: Union[tf.Tensor, Tuple[tf.Tensor]]):
    """Helper method to create an SDE step with diffusion
    that looks like a standard ODE step to the fixed solver above.

    Args:
        sde_step (Callble): The SDE step to apply.
        diffusion (Callble): The diffusion dw/dt to model.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
    """
    solver_diffusion = diffusion
    if not isinstance(y_0, tuple):
        y_0 = (y_0,)

        def tuple_diffusion(t, y):
            return (diffusion(t, y[0]),)

        solver_diffusion = tuple_diffusion

    def fixed_sde_step(drift, y_0, t_0, dt):
        return sde_step(drift, solver_diffusion, y_0, t_0, dt)
    return fixed_sde_step


def euler_maruyama_solver(drift: Callable, diffusion: Callable,
                          y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                          t_eval: tf.Tensor, h: tf.Tensor=None,
                          **unused_kwargs) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched Euer-Maruyama SDE solver.

    Args:
        drift (Callable): The dy/dt component of the SDE.
        diffusion (Callable): The dw/dt component of the SDE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [...].

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_fixed(drift, y_0, h, t_eval,
                       _create_fixed_sde_step(euler_maruyama_step, diffusion, y_0))


def rk_sde_solver(drift: Callable, diffusion: Callable,
                  y_0: Union[tf.Tensor, Tuple[tf.Tensor]],
                  t_eval: tf.Tensor, h: tf.Tensor=None,
                  **unused_kwargs) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor]], tf.Tensor]:
    """Fixed step, batched Runge-Kutta SDE solver.

    Args:
        drift (Callable): The dy/dt component of the SDE.
        diffusion (Callable): The dw/dt component of the SDE.
        y_0 (Union[tf.Tensor, Tuple[tf.Tensor]]): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        h (tf.Tensor): The step size to use for each trajectory. Shape: [...].

    Returns:
        Union[tf.Tensor, Tuple[tf.Tensor]]: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    return solve_fixed(drift, y_0, h, t_eval,
                       _create_fixed_sde_step(rk_sde_step, diffusion, y_0))
