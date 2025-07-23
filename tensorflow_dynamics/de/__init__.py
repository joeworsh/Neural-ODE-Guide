# Copyright 2025 Joe Worsham
# based on https://github.com/titu1994/tfdiffeq

import tensorflow as tf

from tensorflow_dynamics.de.solvers import (
    euler_solver, rk4_solver, rk38_solver,
    midpoint_solver, heun_solver, ralston_solver,
    dopri5_solver,
    euler_maruyama_solver, rk_sde_solver
)

from typing import Callable, Dict, List, Tuple


# map all solvers by name to be used by the API
ODE_SOLVERS = {
    'euler': euler_solver,
    'rk4': rk4_solver,
    'rk38': rk38_solver,
    'midpoint': midpoint_solver,
    'heun': heun_solver,
    'ralston': ralston_solver,
    'dopri5': dopri5_solver
}

SDE_SOLVERS = {
    "em": euler_maruyama_solver,
    "rk": rk_sde_solver
}


class OdeintSolver(tf.keras.Model):
    """A Keras model to represent the solving of an ordinary differential
    equation (ODE) and the computing of gradients wrt to the solution.

    The solver can use direct auto-differentiation or the adjoint
    method to compute the gradient for variable optimization [1].

    Optionally, a continuous loss function may be provided along
    side dy/dt that can accumulate and be solved [2].

    References:

    [1] Chen, Ricky TQ, et al. "Neural ordinary differential equations."
    Advances in neural information processing systems 31 (2018).
    
    [2] Massaroli, Stefano, et al. "Dissecting neural odes."
    Advances in Neural Information Processing Systems 33 (2020): 3952-3963.
    """
    def __init__(self, dy_dt: tf.keras.Model, continuous_loss_func: Callable=None, method: str=None, options: Dict=None,
                 use_adjoint: bool=True, adjoint_method: str=None, adjoint_options: Dict=None):
        """Create a new OdeintSolver with the provided configuration.

        Args:
            dy_dt (tf.keras.Model): The dy/dt ODE.
            continuous_loss_func (Callable, optional): An optional continuous loss function that can accumulate continuously in y.
            method (str, optional): Indicates the integration method to use. Defaults to RK4.
            options (Dict, optional): Key value options to pass to the solver. Defaults to None.
            use_adjoint (bool, optional): True to default to using the adjoint method. Defaults to True.
            adjoint_method (str, optional): The solver to use when computing the adjoint gradient. When None - use the forward solver.
            adjoint_options (Dict, optional): Key value options to pass to the adjoint solver. Defaults to options.
        """
        # super().__init__(autocast=False)
        super().__init__()
        self._dy_dt = dy_dt
        self._continuous_loss_func = continuous_loss_func if continuous_loss_func is not None else self._zero_loss
        self._solver = ODE_SOLVERS.get(method, rk4_solver)
        self._options = options if options is not None else {}
        self._use_adjoint = use_adjoint
        self._adjoint_solver = ODE_SOLVERS.get(adjoint_method, rk4_solver) if adjoint_method is not None else self._solver
        self._adjoint_options = adjoint_options if adjoint_options is not None else self._options

        # stateful parameters set while computing the gradient
        self._n = None
        self._variables = None
        self._flat_batch_shape = None

    def call(self, y_0: tf.Tensor, t_eval: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Solve the configured ODE with the provided
        starting state as an initial value problem.

        Args:
            y_0 (tf.Tensor): The starting state. Shape: [..., n].
            t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].

        Returns:
            tf.Tensor: A solution to the ode. Shape: [..., t, n].
            tf.Tensor: The time steps of the solution. Shape: [..., t].
            tf.Tensor: The continuous loss of the solution. Shape: [...]. Will be zeros if loss functions not provided.
        """
        if self._use_adjoint:
            return self.odeint_adjoint(y_0, t_eval)
        return self.odeint(y_0, t_eval)

    def odeint(self, y_0: tf.Tensor, t_eval: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Solve the configured ODE with the provided
        starting state as an initial value problem.

        Use direct auto-differentiation if the gradient is taken.

        Args:
            y_0 (tf.Tensor): The starting state. Shape: [..., n].
            t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].

        Returns:
            tf.Tensor: A solution to the ode. Shape: [..., t, n].
            tf.Tensor: The time steps of the solution. Shape: [..., t].
            tf.Tensor: The continuous loss of the solution. Shape: [...]. Will be zeros if loss functions not provided.
        """
        # starting_loss = self._zero_loss(t_eval[..., 0], y_0)
        # aug_y_0 = tf.concat([y_0, starting_loss], axis=-1)  # [..., n+1]

        # aug_y_t, t_t = self._solver(self._dynamics_with_loss, aug_y_0, t_eval, **self._options)
        # y_t = aug_y_t[..., :-1]

        # # compute the losses for this solution
        # y_loss = aug_y_t[..., -1]

        y_t, t_t = self._solver(self._dy_dt, y_0, t_eval, **self._options)
        return y_t, t_t

    @tf.custom_gradient
    def odeint_adjoint(self, y_0: tf.Tensor, t_eval: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Solve the configured ODE with the provided
        starting state as an initial value problem.

        Use the adjoint method if the gradient is taken.

        Args:
            y_0 (tf.Tensor): The starting state. Shape: [..., n].
            t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].

        Returns:
            tf.Tensor: A solution to the ode. Shape: [..., t, n].
            tf.Tensor: The time steps of the solution. Shape: [..., t].
            tf.Tensor: The continuous loss of the solution. Shape: [...]. Will be zeros if loss functions not provided.
        """
        # start by running the forward pass
        # stop the gradients from being built directly
        y_t, t_t, y_loss = self.odeint(y_0, t_eval)  # [..., t, n], [..., t], [...]
        y_t = tf.stop_gradient(y_t)  # [..., t, n]
        t_t = tf.stop_gradient(t_t)  # [..., t]
        y_loss = tf.stop_gradient(y_loss)  # [...]

        # define the adjoint gradient of this ODE solution wrt the upstream dL_dy_f
        def grad(dL_dy_f: tf.Tensor, *_unused_grads, variables: List[tf.Tensor]=None) -> tf.Tensor:
            # dL_dy_f -> [..., t, n]
            self._variables = variables if variables is not None else [tf.constant([0,], dtype=tf.float64),]
            
            # the gradient starts at the end of the original computation
            y_f = y_t[..., -1, :]  # [..., n]
            y_f_loss = self._zero_loss(t_eval[..., -1], y_f)  # [..., 1]
            yl_f = tf.concat([y_f, y_f_loss], axis=-1)

            # gather shapes that are needed
            self._n = y_f.shape[-1]

            # flatten and reshape all parameters for adjoint gradient computation
            flat_variables = [tf.reshape(v, [-1]) for v in self._variables]
            flat_params = tf.concat(flat_variables, axis=0)
            # note: using a batch dimension on params because rk_common requires a batch dim right now
            zero_params = tf.zeros(flat_params.shape, tf.float64)[None, :]  # [1, p]

            # construct the adjoint integrand and solve it
            aug_y0 = tf.concat([yl_f, dL_dy_f[..., -1, :]], axis=-1)  # [..., 2n]
            aug_y0 = (aug_y0, zero_params)
            backpass_t = tf.stack([t_eval[..., -1], t_eval[..., 0]], axis=-1)
            y_aug, _ = self._adjoint_solver(self._augmented_dynamics_with_loss, aug_y0,
                                            backpass_t, **self._adjoint_options)
            dL_dy_0 = y_aug[0][..., -1, self._n+1:]
            returns = (dL_dy_0,) + tuple(None for _ in _unused_grads[:-1])  # TODO need to compute time gradients correctly

            # reconstruct parameter shapes correctly
            if variables is not None:
                dL_dparams_flat = y_aug[1][0, -1, :]  # [p,]
                dL_dparams = []
                for v, v_flat in zip(variables, flat_variables):
                    dL_dp_flat = dL_dparams_flat[:v_flat.shape[0]]
                    dL_dparams_flat = dL_dparams_flat[v_flat.shape[0]:]
                    dL_dparams.append(tf.reshape(dL_dp_flat, v.shape))

                returns = (returns, dL_dparams)

            self._n = None
            self._variables = None
            return returns

        return (y_t, t_t, y_loss), grad

    @tf.function
    def _dynamics_with_loss(self, t: tf.Tensor, y_aug: tf.Tensor) -> tf.Tensor:
        """A model of the ODE with the loss function appended to the end.
        Args:
            t (tf.Tensor): Time. Shape: [...].
            y_aug (tf.Tensor): State and loss. Shape: [..., n+1].
        Returns:
            tf.Tensor: dy/dt + dl. Shape: [..., n+1].
        """
        y = y_aug[..., :-1]  # [..., n]
        dy_dt = self._dy_dt(t, y)  # [..., n]
        dl_dt = self._continuous_loss_func(t, y)  # [..., 1]
        return tf.concat([dy_dt, dl_dt], axis=-1)  # [..., n+1]

    @tf.function
    def _augmented_dynamics_with_loss(self, t: tf.Tensor, y_aug: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computing the gradient via the adjoint method requires
        solving an augmented differential equations backwards in time.

        This function defines the augmented, backwards ODE.

        Uses the continuous loss adjoint defined in [1].

        References:

        [1] Massaroli, Stefano, et al. "Dissecting neural odes."
        Advances in Neural Information Processing Systems 33 (2020): 3952-3963.

        Args:
            t (tf.Tensor): Time. Shape: [...].
            y_aug (Tuple[tf.Tensor, tf.Tensor]): State. Shapes: ([..., n'], [p]).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: dy_aug / dt. Shape: [..., n'],
        """
        y = y_aug[0][..., :self._n]  # [..., n]
        min_adj_y = -y_aug[0][..., self._n+1:]  # [..., n]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(y)
            dy_dt = self._dy_dt(t, y)  # [..., n]
            dl_dt = self._continuous_loss_func(t, y)  # [..., 1]

        df_dy = tape.batch_jacobian(dy_dt, y,
                                    unconnected_gradients=tf.UnconnectedGradients.ZERO)  # [..., n, n]
        dl_dy = tape.gradient(dl_dt, y,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)  # [..., n]
        vjp_p = tape.gradient(dy_dt, self._variables,
                              output_gradients=min_adj_y,  # output gradients automatically computes a vector-jacobian
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)  # tuple of [p']
        vjp_p = tf.concat([tf.reshape(d, [-1,]) for d in vjp_p], axis=0)  # [p,]
        vjp_y = (min_adj_y[..., None, :] @ df_dy)[..., 0, :] - dl_dy  # [..., n]
        return (tf.concat([dy_dt, dl_dt, vjp_y], axis=-1), vjp_p[None, :])

    def _zero_loss(self, t: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Simple function to return a zero loss on any state y.

        Args:
            t (tf.Tensor): Time. Not used. Shape: [...].
            y (tf.Tensor): The state to provide a zero loss for.

        Returns:
            tf.Tensor: Zero loss. Shape [..., 1].
        """
        return tf.zeros_like(y[..., :1])  # [..., 1]


def odeint(func: Callable, y_0: tf.Tensor,
           t_eval: tf.Tensor, method=None, **options) -> Tuple[tf.Tensor, tf.Tensor]:
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor of any shape.

    This implementation can handle a variable number of batch
    dimensions, indicated in the shapes below as [...]. The
    batch shapes of y0 and t must be equal.

    Args:
        func (Callable): The dy/dt ODE.
        y_0 (tf.Tensor): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        method (str, optional): Indicates the integration method to use. Defaults to RK4.
        options: keyword arguments passed to the solver.

    Returns:
        tf.Tensor: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    if method is None:
        method = 'rk4'
    return ODE_SOLVERS[method](func, y_0, t_eval, **options)


def sdeint(drift: Callable, diffusion: Callable, y_0: tf.Tensor,
           t_eval: tf.Tensor, method=None, **options) -> Tuple[tf.Tensor, tf.Tensor]:
    """Integrate a system of stochastic differential equations.

    Solves the initial value problem for a non-stiff system of first order SDEs:
        ```
        dy/dt = drift(t, y) + diffusion(t, y), y(t[0]) = y0
        ```
    where y is a Tensor of any shape.

    This implementation can handle a variable number of batch
    dimensions, indicated in the shapes below as [...]. The
    batch shapes of y0 and t must be equal.

    Args:
        drift (Callable): The dy/dt component of the SDE.
        diffusion (Callable): The dw/dt component of the SDE.
        y_0 (tf.Tensor): The initial value at time 0. Shape: [..., n].
        t_eval (tf.Tensor): The time points at which to return a solution. Shape: [..., t].
        method (str, optional): Indicates the integration method to use. Defaults to RK4.
        options: keyword arguments passed to the solver.

    Returns:
        tf.Tensor: A solution to the ode. Shape: [..., t, n].
        tf.Tensor: The time steps of the solution. Shape: [..., t].
    """
    if method is None:
        method = 'em'
    return SDE_SOLVERS[method](drift, diffusion, y_0, t_eval, **options)
