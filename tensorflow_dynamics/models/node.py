# Copyright 2025 Joe Worsham

import tensorflow as tf

from tensorflow_dynamics.models.layers import AutonomousDxDtMLP
from tensorflow_dynamics.de import OdeintSolver

from typing import Dict, List


class BaseNODE(tf.keras.Model):
    """A base Neural Ordinary Differential Equation (NODE) model [1, 3].
    This NODE form is based on [3] where an initial state encoder and a final
    state decoder are applied on the input and output.

    This base implementation can be used to configure standard NODEs [1],
    ANODEs [2], SONODEs [3] and any variable configuration among such
    models.

    References:

    [1] Chen, Ricky TQ, et al. "Neural ordinary differential equations."
    Advances in neural information processing systems 31 (2018).

    [2] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes."
    Advances in neural information processing systems 32 (2019).

    [3] Norcliffe, Alexander, et al. "On second order behaviour in augmented neural odes."
    Advances in neural information processing systems 33 (2020): 5911-5921.
    """
    def __init__(self, dx_dt: tf.keras.Model,
                 continuous_loss: tf.keras.Model=None,
                 encoder: tf.keras.Model=None, decoder: tf.keras.Model=None,
                 t_eval: tf.Tensor=None,
                 method: str=None, options: Dict=None,
                 use_adjoint: bool=True, adjoint_method: str=None, adjoint_options: Dict=None):
        """Create a new base NODE configured with the supplied parameters.

        Args:
            dx_dt (tf.keras.Model): The keras model of state transition dx/dt.
            continuous_loss (tf.keras.Model): Optional continuous loss function to adhere to.
            encoder (tf.keras.Model, optional): An optional model to encode x. Defaults to no encoding.
            decoder (tf.keras.Model, optional): An optional model to decode the solution. Defaults to no decoding.
            t_eval (tf.Tensor, optional): The evaluation times for evolution of the state vector. Shape: [1, t].
            method (str, optional): Indicates the integration method to use. Defaults to RK4.
            options (Dict, optional): Key value options to pass to the solver. Defaults to None.
            use_adjoint (bool, optional): True to default to using the adjoint method. Defaults to True.
            adjoint_method (str, optional): The solver to use when computing the adjoint gradient. When None - use the forward solver.
            adjoint_options (Dict, optional): Key value options to pass to the adjoint solver. Defaults to options.
        """
        super().__init__()
        self._dx_dt = dx_dt
        self._continuous_loss = continuous_loss
        self._encoder = encoder if encoder is not None else tf.identity
        self._decoder = decoder if decoder is not None else tf.identity

        # solver configurations
        self._t_0 = tf.zeros([1,], dtype=tf.float64) if t_eval is None else t_eval[..., 0]
        self._t_f = tf.ones([1,], dtype=tf.float64) if t_eval is None else t_eval[..., -1]
        self._t_eval = self._t_f if t_eval is None else t_eval  # should be shape [1, t]
        self._solver = OdeintSolver(self._dx_dt, continuous_loss, method, options, use_adjoint, adjoint_method, adjoint_options)

    def call(self, x: tf.Tensor, decode=True) -> tf.Tensor:
        """Invoke the NODE with input vector x.

        Args:
            x (tf.Tensor): Input vector. Shape: [..., n].

        Returns:
            tf.Tensor: Final time solution of the ODE solver. Shape: [..., n].
        """
        x = self._encoder(x)
        xt_hat, *_ = self._solver(x, self._t_eval)
        return self._decoder(xt_hat)  # [..., n]


class NODE(BaseNODE):
    """A Neural Ordinary Differential Equation (NODE) model [1].

    References:

    [1] Chen, Ricky TQ, et al. "Neural ordinary differential equations."
    Advances in neural information processing systems 31 (2018).
    """
    def __init__(self, n: int, t_0: tf.Tensor=None, t_f: tf.Tensor=None,
                 layers: List[int]=None, activation: str='tanh',
                 method: str=None, options: Dict=None,
                 use_adjoint: bool=True, adjoint_method: str=None, adjoint_options: Dict=None):
        """Create a new NODE configured with the supplied parameters.

        Args:
            n (int): The dimensionality of the state vector.
            a (int, optional): The optional augmented dimensionality of the state vector. Defaults to zero.
            t_0 (tf.Tensor): The starting time for each solution. Shape: [1].
            t_f (tf.Tensor): The final time for each trajectory. Shape: [1].
            layers (List[int], optional): Configuring MLP nodes and layers. Defaults to [32, 32].
            activation (str, optional): Activation to apply in MLP hidden layers. Defaults to 'tanh'.
            method (str, optional): Indicates the integration method to use. Defaults to RK4.
            options (Dict, optional): Key value options to pass to the solver. Defaults to None.
            use_adjoint (bool, optional): True to default to using the adjoint method. Defaults to True.
            adjoint_method (str, optional): The solver to use when computing the adjoint gradient. When None - use the forward solver.
            adjoint_options (Dict, optional): Key value options to pass to the adjoint solver. Defaults to options.
        """
        dx_dt = AutonomousDxDtMLP(n, layers, activation)
        super().__init__(n, dx_dt, t_0=t_0, t_f=t_f, method=method, options=options,
                         use_adjoint=use_adjoint, adjoint_method=adjoint_method,
                         adjoint_options=adjoint_options)
