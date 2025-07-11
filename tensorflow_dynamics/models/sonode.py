# Copyright 2025 Joe Worsham

import tensorflow as tf

from tensorflow_dynamics.models.layers import Autonomous2ndOrderDxDtMLP, MLPAugmenter, Truncate
from tensorflow_dynamics.models.node import BaseNODE

from typing import Dict, List


class SONODE(BaseNODE):
    """A 2nd-Order Neural Ordinary Differential Equation (NODE) model [1].

    References:

    [1] Norcliffe, Alexander, et al. "On second order behaviour in augmented neural odes."
    Advances in neural information processing systems 33 (2020): 5911-5921.
    """
    def __init__(self, n: int, t_0: tf.Tensor=None, t_f: tf.Tensor=None,
                 vel_layers: List[int]=None, vel_activation: str='relu',
                 dxdt_layers: List[int]=None, dxdt_activation: str='tanh',
                 method: str=None, options: Dict=None,
                 use_adjoint: bool=True, adjoint_method: str=None, adjoint_options: Dict=None):
        """Create a new SONODE configured with the supplied parameters.

        Args:
            n (int): The dimensionality of the state vector.
            t_0 (tf.Tensor): The starting time for each solution. Shape: [1].
            t_f (tf.Tensor): The final time for each trajectory. Shape: [1].
            vel_layers (List[int], optional): Configuring velocity MLP nodes and layers. Defaults to [32, 32].
            vel_activation (str, optional): Activation to apply in velocity MLP hidden layers. Defaults to 'tanh'.
            dxdt_layers (List[int], optional): Configuring dx/dt MLP nodes and layers. Defaults to [32, 32].
            dxdt_activation (str, optional): Activation to apply in dx/dt MLP hidden layers. Defaults to 'tanh'.
            method (str, optional): Indicates the integration method to use. Defaults to RK4.
            options (Dict, optional): Key value options to pass to the solver. Defaults to None.
            use_adjoint (bool, optional): True to default to using the adjoint method. Defaults to True.
            adjoint_method (str, optional): The solver to use when computing the adjoint gradient. When None - use the forward solver.
            adjoint_options (Dict, optional): Key value options to pass to the adjoint solver. Defaults to options.
        """
        augmention = MLPAugmenter(n, vel_layers, vel_activation)
        decoder = Truncate(n)
        dx_dt = Autonomous2ndOrderDxDtMLP(n, dxdt_layers, dxdt_activation)
        super().__init__(n, dx_dt, encoder=augmention, decoder=decoder,
                         t_0=t_0, t_f=t_f, method=method, options=options,
                         use_adjoint=use_adjoint, adjoint_method=adjoint_method,
                         adjoint_options=adjoint_options)
