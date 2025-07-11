# Copyright 2025 Joe Worsham

import tensorflow as tf

from tensorflow_dynamics.models.layers import AutonomousDxDtMLP, Truncate, ZeroAugmenter
from tensorflow_dynamics.models.node import BaseNODE

from typing import Dict, List


class ANODE(BaseNODE):
    """An Augmented Neural Ordinary Differential Equation (ANODE) model [1].

    References:

    [1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes."
    Advances in neural information processing systems 32 (2019).
    """
    def __init__(self, n: int, a: int, t_0: tf.Tensor=None, t_f: tf.Tensor=None,
                 layers: List[int]=None, activation: str='tanh',
                 method: str=None, options: Dict=None,
                 use_adjoint: bool=True, adjoint_method: str=None, adjoint_options: Dict=None):
        """Create a new ANODE configured with the supplied parameters.
        The dimensionality of the ANODE is n+a.

        Args:
            n (int): The dimensionality of the state vector.
            a (int): The augmented dimensionality of the state vector.
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
        augmention = ZeroAugmenter(n, a)
        decoder = Truncate(n)
        dx_dt = AutonomousDxDtMLP(a + n, layers, activation)
        super().__init__(n, dx_dt, encoder=augmention, decoder=decoder,
                         t_0=t_0, t_f=t_f, method=method, options=options,
                         use_adjoint=use_adjoint, adjoint_method=adjoint_method,
                         adjoint_options=adjoint_options)
