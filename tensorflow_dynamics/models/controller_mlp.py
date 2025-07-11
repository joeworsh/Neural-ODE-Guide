# Copyright 2025 Joe Worsham

import tensorflow as tf

from typing import List

from tensorflow_dynamics.components.controller import Controller


class ControllerMlp(Controller):
    """ControllerMlp is a neural network representing a system controller.
    It is trained to learn through integrating the dynamical system.
    """
    def __init__(self, name: str, state_shape: int, control_shape: int, parameter_shape: int,
                 p_in: bool=True, t_in: bool=True,
                 layers: List[int]=None, activation: str='relu', dropout: float=0.0):
        """Create a new ControllerMlp given the provided configuration.

        Args:
            state_shape (int): the number of dimensions in the state space
            control_shape (int): the number of dimensions in the control space
            parameter_shape (int): the number of dimensions in the parameter space
            p_in (bool, optional): Flag to take in parameters in the MLP as input. Defaults to True.
            t_in (bool, optional): Flag to take in time in the MLP as input. Defaults to True.
            layers (List[int], optional): List of layers and nodes. Defaults to [32, 32].
            activation (str, optional): The activation to use on hidden layers. Defaults to 'relu'.
            dropout (float, optional): A dropout rate to apply during training. Defaults to 0.0 (no dropout).
        """
        super().__init__(name)

        self._state_shape = state_shape
        self._control_shape = control_shape
        self._parameter_shape = parameter_shape
        self._p_in = p_in
        self._t_in = t_in
        self._dropout = dropout
        self._dropout_layer = tf.keras.layers.Dropout(dropout)

        # build the MLP to learn/define the system dynamics
        layers = [32, 32] if layers is None else layers
        dense_layers = []
        for nodes in layers:
            mid_layer = tf.keras.layers.Dense(nodes, activation, dtype=tf.float64)
            dense_layers.append(mid_layer)
        
        self._dense_layers = dense_layers

        self._out_layer = tf.keras.layers.Dense(self._control_shape, None, dtype=tf.float64)

    def step(self, x: tf.Tensor, p: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Call into the MLP defining a learnable system.

        Args:
            x (tf.Tensor): The state vector input.
            p (tf.Tensor): The parameter vector input.
            t (tf.Tensor): The system time input.

        Returns:
            tf.Tensor: A learned control output.
        """
        if self._p_in:
            x = tf.concat([x, p], axis=-1)
        if self._t_in:
            x = tf.concat([x, t], axis=-1)
        for layer in self._dense_layers:
            x = layer(x)
            self._dropout_layer(x, training=True)
        return self._out_layer(x)

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def control_shape(self):
        return self._control_shape

    @property
    def parameter_shape(self):
        return self._parameter_shape
