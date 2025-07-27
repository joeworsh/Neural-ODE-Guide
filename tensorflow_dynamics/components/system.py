# Copyright 2025 Joe Worsham

import tensorflow as tf


class System(tf.keras.Model):
    """
    System is the base class for implementing new system dynamics for simulation.
    Note: only additive noise is supported in tfd systems. Noise is not shown here,
    but must be integrated via SDEs or included in recursive estimations.
    """
    def __init__(self) -> None:
        super().__init__(autocast=False)

    def call(self, x: tf.Tensor, u: tf.Tensor, p: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Invoke the system's dynamics and return the change in state x_dot.

        Args:
            x (tf.Tensor): The current state of the system
            u (tf.Tensor): The input controls to the system. Can be None.
            p (tf.Tensor): The parameter vector that defines the system.
            t (tf.Tensor): The current time of the system.

        Returns:
            tf.Tensor: x_dot to represent the change in state
        """
        return self.step(x, u, p, t)

    def step(self, x: tf.Tensor, u: tf.Tensor, p: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """The step function is the function used to implement differentiable dynamics and produce the change in state
        (x_dot).

        Args:
            x (tf.Tensor): The current state of the system
            u (tf.Tensor): The input controls to the system. Can be None.
            p (tf.Tensor): The parameter vector that defines the system.
            t (tf.Tensor): The current time of the system.
        
        Returns:
            tf.Tensor: x_dot to represent the change in state
        """
        raise NotImplementedError("All system must override the step function with custom dynamics.")

    @property
    def state_shape(self):
        raise NotImplementedError

    @property
    def control_shape(self):
        raise NotImplementedError

    @property
    def parameter_shape(self):
        raise NotImplementedError
