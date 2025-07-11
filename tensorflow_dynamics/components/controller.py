# Copyright 2025 Joe Worsham

import tensorflow as tf


class Controller(tf.keras.Model):
    """
    A base class to implement a controller with a fixed frequency.
    """
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def call(self, x: tf.Tensor, p: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Invoke the controller to produce a control vector for the system.

        Args:
            x (tf.Tensor): The current state (or estimated state of the system).
            p (tf.Tensor): The parameter vector that defines the system.
            t (tf.Tensor): The current time of the system.
        
        Returns:
            tf.Tensor: A new control vector for the system.
        """
        return self.compute(x, p, t)

    def compute(self, x: tf.Tensor, p: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Specify the custom business logic for this controller.

        Args:
            x (tf.Tensor): The current state (or estimated state of the system).
            p (tf.Tensor): The parameter vector that defines the system.
            t (tf.Tensor): The current time of the system.

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Returns:
            tf.Tensor: A new control vector for the system.
        """
        raise NotImplementedError("All system must override the step function with custom dynamics.")

    @property
    def name(self):
        return self._name

    @property
    def state_shape(self):
        raise NotImplementedError

    @property
    def control_shape(self):
        raise NotImplementedError

    @property
    def parameter_shape(self):
        raise NotImplementedError
