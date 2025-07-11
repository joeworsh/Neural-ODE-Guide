# Copyright 2025 Joe Worsham

import tensorflow as tf


class Sensor(tf.keras.Model):
    """
    Sensor is the base class for all sensor and measurement simulations applied during the simulation.
    Note: sensor noise is additive and is not included directly here. The noise can be incorporated
    during simulation or recursive estimation.
    """
    def __init__(self, name: str) -> None:
        super().__init__(autocast=False)
        self._name = name

    def call(self, x: tf.Tensor, u: tf.Tensor, p: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Invoke this sensor model given the current state, control and time of the system.

        Args:
            x (tf.Tensor): The ground truth system state.
            u (tf.Tensor): The control vector applied. May be None if no controller is used.
            p (tf.Tensor): The parameters of the system. May be None.
            t (tf.Tensor): The current time of the system.

        Returns:
           tf.Tensor: Observation vector h of the system at time t.
        """
        return self.measure(x, u, p, t)

    def measure(self, x: tf.Tensor, u: tf.Tensor, p: tf.Tensor, t: tf.Tensor) ->  tf.Tensor:
        """Model sensor effects and return a modified sensor array for this system.

        Args:
            x (tf.Tensor): The ground truth system state.
            u (tf.Tensor): The control vector applied. May be None.
            p (tf.Tensor): The parameters of the system. May be None.
            t (tf.Tensor): The current time of the system.

        Returns:
            tf.Tensor: Observation vector h of the system at time t.
        """
        raise NotImplementedError("All sensors must implement the measure method to model sensor effects.")

    @property
    def name(self):
        return self._name

    @property
    def observation_shape(self):
        raise NotImplementedError

    @property
    def state_shape(self):
        raise NotImplementedError

    @property
    def control_shape(self):
        raise NotImplementedError

    @property
    def parameter_shape(self):
        raise NotImplementedError
