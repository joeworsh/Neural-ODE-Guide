# Copyright 2025 Joe Worsham

import tensorflow as tf
import tensorflow_probability as tfp


class Noise(tf.keras.Model):
    """
    Noise model that is based on the current state.
    A WhiteNoiseModel should be Brownian in motion.
    """
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def call(self, x: tf.Tensor, u: tf.Tensor, t: tf.Tensor) -> tfp.distributions.Distribution:
        """Produce a noise vector based on current state.

        Args:
            x (tf.Tensor): the state of the system at time t
            u (tf.Tensor): the control of the system at time t
            t (tf.Tensor): the current time of the system

        Returns:
            tfp.distributions.Distribution: a noise distribution to feed into the dynamical system
        """
        return self.compute(x, u, t)

    def compute(self, x: tf.Tensor, u: tf.Tensor, t: tf.Tensor) -> tfp.distributions.Distribution:
        """Noise business logic to implemented by the noise model.

        Args:
            x (tf.Tensor): the state of the system at time t
            u (tf.Tensor): the control of the system at time t
            t (tf.Tensor): the current time of the system

        Raises:
            NotImplementedError: a noise vector to feed into the dynamical system

        Returns:
            tfp.distributions.Distribution: a noise distribution to feed into the dynamical system
        """
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def noise_shape(self) -> int:
        """
        Returns:
            number of dimensions produced by this noise model
        """
        raise NotImplementedError


class WhiteNoise(Noise):
    """A simple white noise brownian motion model that can be applied
    to any n-dimensional system.
    """
    def __init__(self, name: str, scale_diag: tf.Tensor) -> None:
        """Create a new WhiteNoise model.

        Args:
            name (str): The unique name of this noise model.
            scale_diag (tf.Tensor): The diagonal scale (\sigma) of this noise model.
        """
        super().__init__(name)
        self._scale_diag = scale_diag

    def compute(self, x: tf.Tensor, u: tf.Tensor, t: tf.Tensor) -> tfp.distributions.Distribution:
        """Compute the distribution of the white noise model applied to x.
        This function will scale the distribution to match the batch dimensions of x.

        Args:
            x (tf.Tensor): The current state of the system. Used only for determining batch shape.
            u (tf.Tensor): System controls. Not used.
            t (tf.Tensor): System time. Not used.

        Returns:
            tfp.distributions.Distribution: White noise distribution for system.
        """
        n_scale_diag = tf.reshape(self._scale_diag, (1,)*len(x.shape[:-1]) + self._scale_diag.shape)
        n_scale_diag = tf.tile(n_scale_diag, x.shape[:-1] + (1,))
        return tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(n_scale_diag),
            scale_diag=n_scale_diag
        )

    def noise_shape(self) -> int:
        """
        Returns:
            int: Number of dimensions in the noise vector.
        """
        return self._scale_diag.shape[-1]
