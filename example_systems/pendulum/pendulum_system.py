# Copyright 2025 Joe Worsham

from tensorflow_dynamics.components.system import System

import tensorflow as tf


class PendulumSystem(System):
    """https://ctms.engin.umich.edu/CTMS/index.php?aux=Activities_Pendulum
    """
    def step(self, x: tf.Tensor, u: tf.Tensor, p: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        theta, omega = tf.unstack(x, axis=-1)
        g, m1, m2, l, b = tf.unstack(p, axis=-1)

        # compute several constants used during the dynamics
        lg = (m2 * l + 0.5 * m1 * l) / (m2 + m1)  # center of mass
        io = (m1 * l ** 2) / 3 + m2 * l ** 2
        transform = -(m2 + m1) * g * lg
        
        omega_dot = (transform * tf.math.sin(theta) - b * omega) / io
        return tf.stack([omega, omega_dot], axis=-1)

    @property
    def state_shape(self):
        return 2

    @property
    def control_shape(self):
        return 0

    @property
    def parameter_shape(self):
        return 5
