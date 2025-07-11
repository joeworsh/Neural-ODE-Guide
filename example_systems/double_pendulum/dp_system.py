# Copyright 2025 Joe Worsham

from tensorflow_dynamics.components.system import System

import tensorflow as tf


class DoublePendulumSystem(System):
    """https://www.myphysicslab.com/pendulum/double-pendulum-en.html
    """
    def step(self, x: tf.Tensor, u: tf.Tensor, p: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        t1, t2, w1, w2 = tf.unstack(x, axis=-1)
        g, m1, l1, m2, l2 = tf.unstack(p, axis=-1)

        denom = 2*m1 + m2 - m2*tf.math.cos(2*t1 + 2*t2) 

        w1_dot = -g*(2*m1 + m2) * tf.math.sin(t1)
        w1_dot -= m2*g*tf.math.sin(t1 - 2*t2)
        w1_dot -= 2*tf.math.sin(t1 - t2)*m2*(w2**2*l2 + w1**2*l1*tf.math.cos(t1 - t2))
        w1_dot /= l1*denom

        w2_dot = 2*tf.math.sin(t1 - t2)
        w2_dot *= (w1**2*l1*(m1 + m2) + g*(m1 + m2)*tf.math.cos(t1) + w2**2*l2*m2*tf.math.cos(t1 - t2))
        w2_dot /= l2*denom
        return tf.stack([w1, w2, w1_dot, w2_dot], axis=-1)

    @property
    def state_shape(self):
        return 4

    @property
    def control_shape(self):
        return 0

    @property
    def parameter_shape(self):
        return 5
