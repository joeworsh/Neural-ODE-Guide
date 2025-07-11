# Copyright 2025 Joe Worsham

from tensorflow_dynamics.components.sensor import Sensor

import tensorflow as tf


class BobSensor(Sensor):
    def measure(self, x, u, p, t):
        t1, t2, *_ = tf.unstack(x, axis=-1)
        *_, l1, _, l2 = tf.unstack(p, axis=-1)
        px = l1 * tf.math.sin(t1) + l2 * tf.math.sin(t2)
        py = -l1 * tf.math.cos(t1) - l2 * tf.math.cos(t2)
        pts = tf.stack([px, py], axis=-1)
        return pts
    
    @property
    def observation_shape(self):
        return 2

    @property
    def state_shape(self):
        return 4

    @property
    def control_shape(self):
        return 0

    @property
    def parameter_shape(self):
        return 5
