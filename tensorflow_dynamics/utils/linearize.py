# Copyright 2025 Joe Worsham

from tensorflow_dynamics.components.sensor import Sensor
from tensorflow_dynamics.components.system import System

import tensorflow as tf


@tf.function
def linearize_sensor(sensor: Sensor, x, u, p, n, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(u)
        tape.watch(n)
        y = sensor(x, u, p, n, t)
    H_x = tape.batch_jacobian(y, x)
    H_u = tape.batch_jacobian(y, u)
    J = tape.batch_jacobian(y, n)
    return H_x, H_u, J


@tf.function
def linearize_system(system: System, x, u, p, w, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(u)
        tape.watch(w)
        x_dot = system(x, u, p, w, t)
    F = tape.batch_jacobian(x_dot, x)
    G = tape.batch_jacobian(x_dot, u)
    L = tape.batch_jacobian(x_dot, w)
    return F, G, L
