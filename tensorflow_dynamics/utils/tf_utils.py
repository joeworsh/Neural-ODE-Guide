# Copyright 2025 Joe Worsham

import tensorflow as tf
import tensorflow_probability as tfp


def block_diagonal(matrices):
    linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in matrices]
    linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
    return linop_block_diagonal.to_dense()


def tril(x: tf.Tensor, dim: int):
    xt = tfp.math.fill_triangular(x)
    perm = list(range(len(xt.shape)))
    perm[-2] = len(xt.shape) - 1
    perm[-1] = len(xt.shape) - 2
    return xt+tf.transpose(xt, perm=perm)-tf.eye(dim, dtype=x.dtype)*xt


def tril_inv(x, dim):
    xt_hat = tf.linalg.band_part(x, -1, 0)
    return tfp.math.fill_triangular_inverse(xt_hat)
