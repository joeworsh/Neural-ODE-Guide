# Copyright 2025 Joe Worsham

import tensorflow as tf

from typing import List


class AutonomousDxDtMLP(tf.keras.Model):
    """Standard MLP for modeling the dx/dt of an autonomous
    (time-invariant) system. The time input t is dropped.
    """
    def __init__(self, n: int, layers: List[int]=None, activation: str='tanh',
                 initializer=None):
        """Create a new AutonomousDxDtMLP layer and it's weights.

        Args:
            n (int): The dimensionality of the state vector.
            layers (List[int], optional): Configuring MLP nodes and layers. Defaults to [32, 32].
            activation (str, optional): Activation to apply in MLP hidden layers. Defaults to 'tanh'.
        """
        super().__init__()

        # build the dx_dt MLP (the ODE)
        layers = [32, 32] if layers is None else layers
        dense_layers = []
        for nodes in layers:
            mid_layer = tf.keras.layers.Dense(nodes, activation, dtype=tf.float64,
                                              kernel_initializer=initializer)
            dense_layers.append(mid_layer)
        
        self._dense_layers = dense_layers
        self._out_layer = tf.keras.layers.Dense(n, None, dtype=tf.float64,
                                                kernel_initializer=initializer)

    def call(self, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Invoke the ODE MLP layer.

        Args:
            t (tf.Tensor): Time input. Dropped.
            x (tf.Tensor): Input state vector. Shape [..., n].

        Returns:
            tf.Tensor: dx/dt derivative vector. Shape [..., n].
        """
        for dense_layer in self._dense_layers:
            x = dense_layer(x)
        return self._out_layer(x)

    
class Autonomous2ndOrderDxDtMLP(tf.keras.Model):
    """Standard MLP for modeling the dx/dt of a second order
    autonomous (time-invariant) system. The state is assumed to be a first
    order form of a second order system (i.e. [x, x']). The dx/dt of the
    system (modeled by the MLP) is [x', x''].

    The time input t is dropped.
    """
    def __init__(self, n: int, layers: List[int]=None, activation: str='tanh',
                 initializer=None):
        """Create a new Autonomous2ndOrderDxDtMLP layer and it's weights.

        Args:
            n (int): The dimensionality of the state vector (not including the velocity).
            layers (List[int], optional): Configuring MLP nodes and layers. Defaults to [32, 32].
            activation (str, optional): Activation to apply in MLP hidden layers. Defaults to 'tanh'.
        """
        super().__init__()

        # build the dx_dt MLP (the ODE)
        layers = [32, 32] if layers is None else layers
        dense_layers = []
        for nodes in layers:
            mid_layer = tf.keras.layers.Dense(nodes, activation, dtype=tf.float64,
                                              kernel_initializer=initializer)
            dense_layers.append(mid_layer)
        
        self._dense_layers = dense_layers
        self._out_layer = tf.keras.layers.Dense(n, None, dtype=tf.float64,
                                                kernel_initializer=initializer)

    def call(self, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Invoke the 2nd order ODE MLP layer.

        Args:
            t (tf.Tensor): Time input. Dropped.
            x (tf.Tensor): Input state vector. Shape [..., 2*n].

        Returns:
            tf.Tensor: dx/dt derivative vector. Shape [..., 2*n].
        """
        # vel = x[..., self._n:]
        vel, _ = tf.split(x, 2, axis=-1)
        acc = x
        for dense_layer in self._dense_layers:
            acc = dense_layer(acc)
        acc = self._out_layer(acc)  # [..., n]
        return tf.concat([vel, acc], axis=-1)  # [..., 2n]


class LinearAugmenter(tf.keras.Model):
    """A learnable linear NODE augmenter which produces an augmentation
    on x that is a linear transformation of input x.

    Used by non-zero ANODEs [1] and SONODES [2]. The use of a linear
    transform is motivated by [3].

    References:

    [1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes."
    Advances in neural information processing systems 32 (2019).

    [2] Norcliffe, Alexander, et al. "On second order behaviour in augmented neural odes."
    Advances in neural information processing systems 33 (2020): 5911-5921.

    [3] Massaroli, Stefano, et al. "Dissecting neural odes."
    Advances in Neural Information Processing Systems 33 (2020): 3952-3963.
    """
    def __init__(self, a: int):
        """Create a new LinearAugmenter with the configured learning model.

        Args:
            a (int): The augmented dimensionality of the state vector.
        """
        super().__init__(autocast=False)
        self._out_layer = tf.keras.layers.Dense(a, None, dtype=tf.float64)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the linear augmentation of the input
        x and concatenate it onto the end of x for NODE
        processing.

        Args:
            x (tf.Tensor): The input vector to augment. Shape [..., n].

        Returns:
            tf.Tensor: The learned augmentation. Shape [..., a+n].
        """
        aug_x = self._out_layer(x)

        # combine x and the augmentation together
        return tf.concat([x, aug_x], axis=-1)  # [..., n+a]


class LinearTransform(tf.keras.Model):
    """A learnable linear NODE transform which produces an encoding
    on x that is a linear transformation of input x.

    Used by non-zero ANODEs [1]. The use of a linear transform is
    motivated by [2].

    References:

    [1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes."
    Advances in neural information processing systems 32 (2019).

    [2] Massaroli, Stefano, et al. "Dissecting neural odes."
    Advances in Neural Information Processing Systems 33 (2020): 3952-3963.
    """
    def __init__(self, a: int):
        """Create a new LinearTransform with the configured learning linear layer.

        Args:
            a (int): The dimensionality of the transformed state vector.
        """
        super().__init__(autocast=False)
        self._out_layer = tf.keras.layers.Dense(a, None, dtype=tf.float64)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the linear transformation of the input
        for NODE processing.

        Args:
            x (tf.Tensor): The input vector to transform. Shape [..., n].

        Returns:
            tf.Tensor: The learned transformation. Shape [..., a].
        """
        return self._out_layer(x)


class MLPAugmenter(tf.keras.Model):
    """A learnable NODE augmenter which produces an augmentation
    on x that is a function of input x.

    Used by non-zero ANODEs [1] and SONODES [2].

    References:

    [1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes."
    Advances in neural information processing systems 32 (2019).

    [2] Norcliffe, Alexander, et al. "On second order behaviour in augmented neural odes."
    Advances in neural information processing systems 33 (2020): 5911-5921.
    """
    def __init__(self, a: int, layers: List[int]=None, activation: str='relu'):
        """Create a new MLPAugmenter with the configured learning model.

        Args:
            a (int): The augmented dimensionality of the state vector.
            layers (List[int], optional): Configuring MLP nodes and layers. Defaults to [32, 32].
            activation (str, optional): Activation to apply in MLP hidden layers. Defaults to 'tanh'.
        """
        super().__init__(autocast=False)

        # build the augmentation encoder
        layers = [32, 32] if layers is None else layers
        self._dense_layers = []
        for nodes in layers:
            mid_layer = tf.keras.layers.Dense(nodes, activation, dtype=tf.float64)
            self._dense_layers.append(mid_layer)
        
        self._out_layer = tf.keras.layers.Dense(a, None, dtype=tf.float64)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the augmentation of the input
        x and concatenate it onto the end of x for NODE
        processing.

        Args:
            x (tf.Tensor): The input vector to augment. Shape [..., n].

        Returns:
            tf.Tensor: The learned augmentation. Shape [..., a+n].
        """
        aug_x = x
        for layer in self._dense_layers:
            aug_x = layer(aug_x)
        aug_x = self._out_layer(aug_x)

        # combine x and the augmentation together
        return tf.concat([x, aug_x], axis=-1)  # [..., n+a]


class MLPTransform(tf.keras.Model):
    """A learnable NODE transformation which produces a non-linear transform
    on x that is a function of input x.

    Used by non-zero ANODEs [1] and SONODES [2].

    Note: [3] strongly advises against using a pure nonlinear encoding/decoding
    before a NODE because it renders the NODE useless.

    References:

    [1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes."
    Advances in neural information processing systems 32 (2019).

    [2] Norcliffe, Alexander, et al. "On second order behaviour in augmented neural odes."
    Advances in neural information processing systems 33 (2020): 5911-5921.

    [3] Massaroli, Stefano, et al. "Dissecting neural odes."
    Advances in Neural Information Processing Systems 33 (2020): 3952-3963.
    """
    def __init__(self, a: int, layers: List[int]=None, activation: str='relu'):
        """Create a new MLPTransform with the configured learning model.

        Args:
            a (int): The transformed dimensionality of the state vector.
            layers (List[int], optional): Configuring MLP nodes and layers. Defaults to [32, 32].
            activation (str, optional): Activation to apply in MLP hidden layers. Defaults to 'relu'.
        """
        super().__init__(autocast=False)

        # build the nonlinear transform
        layers = [32, 32] if layers is None else layers
        self._dense_layers = []
        for nodes in layers:
            mid_layer = tf.keras.layers.Dense(nodes, activation, dtype=tf.float64)
            self._dense_layers.append(mid_layer)
        
        self._out_layer = tf.keras.layers.Dense(a, None, dtype=tf.float64)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the transformation of the input
        for NODE processing.

        Args:
            x (tf.Tensor): The input vector to transform. Shape [..., n].

        Returns:
            tf.Tensor: The learned transformation. Shape [..., a].
        """
        aug_x = x
        for layer in self._dense_layers:
            aug_x = layer(aug_x)
        return self._out_layer(aug_x)


class Truncate(tf.keras.Model):
    """A NODE decoder which truncates an augmented
    state to the first n dimensions.

    Used by non-zero ANODEs [1] and SONODES [2].

    References:

    [1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes."
    Advances in neural information processing systems 32 (2019).

    [2] Norcliffe, Alexander, et al. "On second order behaviour in augmented neural odes."
    Advances in neural information processing systems 33 (2020): 5911-5921.
    """
    def __init__(self, n: int):
        """Create a new Truncate for the specified size.

        Args:
            a (int): The number of dimensions to keep in the truncation.
        """
        super().__init__(autocast=False)
        self._n = n

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the fixed truncation of the input.

        Args:
            x (tf.Tensor): The input vector to truncate. Shape [..., a+n].

        Returns:
            tf.Tensor: The reduced decoded vector. Shape [..., n].
        """
        return x[..., :self._n]


class ZeroAugmenter(tf.keras.Model):
    """An augmenter that adds a configured number of
    zeros onto the end of a NODE starting state.

    Used by zero-based ANODEs [1].

    References:

    [1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes."
    Advances in neural information processing systems 32 (2019).
    """
    def __init__(self, n: int, a: int):
        """Create a new ZeroAugmenter with the configured dimensions.

        Args:
            n (int): The dimension of the standard input vector.
            a (int): The number of augmented dimensions to add.
        """
        super().__init__(autocast=False)

        # augmented linear map
        self._a_map = tf.linalg.diag(tf.ones(n, dtype=tf.float64), num_rows=n+a)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the augmented vector by concatening
        zeros to the end of x.

        Args:
            x (tf.Tensor): The input vector to augment. Shape [..., n].

        Returns:
            tf.Tensor: The zero-based augmentation. Shape [..., a+n].
        """
        return (self._a_map @ x[..., None])[..., 0]  # [..., n+a]
