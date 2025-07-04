from __future__ import absolute_import

from tensorflow.keras import backend as K
from utils import utils


class Loss(object):
    """Abstract class for defining the loss function to be minimized.
    The loss function should be built by defining `build_loss` function.

    The attribute `name` should be defined to identify loss function with verbose outputs.
    Defaults to 'Unnamed Loss' if not overridden.
    """
    def __init__(self):
        self.name = "Unnamed Loss"

    def __str__(self):
        return self.name

    def build_loss(self):
        """Implement this function to build the loss function expression.
        Any additional arguments required to build this loss function may be passed in via `__init__`.

        Ideally, the function expression must be compatible with all keras backends and `channels_first` or
        `channels_last` image_data_format(s). `utils.slicer` can be used to define data format agnostic slices.
        (just define it in `channels_first` format, it will automatically shuffle indices for tensorflow
        which uses `channels_last` format).

        ```python
        # theano slice
        conv_layer[:, filter_idx, ...]

        # TF slice
        conv_layer[..., filter_idx]

        # Backend agnostic slice
        conv_layer[utils.slicer[:, filter_idx, ...]]
        ```

        [utils.get_img_shape](vis.utils.utils.md#get_img_shape) is another optional utility that make this easier.

        Returns:
            The loss expression.
        """
        raise NotImplementedError()


class ActivationMaximization(Loss):
    def __init__(self, layer, filter_indices):
        super(ActivationMaximization, self).__init__()
        self.name = "ActivationMax Loss"
        self.layer = layer
        self.filter_indices = utils.listify(filter_indices)

    def build_loss(self):
        layer_output = self.layer.output

        # Fallback para determinar la dimensión del tensor
        try:
            rank = K.ndim(layer_output)
        except AttributeError:
            rank = len(getattr(layer_output, 'shape', []))
        is_dense = (rank == 2)

        loss = 0.0
        for idx in self.filter_indices:
            if is_dense:
                loss += -K.mean(layer_output[:, idx])
            else:
                # Utiliza utils.slicer para compatibilidad channels_first/last
                loss += -K.mean(layer_output[utils.slicer[:, idx, ...]])

        return loss
