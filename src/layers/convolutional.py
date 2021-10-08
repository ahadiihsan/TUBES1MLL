from __future__ import annotations


try:
    from cs231n.fast_conv_cython import col2im_cython, im2col_cython
except ImportError:
    print('run python setup.py build_ext --inplace')



from typing import Tuple, Optional

import numpy as np

from base import Layer
from errors import InvalidPaddingModeError
from cs231n.fast_conv import im2col, col2im


class ConvLayer2D(Layer):

    def __init__(
        self, w: np.array,
        b: np.array,
        padding: str = 'valid',
        stride: int = 1
    ):
        self._w, self._b = w, b
        self._padding = padding
        self._stride = stride
        self._k = None
        self._dw, self._db = None, None
        self._a_prev = None

    @classmethod
    def initialize(
        cls, filters: int,
        kernel_shape: Tuple[int, int, int],
        padding: str = 'valid',
        stride: int = 1
    ) -> ConvLayer2D:
        w = np.random.randn(*kernel_shape, filters) * 0.1
        b = np.random.randn(filters) * 0.1
        return cls(w=w, b=b, padding=padding, stride=stride)

    @classmethod
    def get_kernel_shape(self) :
        print("test")
        return self.w.shape

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        return self._w, self._b

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        if self._dw is None or self._db is None:
            return None
        return self._dw, self._db

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._a_prev = np.array(a_prev, copy=True)
        output_shape = self.calculate_output_dims(input_dims=a_prev.shape)
        n, h_in, w_in, _ = a_prev.shape
        _, h_out, w_out, _ = output_shape
        h_f, w_f, _, n_f = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        output = np.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f

                output[:, i, j, :] = np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    self._w[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )

        return output + self._b

    def backward_pass(self, da_curr: np.array) -> np.array:
        _, h_out, w_out, _ = da_curr.shape
        n, h_in, w_in, _ = self._a_prev.shape
        h_f, w_f, _, _ = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=self._a_prev, pad=pad)
        output = np.zeros_like(a_prev_pad)

        self._db = da_curr.sum(axis=(0, 1, 2)) / n
        self._dw = np.zeros_like(self._w)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self._w[np.newaxis, :, :, :, :] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
                self._dw += np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )

        self._dw /= n
        return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :]

    def set_wights(self, w: np.array, b: np.array) -> None:
        self._w = w
        self._b = b

    def calculate_output_dims(
        self, input_dims: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        n, h_in, w_in, _ = input_dims
        h_f, w_f, _, n_f = self._w.shape
        if self._padding == 'same':
            return n, h_in, w_in, n_f
        elif self._padding == 'valid':
            h_out = (h_in - h_f) // self._stride + 1
            w_out = (w_in - w_f) // self._stride + 1
            return n, h_out, w_out, n_f
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self._padding}"
            )

    def calculate_pad_dims(self) -> Tuple[int, int]:
        if self._padding == 'same':
            h_f, w_f, _, _ = self._w.shape
            return (h_f - 1) // 2, (w_f - 1) // 2
        elif self._padding == 'valid':
            return 0, 0
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self._padding}"
            )

    @staticmethod
    def pad(array: np.array, pad: Tuple[int, int]) -> np.array:
        return np.pad(
            array=array,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant'
        )


class FastConvLayer2D(ConvLayer2D):

    def __init__(
        self, w: np.array,
        b: np.array,
        padding: str = 'valid',
        stride: int = 1
    ):
        super(FastConvLayer2D, self).__init__(
            w=w, b=b, padding=padding, stride=stride
        )
        self._cols = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._a_prev = np.array(a_prev, copy=True)
        n, h_out, w_out, _ = self.calculate_output_dims(input_dims=a_prev.shape)
        h_f, w_f, _, n_f = self._w.shape
        pad = self.calculate_pad_dims()
        w = np.transpose(self._w, (3, 2, 0, 1))

        self._cols = im2col(
            array=np.moveaxis(a_prev, -1, 1),
            filter_dim=(h_f, w_f),
            pad=pad[0],
            stride=self._stride
        )

        result = w.reshape((n_f, -1)).dot(self._cols)
        output = result.reshape(n_f, h_out, w_out, n)

        return output.transpose(3, 1, 2, 0) + self._b

    def backward_pass(self, da_curr: np.array) -> np.array:
        n, h_out, w_out, _ = self.calculate_output_dims(
            input_dims=self._a_prev.shape)
        h_f, w_f, _, n_f = self._w.shape
        pad = self.calculate_pad_dims()

        self._db = da_curr.sum(axis=(0, 1, 2)) / n
        da_curr_reshaped = da_curr.transpose(3, 1, 2, 0).reshape(n_f, -1)

        w = np.transpose(self._w, (3, 2, 0, 1))
        dw = da_curr_reshaped.dot(self._cols.T).reshape(w.shape)
        self._dw = np.transpose(dw, (2, 3, 1, 0))

        output_cols = w.reshape(n_f, -1).T.dot(da_curr_reshaped)

        output = col2im(
            cols=output_cols,
            array_shape=np.moveaxis(self._a_prev, -1, 1).shape,
            filter_dim=(h_f, w_f),
            pad=pad[0],
            stride=self._stride
        )
        return np.transpose(output, (0, 2, 3, 1))


class SuperFastConvLayer2D(ConvLayer2D):

    def __init__(
            self, w: np.array,
            b: np.array,
            padding: str = 'valid',
            stride: int = 1
    ):
        super(SuperFastConvLayer2D, self).__init__(
            w=w, b=b, padding=padding, stride=stride
        )
        self._cols = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._a_prev = np.array(a_prev, copy=True)
        n, h_out, w_out, _ = self.calculate_output_dims(input_dims=a_prev.shape)
        h_f, w_f, _, n_f = self._w.shape
        pad = self.calculate_pad_dims()
        w = np.transpose(self._w, (3, 2, 0, 1))

        self._cols = im2col_cython(
            np.moveaxis(a_prev, -1, 1),
            h_f,
            w_f,
            pad[0],
            self._stride
        )

        result = w.reshape((n_f, -1)).dot(self._cols)
        output = result.reshape(n_f, h_out, w_out, n)

        return output.transpose(3, 1, 2, 0) + self._b

    def backward_pass(self, da_curr: np.array) -> np.array:
        n, h_out, w_out, _ = self.calculate_output_dims(
            input_dims=self._a_prev.shape)
        h_f, w_f, _, n_f = self._w.shape
        pad = self.calculate_pad_dims()

        self._db = da_curr.sum(axis=(0, 1, 2)) / n
        da_curr_reshaped = da_curr.transpose(3, 1, 2, 0).reshape(n_f, -1)

        w = np.transpose(self._w, (3, 2, 0, 1))
        dw = da_curr_reshaped.dot(self._cols.T).reshape(w.shape)
        self._dw = np.transpose(dw, (2, 3, 1, 0))

        output_cols = w.reshape(n_f, -1).T.dot(da_curr_reshaped)

        a_prev = np.moveaxis(self._a_prev, -1, 1)
        output = col2im_cython(
            output_cols,
            a_prev.shape[0],
            a_prev.shape[1],
            a_prev.shape[2],
            a_prev.shape[3],
            h_f,
            w_f,
            pad[0],
            self._stride
        )
        return np.transpose(output, (0, 2, 3, 1))
