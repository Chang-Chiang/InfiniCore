from ast import List
import numpy as np
import gguf
from typing import List, Optional
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def leaky_relu(
    x: np.ndarray,
    negative_slope: float = 0.01,
):
    return np.where(x >= 0, x, negative_slope * x)


class LeakyReluTestCase(InfiniopTestCase):
    def __init__(
        self,
        x: np.ndarray,
        shape_x: List[int] | None,
        stride_x: List[int] | None,
        negative_slope: np.ndarray,
        negative_slope_stride: Optional[List[int]],
        y: np.ndarray,
        shape_y: List[int] | None,
        stride_y: List[int] | None,

    ):
        super().__init__("leaky_relu")
        self.x = x
        self.shape_x = shape_x
        self.stride_x = stride_x
        self.negative_slope = negative_slope
        self.negative_slope_stride = negative_slope_stride
        self.y = y
        self.shape_y = shape_y
        self.stride_y = stride_y


    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.shape_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.shape_x)
        if self.shape_y is not None:
            test_writer.add_array(test_writer.gguf_key("y.shape"), self.shape_y)
        if self.stride_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.stride_x))
        if self.negative_slope_stride is not None:
            test_writer.add_array(test_writer.gguf_key("negative_slope_stride.strides"), gguf_strides(*self.negative_slope_stride))
        test_writer.add_array(
            test_writer.gguf_key("y.strides"),
            gguf_strides(*self.stride_y if self.stride_y is not None else contiguous_gguf_strides(self.shape_y))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("x"), self.x, raw_dtype=np_dtype_to_ggml(self.x.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("negative_slope"),
            self.negative_slope,
            raw_dtype=np_dtype_to_ggml(self.negative_slope.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("y"), self.y, raw_dtype=np_dtype_to_ggml(self.y.dtype)
        )
        ans = leaky_relu(
            self.x.astype(np.float64),
            self.y.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("leaky_relu.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, x_stride, y_stride, negative_slope
        ((13, 4), None, None, 0.01),
        ((13, 4), (10, 1), (10, 1), 0.01),
        ((13, 4), None, None, 0.1),
        ((13, 4, 4), None, None, 0.01),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), 0.01),
        # ((13, 4, 4), (0, 4, 1), None),
        ((16, 5632), None, None, 0.01),
        ((16, 5632), (13312, 1), (13312, 1), 0.01),
        ((4, 4, 5632), None, None, 0.01),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), 0.01),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_x, stride_y in _TEST_CASES_:
            x = np.random.rand(*shape).astype(dtype)
            negative_slope = np.full(shape, 0.01, dtype=dtype)
            y = np.empty(tuple(0 for _ in shape), dtype=dtype)
            x = process_zero_stride_tensor(x, stride_x)
            test_case = LeakyReluTestCase(
                x=x,
                shape_x=shape,
                stride_x=stride_x,
                negative_slope=negative_slope,
                negative_slope_stride=stride_x,
                y=y,
                shape_y=shape,
                stride_y=stride_y,
            )
            test_cases.append(test_case)
            
    test_writer.add_tests(test_cases)
    test_writer.save()
    