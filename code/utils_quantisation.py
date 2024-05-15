from typing import List, cast
import numpy as np
import numpy.typing as npt

# Type aliases for clarity
NDArrayFloat = npt.NDArray[np.float_]
NDArrayInt = npt.NDArray[np.int64]

# Constants
N = 2**50  # Large number, used for...
M = 2**59  # Modulus for modular arithmetic
CR = 2.0   # Clipping range

modulus = np.vectorize(lambda x : x % M)

def add_mod(x: NDArrayInt, y: NDArrayInt) -> NDArrayInt:
    """
    Performs modular addition on two integer numpy arrays with a predefined modulus.

    Parameters:
        x : NDArrayInt
            The first integer numpy array.
        y : NDArrayInt
            The second integer numpy array.

    Returns:
        NDArrayInt
            The result of modular addition of `x` and `y`.
    """
    return (x + y) % M




def _stochastic_round(arr: NDArrayFloat) -> NDArrayInt:
    """
    Applies stochastic rounding to a float numpy array, converting it to an integer numpy array.

    Parameters:
        arr : NDArrayFloat
            The float numpy array to be stochastically rounded.

    Returns:
        NDArrayInt
            The stochastically rounded integer numpy array.
    """
    ret: NDArrayInt = np.ceil(arr).astype(np.int64)
    rand_arr = np.random.rand(*ret.shape)
    ret[rand_arr < ret - arr] -= 1    
    return ret



def quantize(parameters: List[NDArrayFloat], clipping_range: float, target_range: np.int64) -> List[NDArrayInt]:
    """
    Quantizes a list of float numpy arrays into integer numpy arrays within a specified target range.

    Parameters:
        parameters : List[NDArrayFloat]
            The list of float numpy arrays to be quantized.
        clipping_range : float
            The range within which values are clipped before quantization.
        target_range : np.int64
            The integer range to quantize the values into.

    Returns:
        List[NDArrayInt]
            The list of quantized integer numpy arrays.
    """
    quantized_list: List[NDArrayInt] = []
    quantizer = target_range / (2 * clipping_range)
    for arr in parameters:
        pre_quantized = (np.clip(arr, -clipping_range, clipping_range) + clipping_range) * quantizer
        quantized = _stochastic_round(pre_quantized)
        quantized_list.append(quantized)
    return quantized_list

def dequantize(quantized_parameters: List[NDArrayInt], clipping_range: float, target_range: np.int64, ag_no: int) -> List[NDArrayFloat]:
    """
    Dequantizes a list of integer numpy arrays back into float numpy arrays, adjusting for a specified clipping range.

    Parameters:
        quantized_parameters : List[NDArrayInt]
            The list of quantized integer numpy arrays to be dequantized.
        clipping_range : float
            The clipping range to adjust the dequantized values within.
        target_range : np.int64
            The original target range used for quantization.
        ag_no : int
            A factor to adjust the dequantization process, potentially representing the number of aggregating agents.

    Returns:
        List[NDArrayFloat]
            The list of dequantized float numpy arrays.
    """
    reverse_quantized_list: List[NDArrayFloat] = []
    quantizer = (2 * clipping_range) / target_range
    shift = -clipping_range * ag_no
    for arr in quantized_parameters:
        recon_arr = arr.astype(np.float_) * quantizer + shift
        reverse_quantized_list.append(recon_arr)
    return reverse_quantized_list


def dequantize_mean(
    quantized_parameters: List[NDArrayInt],
    clipping_range: float,
    target_range: np.int64,
    ag_no: int,
) -> List[NDArrayFloat]:
    """
    Dequantizes integer Numpy arrays back to float Numpy arrays with an adjusted range based on
    clipping range, target range, and the number of agents.

    This function reverses the quantization process applied to the original float arrays, taking into account
    the clipping range, target range, and an adjustment factor derived from the number of agents. It's used
    to restore the approximate original float values from their quantized integer representations, applying
    an average based on the number of agents to adjust the quantization scale.

    Parameters:
        quantized_parameters : List[NDArrayInt]
            A list of quantized integer Numpy arrays to be dequantized.
        clipping_range : float
            The maximum absolute value allowed in the original float arrays.
        target_range : np.int64
            The target range used during the quantization process.
        ag_no : int
            The number of agents involved, which affects the dequantization scale.

    Returns:
        List[NDArrayFloat]
            A list of dequantized float Numpy arrays, with values approximately restored to
            their original scale and centered around the original clipping range.

    Example:
        >>> quantized_parameters = [np.array([1000, -1000, 2000], dtype=np.int64)]
        >>> clipping_range = 1.0
        >>> target_range = 10000
        >>> ag_no = 5
        >>> dequantized = dequantize_mean(quantized_parameters, clipping_range, target_range, ag_no)
        >>> print(dequantized)
        [array([-0.2, 0.2, -0.4], dtype=float64)]
    """
    reverse_quantized_list: List[NDArrayFloat] = []
    quantizer = (2 * clipping_range) / (target_range * ag_no)
    shift = -clipping_range
    for arr in quantized_parameters:
        # Convert integer array to float and apply dequantization formula
        recon_arr = (arr.astype(np.float_) * quantizer) + shift
        reverse_quantized_list.append(recon_arr)
    return reverse_quantized_list