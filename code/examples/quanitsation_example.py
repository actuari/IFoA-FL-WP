import numpy as np
import utils_quantisation

def generate_random_data(seed: int = 42, size: int = 4, scale: float = 3.0) -> np.ndarray:
    """
    Generate random data for demonstration purposes.

    Parameters:
        seed : int
            Seed for the random number generator to ensure reproducibility. Defaults to 42.
        size : int
            The size of the generated data array. Defaults to 4.
        scale : float
            The scale factor to adjust the generated data. Defaults to 3.0.

    Returns: 
        np.ndarray 
            A numpy array of generated data.
    """
    np.random.seed(seed)
    original_data = np.random.rand(size) - 0.5  # Generate data in the range [-0.5, 0.5]
    return np.add(original_data, scale) / (2 * scale)

def demonstrate_quantization(data: np.ndarray, scale: float, precision: int) -> None:
    """
    Demonstrate the process of quantizing and dequantizing data, displaying the original,
    quantized, and dequantized data.

    Parameters
        data : np.ndarray 
            The original data to be quantized.
        scale: float 
            The scale factor used for quantization and dequantization.
        precision: int 
            The precision (number of levels) used for quantization.
    """
    print("Original Data:")
    print(data)

    # Quantize the data
    quantized_data = utils_quantisation.quantize([data], scale, precision)
    print("\nQuantized Data:")
    print(quantized_data)

    # Dequantize the data
    dequantized_data = utils_quantisation.dequantize(quantized_data, scale, precision, 1)
    print("\nDequantized Data:")
    print(dequantized_data)

if __name__ == "__main__":
    # Example usage:
    original_data = generate_random_data(size=4, scale=3.0)
    demonstrate_quantization(original_data, scale=3.0, precision=2**20)