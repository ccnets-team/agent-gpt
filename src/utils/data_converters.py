# utils/data_converters.py
import numpy as np

"""
This module provides utility functions for converting nested data structures
between Python lists and NumPy arrays, as well as handling NaN/Inf values.

Because HTTP and JSON serialization often require basic Python types (lists,
dicts, floats), you may need to transform nested NumPy arrays or special float
values (NaN, Inf) into more standard representations. Conversely, when receiving
JSON data, you might want to restore it back into NumPy arrays for faster
numeric processing.

Functions:
  1) convert_nested_lists_to_ndarrays(data, dtype):
       Recursively converts lists (and nested dicts/tuples) to NumPy arrays,
       preserving the structure and replacing None with a placeholder if needed.

  2) convert_ndarrays_to_nested_lists(data):
       Recursively converts NumPy arrays back to lists (and preserves 
       dict/tuple structure), ensuring a JSON-friendly format.

  3) replace_nans_infs(obj):
       Recursively scans for NaN or ±Inf float values and replaces them
       with the strings "NaN", "Infinity", or "-Infinity".
"""

def convert_nested_lists_to_ndarrays(data, dtype):
    """
    Recursively converts all lists in a nested structure (dict, list, tuple) to
    NumPy arrays while preserving the original structure. Handles None values gracefully.

    Args:
        data: The input data, which can be a dict, list, tuple, or other types.
        dtype: The desired NumPy dtype for the arrays.

    Returns:
        The data with all lists converted to NumPy arrays where applicable.
    """
    if isinstance(data, list):
        if all(item is not None for item in data):
            return np.array([convert_nested_lists_to_ndarrays(item, dtype) for item in data], dtype=dtype)
        else:
            return [convert_nested_lists_to_ndarrays(item, dtype) if item is not None else None for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_nested_lists_to_ndarrays(item, dtype) for item in data)
    elif isinstance(data, dict):
        return {key: convert_nested_lists_to_ndarrays(value, dtype) for key, value in data.items()}
    else:
        return data

def convert_ndarrays_to_nested_lists(data):
    """
    Recursively converts all NumPy arrays in a nested structure (dict, list, tuple)
    to Python lists while preserving the original structure.

    Args:
        data: The input data, which can be a dict, list, tuple, or np.ndarray.

    Returns:
        The data with all NumPy arrays converted to Python lists.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_ndarrays_to_nested_lists(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_ndarrays_to_nested_lists(item) for item in data)
    elif isinstance(data, dict):
        return {key: convert_ndarrays_to_nested_lists(value) for key, value in data.items()}
    else:
        return data

def replace_nans_infs(obj):
    """
    Recursively converts NaN/Inf floats in a nested structure
    (lists, tuples, dicts) into strings: "NaN", "Infinity", "-Infinity".
    """
    if isinstance(obj, list):
        return [replace_nans_infs(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(replace_nans_infs(v) for v in obj)
    elif isinstance(obj, dict):
        return {k: replace_nans_infs(v) for k, v in obj.items()}

    # Check if it's a float or np.floating
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return "NaN"
        elif np.isposinf(obj):
            return "Infinity"
        elif np.isneginf(obj):
            return "-Infinity"
        else:
            return float(obj)  # Return as a normal float if it's finite

    # For everything else, return as is
    return obj