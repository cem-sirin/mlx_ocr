import numpy as np
from shapely import Polygon


def polygon_to_array(polygon: Polygon) -> np.ndarray:
    """Convert a shapely polygon to a numpy array"""
    poly_arr = np.array(polygon.exterior.coords)[:-1]
    assert poly_arr.ndim == 2 and poly_arr.shape[1] == 2, "polygon array must be a 2D array of (x, y) coordinates"
    return poly_arr
