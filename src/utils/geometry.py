import math


def compute_centroid(bbox: list) -> tuple[float, float]:
    """
    Compute the centroid of a bounding box in [x1, y1, x2, y2] format.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean_distance(p1: tuple | None, p2: tuple | None) -> float:
    """
    Compute the Euclidean distance between two points.
    """
    if p1 is None or p2 is None:
        return float('inf')
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
