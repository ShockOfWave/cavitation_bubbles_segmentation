import cv2
import numpy as np


def draw_mask(frame: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a semi-transparent mask on a frame.
    mask: binary mask (0 or 1), sized to match the frame.
    color: tuple (B, G, R) - mask color.
    alpha: transparency coefficient.
    """
    mask = mask.astype(np.uint8)
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(frame, dtype=np.uint8)
    colored_mask[mask == 1] = color
    frame = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
    return frame
