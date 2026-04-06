import cv2
import numpy as np


def draw_mask(frame: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.5) -> np.ndarray:
    """
    Накладывает полупрозрачную маску на кадр.
    mask: бинарная маска (0 или 1), размером, соответствующим кадру.
    color: кортеж (B, G, R) - цвет для маски.
    alpha: коэффициент прозрачности.
    """
    mask = mask.astype(np.uint8)
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(frame, dtype=np.uint8)
    colored_mask[mask == 1] = color
    frame = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
    return frame
