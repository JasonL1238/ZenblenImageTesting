import numpy as np


def compute_blend_score(unblended_mask: np.ndarray, roi_mask: np.ndarray) -> float:
    """
    Returns the fraction of the ROI that is blended (0.0 = nothing blended,
    1.0 = fully blended).

    Both masks are HxW uint8 where 255 = active region.
    """
    roi_pixels = (roi_mask > 0).sum()
    if roi_pixels == 0:
        return 1.0
    unblended_in_roi = ((unblended_mask > 0) & (roi_mask > 0)).sum()
    return float(1.0 - unblended_in_roi / roi_pixels)


def overlay_mask(
    image: np.ndarray,
    unblended_mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """Semi-transparent colored overlay of unblended regions on the source image."""
    import cv2
    vis = image.copy().astype(np.float32)
    overlay = np.zeros_like(vis)
    overlay[unblended_mask > 0] = color  # BGR
    blended = cv2.addWeighted(vis, 1.0 - alpha, overlay, alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)
