"""야간·저조도 영상 전처리."""
import cv2


def apply_clahe(frame, clip_limit: float = 2.0, tile: int = 8):
    """LAB 색공간의 밝기(L) 채널에만 CLAHE를 적용해 색은 유지하고 대비를 높인다."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
