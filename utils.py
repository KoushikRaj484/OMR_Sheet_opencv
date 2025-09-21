import cv2
import numpy as np

# ---------------- Parameters ----------------
NUM_COLUMNS = 5
MIN_R = 8
MAX_R = 28
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 18
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 28
MIN_FILL_RATIO = 0.20
MULTI_FACTOR = 0.6

# ---------------- Circle Detection ----------------
def detect_circles(gray):
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST, param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2, minRadius=MIN_R, maxRadius=MAX_R
    )
    if circles is None:
        return []
    return np.round(circles[0]).astype(int)


def make_combined_mask(img):
    b, g, r = cv2.split(img)
    avg_gr = ((g.astype(np.int16) + r.astype(np.int16)) // 2).astype(np.uint8)
    blue_diff = cv2.subtract(b, avg_gr)
    _, blue_mask = cv2.threshold(blue_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask_gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(blue_mask, mask_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    return combined


def cluster_columns(xs, K=NUM_COLUMNS):
    Z = xs.astype(np.float32).reshape(-1, 1)
    if len(Z) < K or K <= 1:
        return np.zeros(len(Z), dtype=int), np.array([0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return labels.flatten(), centers.flatten()


def group_rows_by_y(circle_list):
    if not circle_list:
        return []
    circle_list = sorted(circle_list, key=lambda c: c[1])
    ys = np.array([c[1] for c in circle_list])
    diffs = np.diff(ys)
    if len(diffs) == 0:
        splits = []
    else:
        med = np.median(diffs)
        thresh = max((med * 1.4), 8)
        splits = np.where(diffs > thresh)[0] + 1
    rows, start = [], 0
    for s in splits:
        rows.append(circle_list[start:s])
        start = s
    rows.append(circle_list[start:])
    return rows


def evaluate_group(group, mask):
    group = sorted(group, key=lambda c: c[0])
    fill_ratios = []
    for (cx, cy, r) in group:
        mm = np.zeros(mask.shape, dtype=np.uint8)
        rr = max(1, int(r * 0.75))
        cv2.circle(mm, (int(cx), int(cy)), rr, 255, -1)
        filled = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=mm))
        area = np.pi * (rr ** 2)
        ratio = (filled / area) if area > 0 else 0.0
        fill_ratios.append(ratio)

    if len(fill_ratios) == 0:
        return None, "blank", fill_ratios
    best = int(np.argmax(fill_ratios))
    best_val = fill_ratios[best]
    second = sorted(fill_ratios, reverse=True)[1] if len(fill_ratios) > 1 else 0.0

    if best_val < 0.20:
        return None, "blank", fill_ratios
    if second > 0.6 * best_val:
        return None, "multiple", fill_ratios
    return best, "selected", fill_ratios
