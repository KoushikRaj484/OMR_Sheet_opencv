import cv2
import numpy as np
import pandas as pd
from PIL import Image
from utils import detect_circles, make_combined_mask, cluster_columns, group_rows_by_y, evaluate_group

def evaluate_single(img_file, sets, key_file):
    img = np.array(Image.open(img_file))
    img = cv2.resize(img,(1100,907))
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    circles = detect_circles(gray)
    if len(circles) == 0:
        return None, None, None, None, None

    mask = make_combined_mask(img_cv)
    xs = np.array([c[0] for c in circles])
    labels, centers = cluster_columns(xs)
    columns = {}
    for i, lab in enumerate(labels):
        columns.setdefault(int(lab), []).append(tuple(circles[i]))

    col_order = np.argsort(centers)
    ordered_columns = [columns.get(int(ci), []) for ci in col_order]

    qnum, results = 1, []
    out_img = img_cv.copy()
    for col_idx, col in enumerate(ordered_columns):
        rows = group_rows_by_y(col)
        for row in rows:
            if len(row) != 4:
                if len(row) > 4:
                    row = sorted(row, key=lambda c: c[0])[:4]
            sel, status, ratios = evaluate_group(row, mask)
            results.append((qnum, sel, status, ratios))
            qnum += 1

    ans = []
    for (q, s, stc, r) in results:
        letter = "-" if s is None else "ABCD"[s]
        ans.append(letter.lower())

    # --- Load Answer Key dynamically ---
    if str(key_file).endswith(".csv"):
        df = pd.read_csv(key_file)
        df.columns = df.columns.str.strip()
    else:
        df_dict = pd.read_excel(key_file, sheet_name=None)
        df = df_dict[sets]
        df.columns = df.columns.str.strip()

    l, c, paper = [], 0, []
    for col in df.columns:
        paper.append(col)
        count = 0
        for val in df[col]:
            if c >= len(ans):
                break
            if isinstance(val, str) and val[-1].lower() == ans[c]:
                count += 1
            c += 1
        l.append(count)

    total = sum(l)
    return paper, l, total, cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), ans
