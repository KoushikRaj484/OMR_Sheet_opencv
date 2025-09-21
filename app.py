import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="OMR Evaluator", layout="wide")

# =========================
# Inject Custom CSS
# =========================
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        font-family: "Segoe UI", sans-serif;
    }
    .main-title {
        color: #2E86C1;
        text-align: center;
        font-size: 42px !important;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #444;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 2px dashed #bbb;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        background: #f9f9f9;
        transition: all 0.3s ease-in-out;
    }
    .upload-box:hover {
        border: 2px dashed #2E86C1;
        background: #f0f6ff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2E86C1, #1B4F72);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1B4F72, #154360);
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# OMR Utility Functions
# =========================
NUM_COLUMNS = 5
MIN_R = 8
MAX_R = 28
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 18
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 28

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
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return labels.flatten(), centers.flatten()

def group_rows_by_y(circle_list):
    if not circle_list:
        return []
    circle_list = sorted(circle_list, key=lambda c: c[1])
    ys = np.array([c[1] for c in circle_list])
    diffs = np.diff(ys)
    splits = np.where(diffs > max(np.median(diffs) * 1.4, 8))[0] + 1 if len(diffs) else []
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
    if not fill_ratios:
        return None, "blank", fill_ratios
    best = int(np.argmax(fill_ratios))
    best_val = fill_ratios[best]
    second = sorted(fill_ratios, reverse=True)[1] if len(fill_ratios) > 1 else 0.0
    if best_val < 0.20:
        return None, "blank", fill_ratios
    if second > 0.6 * best_val:
        return None, "multiple", fill_ratios
    return best, "selected", fill_ratios

def evaluate_single(img_file, sets, key_file):
    # Load and preprocess image
    img = np.array(Image.open(img_file))
    img = cv2.resize(img, (1100, 907))
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Detect circles
    circles = detect_circles(gray)
    if circles is None or len(circles) == 0:
        return None, None, None, None, None

    # Create mask
    mask = make_combined_mask(img_cv)

    # Cluster into columns
    xs = np.array([c[0] for c in circles])
    labels, centers = cluster_columns(xs)
    columns = {}
    for i, lab in enumerate(labels):
        columns.setdefault(int(lab), []).append(tuple(circles[i]))

    col_order = np.argsort(centers)
    ordered_columns = [columns.get(int(ci), []) for ci in col_order]

    # Group rows and evaluate
    qnum, results = 1, []
    for col in ordered_columns:
        rows = group_rows_by_y(col)
        for row in rows:
            if len(row) > 4:
                row = sorted(row, key=lambda c: c[0])[:4]
            sel, status, ratios = evaluate_group(row, mask)
            results.append((qnum, sel, status, ratios))
            qnum += 1

    # Build detected answers
    ans = []
    for (_, s, _, _) in results:
        if s is None or s == "blank":
            letter = "-"
        else:
            letter = "ABCD"[s]
        ans.append(letter.lower())

    # Load answer key
    try:
        df = pd.read_excel(key_file, sheet_name=sets)
    except Exception as e:
        st.error(f"❌ Failed to read key sheet '{sets}': {e}")
        return None, None, None, None, ans

    df.columns = df.columns.str.strip()

    l, c, paper = [], 0, []
    for col in df.columns:
        paper.append(col)
        count = 0
        for val in df[col]:
            if c >= len(ans):  # avoid IndexError
                break
            if isinstance(val, str) and val.strip():  # non-empty string
                if val.strip()[-1].lower() == ans[c]:
                    count += 1
            c += 1
        l.append(count)

    total = sum(l)
    return paper, l, total, cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), ans


# =========================
# Dashboard Visualization
# =========================
def display_aggregate(all_results_df):
    if all_results_df.empty:
        st.info("No results to display.")
        return

    # 🔹 Clean Filename column (remove extension like .jpg, .jpeg, .png)
    all_results_df["ID"] = all_results_df["ID"].str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)

    st.subheader("📊 Results Summary")
    st.dataframe(all_results_df, height=400, use_container_width=True)

    st.markdown("### 📌 Total Scores per Student")
    fig_bar = px.bar(all_results_df, x="ID", y="Total", text="Total",
                     color="Total", color_continuous_scale="Blues", height=400)
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🥧 Subject-Wise Score Distribution")
    subject_cols = [col for col in all_results_df.columns if col not in ["ID", "Total"]]
    if subject_cols:
        subject_totals = all_results_df[subject_cols].sum().reset_index()
        subject_totals.columns = ["Subject", "Score"]
        fig_pie = px.pie(subject_totals, names="Subject", values="Score", hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)


# =========================
# Main App (Frontend Layout)
# =========================
st.markdown('<p class="main-title">📝 Multi-File OMR Evaluator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload multiple OMR sheet images and a single Answer Key (Excel) to evaluate results.</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="upload-box">OMR Sheet Images</div>', unsafe_allow_html=True)
    img_files = st.file_uploader("Upload OMR Sheets", type=["png","jpg","jpeg"], accept_multiple_files=True, label_visibility="collapsed")
with col2:
    st.markdown('<div class="upload-box">Answer Key (Excel)</div>', unsafe_allow_html=True)
    key_file = st.file_uploader("Upload Answer Key", type=["xls","xlsx"], label_visibility="collapsed")

sheet_name = None
if key_file:
    try:
        xls = pd.ExcelFile(key_file)
        sheet_names = xls.sheet_names
        sheet_name = st.selectbox("Select Sheet from Key File", sheet_names)
    except Exception:
        st.error("❌ Failed to read Excel file.")

if st.button("Evaluate"):
    if not img_files:
        st.warning("Please upload at least one OMR sheet.")
    elif not key_file:
        st.warning("Please upload the Answer Key Excel file.")
    elif not sheet_name:
        st.warning("Please select a sheet from the Answer Key.")
    else:
        all_results = []
        for img_file in img_files:
            paper, scores, total, _, ans = evaluate_single(img_file, sheet_name, key_file)
            if paper is None:
                st.error(f"❌ No circles detected in {img_file.name}")
                continue
            row = {"ID": img_file.name}
            for p, s in zip(paper, scores):
                row[p] = s
            row["Total"] = total
            all_results.append(row)

        if all_results:
            df_all = pd.DataFrame(all_results)
            display_aggregate(df_all)
