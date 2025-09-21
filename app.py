import streamlit as st
import pandas as pd
from evaluator import evaluate_single
from dashboard import display_aggregate

st.set_page_config(page_title="OMR Evaluator", layout="wide")
st.sidebar.title("⚙️ Settings")

# ---------------- Sidebar ----------------
img_files = st.sidebar.file_uploader(
    "Upload OMR sheet images", type=["png","jpg","jpeg"], accept_multiple_files=True
)

key_file = st.sidebar.file_uploader(
    "Upload Answer Key (Excel)", type=["xls","xlsx"]
)

# If Excel key uploaded, show sheet selection
sheet_name = None
if key_file:
    try:
        xls = pd.ExcelFile(key_file)
        sheet_names = xls.sheet_names
        sheet_name = st.sidebar.selectbox("Select Sheet from Key File", sheet_names)
    except Exception as e:
        st.sidebar.error("❌ Failed to read Excel file.")

run = st.sidebar.button("Evaluate")

# ---------------- Main App ----------------
st.title("📝 Multi-File OMR Evaluator")

if run:
    if not img_files:
        st.warning("Please upload at least one OMR sheet.")
    elif not key_file:
        st.warning("Please upload the Answer Key Excel file.")
    elif not sheet_name:
        st.warning("Please select a sheet from the Answer Key.")
    else:
        all_results = []
        for img_file in img_files:
            paper, scores, total, out_img, ans = evaluate_single(img_file, sheet_name, key_file)
            if paper is None:
                st.error(f"❌ No circles detected in {img_file.name}")
                continue

            row = {"Filename": img_file.name}
            for p, s in zip(paper, scores):
                row[p] = s
            row["Total"] = total
            all_results.append(row)

        if all_results:
            df_all = pd.DataFrame(all_results)
            display_aggregate(df_all)
