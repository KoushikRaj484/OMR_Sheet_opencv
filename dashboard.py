import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def display_aggregate(all_results_df):
   #  st.header("📊 OMR Results Dashboard (Clean View)")

    if all_results_df.empty:
        st.info("No results to display.")
        return

    # --- Clean DataFrame ---
    numeric_cols = all_results_df.select_dtypes(include="number").columns

    def highlight_scores(val):
        if val == all_results_df[numeric_cols].max().max():
            color = "background-color: #85C1E9; font-weight: bold"
        elif val == all_results_df[numeric_cols].min().min():
            color = "background-color: #F1948A; font-weight: bold"
        else:
            color = ""
        return color

    styled_df = all_results_df.style.format({col: "{:.0f}" for col in numeric_cols}) \
        .applymap(highlight_scores, subset=numeric_cols) \
        .set_properties(**{"text-align": "center"}) \
        .set_table_styles([{"selector": "th", "props": [("text-align", "center"), 
                                                        ("background-color", "#2E86C1"), 
                                                        ("color", "white")]}])
    st.dataframe(styled_df, height=400, width=1200)

    st.markdown("### 📌 Total Scores per Student")
    fig_bar = px.bar(
        all_results_df,
        x="Filename",
        y="Total",
        text="Total",
        color="Total",
        color_continuous_scale="Blues",
        height=400
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        xaxis_title="OMR Sheet",
        yaxis_title="Total Score",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Visualization 2: Pie Chart (Subject-Wise Score Distribution) ---
    st.markdown("### 🥧 Subject-Wise Score Distribution")
    subject_cols = [col for col in numeric_cols if col != "Total"]
    if subject_cols:
        subject_totals = all_results_df[subject_cols].sum().reset_index()
        subject_totals.columns = ["Subject", "Score"]

        fig_pie = px.pie(
            subject_totals,
            names="Subject",
            values="Score",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.3
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    