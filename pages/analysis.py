# pages/analysis.py
import streamlit as st
import pandas as pd
from utils.extract import analyze_text
import altair as alt

st.title("📊 데이터 분석 페이지")

uploaded = st.file_uploader("CSV 업로드 (text 컬럼 필요)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("text 컬럼이 없습니다.")
    else:
        st.success("CSV 분석 진행 중...")
        results = [analyze_text(t)[0]['sentiment'] for t in df['text']]
        df['sentiment'] = results

        chart = alt.Chart(df).mark_bar().encode(x='sentiment', y='count()')
        st.altair_chart(chart, use_container_width=True)
