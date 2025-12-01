import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.title("🔍 탐색 페이지 (WordCloud + 감정 분포)")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 데이터 미리보기")
    st.dataframe(df.head())

    # ===== 1) WordCloud 생성 =====
    st.subheader("☁ Word Cloud")

    # 컬럼 존재 여부 체크
    if "text" not in df.columns:
        st.error("❌ CSV에 'text' 컬럼이 없습니다.")
        st.stop()

    # NaN 제거 + 문자열 변환
    df["text"] = df["text"].fillna("").astype(str)

    # 텍스트 전체 합치기
    text_data = " ".join(df["text"]).strip()

    if len(text_data.split()) == 0:
        st.warning("⚠ WordCloud를 생성할 단어가 없습니다. 텍스트 컬럼 내용을 확인하세요.")
    else:
        # WordCloud 생성
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # ===== 2) 감정 분포 =====
    st.subheader("📊 감정 분포 그래프")

    if "sentiment" not in df.columns:
        st.error("❌ CSV에 'sentiment' 컬럼이 없습니다.")
        st.stop()

    sentiment_counts = df["sentiment"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(sentiment_counts.index, sentiment_counts.values)
    ax2.set_xlabel("Sentiment")
    ax2.set_ylabel("Count")
    ax2.set_title("Sentiment Distribution")

    st.pyplot(fig2)

else:
    st.info("CSV 파일을 업로드하면 WordCloud와 감정 분포를 볼 수 있습니다.")
