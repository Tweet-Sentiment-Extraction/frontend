import streamlit as st
import pandas as pd
import altair as alt
import torch
import torch.nn.functional as F
import transformers
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# =========================================================
# 1. Utils 모듈 가져오기 (QA 모델 및 로직 재사용)
# =========================================================
# utils/extract.py에서 get_span 함수를 직접 가져옵니다.
try:
    from utils.extract import get_span
except ImportError:
    st.error("❌ 'utils/extract.py'를 찾을 수 없습니다. 폴더 구조를 확인해주세요.")
    st.stop()

# =========================================================
# 2. 감정 분류 모델 로드 (RoBERTa)
# =========================================================
@st.cache_resource
def load_sentiment_classifier():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    except AttributeError:
        tokenizer = transformers.models.auto.AutoTokenizer.from_pretrained(model_name)
        model = transformers.models.auto.AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# =========================================================
# 3. 워드클라우드 생성 헬퍼 함수
# =========================================================
def make_wordcloud(texts, title):
    # 빈 텍스트 필터링
    valid_texts = [str(t) for t in texts if str(t).strip()]
    
    if not valid_texts:
        st.info(f"📌 '{title}'에 해당하는 데이터가 없어 워드클라우드를 생성하지 않습니다.")
        return
    
    text_combined = " ".join(valid_texts)
    
    # 폰트 경로 설정이 필요하다면 font_path 인자를 추가하세요.
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_combined)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=15)
    st.pyplot(fig)

# =========================================================
# 4. 메인 UI 및 분석 로직
# =========================================================
st.set_page_config(page_title="대량 데이터 분석", page_icon="📊", layout="wide")

st.title("📊 대량 데이터 분석 및 시각화")
st.markdown("""
CSV 파일을 업로드하면 **감정 분류(RoBERTa)** 와 **감정 근거 추출(QA Model)** 을 일괄 수행합니다.
""")

# 감정 분류 모델 로드
cls_tokenizer, cls_model = load_sentiment_classifier()

# 파일 업로드
uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    
    st.markdown("### 1️⃣ 데이터 확인")
    st.dataframe(df_raw.head(), use_container_width=True)
    
    # 텍스트 컬럼 선택
    cols = list(df_raw.columns)
    default_idx = cols.index("text") if "text" in cols else 0
    text_col = st.selectbox("분석할 텍스트 컬럼을 선택하세요:", cols, index=default_idx)
    
    # 행 개수 제한
    max_rows = st.slider("분석할 최대 행 수 (처리 속도 조절)", 1, len(df_raw), min(100, len(df_raw)))
    
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        
        # 분석 대상 데이터 슬라이싱
        df_process = df_raw.iloc[:max_rows].copy()
        
        results_sentiment = []
        results_span = []
        
        st.markdown("### 2️⃣ AI 분석 진행 중...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(df_process)
        
        # --- Batch Analysis Loop ---
        for i, row in df_process.iterrows():
            text_val = str(row[text_col])
            
            # [Step 1] 감정 분류 (RoBERTa)
            inputs = cls_tokenizer(text_val, return_tensors="pt")
            with torch.no_grad():
                logits = cls_model(**inputs).logits
            scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            labels = ["negative", "neutral", "positive"]
            top_idx = np.argmax(scores)
            pred_label = labels[top_idx]
            
            # [Step 2] 근거 추출 (utils.extract.get_span 재사용)
            if pred_label == "neutral":
                # 중립은 문장 전체를 근거로 봄
                span_val = text_val
            else:
                # get_span 함수 호출 (utils/extract.py)
                # 반환값: span, prob, char_start, char_end
                extracted_span, _, _, _ = get_span(text_val, pred_label)
                span_val = extracted_span
            
            results_sentiment.append(pred_label)
            results_span.append(span_val)
            
            # 진행률 업데이트 (UI 부하 감소를 위해 5건 단위)
            if i % 5 == 0 or i == total - 1:
                progress_bar.progress((i + 1) / total)
                status_text.caption(f"{i + 1} / {total} 처리 중...")
        
        # 결과 저장
        df_process['predicted_sentiment'] = results_sentiment
        df_process['rationale_span'] = results_span
        
        progress_bar.progress(1.0)
        status_text.success("✅ 분석이 완료되었습니다!")
        
        st.divider()
        
        # --- 결과 표시 및 시각화 ---
        st.markdown("### 3️⃣ 분석 결과 요약")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📊 감정 분포 차트")
            chart = alt.Chart(df_process).mark_bar().encode(
                x=alt.X('predicted_sentiment', title='Sentiment', sort=['positive', 'neutral', 'negative']),
                y=alt.Y('count()', title='Count'),
                color=alt.Color('predicted_sentiment', 
                                scale=alt.Scale(domain=['positive', 'neutral', 'negative'], 
                                                range=['#00BF63', '#FFDE59', '#EB1D26']),
                                legend=None)
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
            
        with col2:
            st.markdown("#### 🥧 감정 비율")
            counts = df_process['predicted_sentiment'].value_counts()
            
            # 파이차트 색상 고정
            colors_map = {'positive': '#00BF63', 'neutral': '#FFDE59', 'negative': '#EB1D26'}
            pie_colors = [colors_map.get(idx, '#999999') for idx in counts.index]
            
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            ax_pie.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=pie_colors)
            st.pyplot(fig_pie)

        st.markdown("### 4️⃣ 감정별 핵심 구문 (WordCloud)")
        st.info("단순 빈도수가 아닌, **AI가 판단한 감정의 원인(Span)** 만을 모아 시각화했습니다.")
        
        wc_col1, wc_col2, wc_col3 = st.columns(3)
        
        with wc_col1:
            st.markdown("**🥰 Positive Rationale**")
            pos_data = df_process[df_process['predicted_sentiment'] == 'positive']['rationale_span'].tolist()
            make_wordcloud(pos_data, "Positive")

        with wc_col2:
            st.markdown("**😐 Neutral Context**")
            neu_data = df_process[df_process['predicted_sentiment'] == 'neutral']['rationale_span'].tolist()
            make_wordcloud(neu_data, "Neutral")

        with wc_col3:
            st.markdown("**😡 Negative Rationale**")
            neg_data = df_process[df_process['predicted_sentiment'] == 'negative']['rationale_span'].tolist()
            make_wordcloud(neg_data, "Negative")
            
        st.divider()
        
        st.markdown("### 5️⃣ 데이터 다운로드")
        st.dataframe(df_process[[text_col, 'predicted_sentiment', 'rationale_span']].head(20), use_container_width=True)
        
        csv_data = df_process.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드",
            data=csv_data,
            file_name="batch_analysis_result.csv",
            mime="text/csv"
        )

else:
    st.info("👆 위에서 CSV 파일을 업로드하면 분석이 시작됩니다.")