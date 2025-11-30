import os
import streamlit as st
import pandas as pd
import altair as alt
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)
import torch
import torch.nn.functional as F

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================================
# 1. 기본 설정
# =========================================
st.set_page_config(
    page_title="ZARA Review Sentiment Span Analyzer",
    page_icon="👗",
    layout="wide",
)

st.title("👗 ZARA 리뷰 감정 구문 분석")
st.markdown(
    """
ZARA 상품 리뷰에서 **긍정 / 중립 / 부정 감정**을 예측하고,  
그 감정을 판단한 **핵심 스팬(span)** 을 추출하는 데모입니다.
"""
)

# =========================================
# 2. 모델 로드
#    - QA: frontend/model_saved (DeBERTa QA)
#    - Classifier: cardiffnlp/twitter-roberta-base-sentiment (3-class)
# =========================================

# --- 2-1. QA 스팬 추출 모델 (당신이 만든 DeBERTa QA) ---
QA_MODEL_PATH = "./model_saved"  # frontend/model_saved


@st.cache_resource(show_spinner="DeBERTa QA 모델을 로드하는 중입니다...")
def load_qa_model(model_path: str):
    abs_path = os.path.abspath(model_path)

    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"QA 모델 디렉터리를 찾을 수 없습니다: {abs_path}")

    qa_tokenizer = AutoTokenizer.from_pretrained(
        abs_path,
        local_files_only=True,
    )
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        abs_path,
        local_files_only=True,
    )
    qa_model.eval()
    return qa_tokenizer, qa_model


# --- 2-2. 감정 분류용 classifier (간단한 3-class 모델) ---
CLS_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


@st.cache_resource(show_spinner="감정 분류 모델을 로드하는 중입니다...")
def load_classifier():
    cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
    cls_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME)
    cls_model.eval()
    return cls_tokenizer, cls_model


# --- 실제 모델 로딩 ---
try:
    qa_tokenizer, qa_model = load_qa_model(QA_MODEL_PATH)
except Exception as e:
    st.error(
        "❌ QA 모델(DeBERTa)을 불러오지 못했습니다.\n\n"
        f"- 찾으려는 경로: `{os.path.abspath(QA_MODEL_PATH)}`\n"
        f"- 오류 메시지: `{e}`\n\n"
        "model_saved 폴더 위치와 내용(config.json, model.safetensors 등)을 다시 확인해 주세요."
    )
    st.stop()

try:
    cls_tokenizer, cls_model = load_classifier()
except Exception as e:
    st.error(
        "❌ 감정 분류용 모델을 불러오지 못했습니다.\n\n"
        f"- 모델 이름: `{CLS_MODEL_NAME}`\n"
        f"- 오류 메시지: `{e}`\n\n"
        "인터넷 연결 또는 HuggingFace cache 설정을 확인해 주세요."
    )
    st.stop()


# =========================================
# 3. 감정 분류 함수 (classifier)
# =========================================
def classify_sentiment(text: str):
    """
    간단한 3-class sentiment classifier.
    cardiffnlp/twitter-roberta-base-sentiment 기준:
    LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive
    """
    inputs = cls_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = cls_model(**inputs).logits  # (1, 3)

    probs = F.softmax(logits, dim=-1)[0].tolist()
    label_names = ["negative", "neutral", "positive"]
    prob_dict = {
        "negative": probs[0],
        "neutral": probs[1],
        "positive": probs[2],
    }

    pred_idx = int(torch.argmax(logits, dim=-1).item())
    pred_label = label_names[pred_idx]

    return pred_label, prob_dict


# =========================================
# 4. 스팬 추출 함수 (QA DeBERTa)
# =========================================
def extract_span_with_qa(sentiment_label: str, review_text: str):
    """
    Tweet Sentiment Extraction 포맷처럼,
    question = sentiment_label, context = review_text 로 넣어서
    해당 감정에 대한 근거 스팬을 추출
    """
    inputs = qa_tokenizer(
        sentiment_label,
        review_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_probs = F.softmax(outputs.start_logits, dim=-1)
    end_probs = F.softmax(outputs.end_logits, dim=-1)

    start_idx = int(torch.argmax(start_probs))
    end_idx = int(torch.argmax(end_probs))
    if end_idx < start_idx:
        end_idx = start_idx

    span = qa_tokenizer.decode(
        inputs["input_ids"][0][start_idx : end_idx + 1],
        skip_special_tokens=True,
    ).strip()

    return span


# =========================================
# 5. 워드클라우드 생성 함수
# =========================================
def make_wordcloud(texts, title: str):
    """
    texts: 리스트[str] 형태 (스팬들)
    """
    all_text = " ".join(t for t in texts if isinstance(t, str) and t.strip())
    if not all_text.strip():
        st.info(f"'{title}' 감정으로 분류된 리뷰/스팬이 충분하지 않아 워드클라우드를 만들 수 없습니다.")
        return

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)


# =========================================
# 6. 탭 구성: (1) 단일 문장 분석, (2) CSV 업로드 분석
# =========================================
tab_single, tab_csv = st.tabs(["✨ 단일 문장 분석", "📂 CSV 업로드 & 워드클라우드"])

# -----------------------------
# 탭 1: 한 문장 분석
# -----------------------------
with tab_single:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("📝 리뷰 입력")
        review_text = st.text_area(
            "분석할 ZARA 리뷰를 영어로 입력해 주세요:",
            height=200,
            placeholder=(
                "예시)\n"
                "The fabric feels cheap and the sizing is inconsistent, "
                "but I love the color and fit overall."
            ),
        )
        analyze_btn = st.button("AI로 분석하기", type="primary", use_container_width=True)

    with col_result:
        if analyze_btn and review_text.strip():
            text_clean = review_text.strip()

            # 1) 감정 분류
            with st.spinner("감정을 분류하는 중입니다..."):
                pred_sentiment, prob_dict = classify_sentiment(text_clean)

            # 2) 스팬 추출
            with st.spinner("해당 감정에 대한 스팬을 추출하는 중입니다..."):
                predicted_span = extract_span_with_qa(pred_sentiment, text_clean)

            # 결과 헤더
            st.subheader("📌 분석 결과")

            emoji_map = {
                "positive": "🥰 긍정",
                "neutral": "😐 중립",
                "negative": "😡 부정",
            }
            st.success(
                f"예측 감정: **{emoji_map[pred_sentiment]}**  "
            )

            # 하이라이트
            st.markdown("#### 🔍 리뷰에서 감정 스팬 하이라이트 (DeBERTa QA)")

            lower_review = text_clean.lower()
            lower_span = predicted_span.lower()

            if lower_span and lower_span in lower_review:
                start = lower_review.find(lower_span)
                end = start + len(predicted_span)
                orig_span = text_clean[start:end]
                highlighted = (
                    text_clean[:start]
                    + f"**:red[{orig_span}]**"
                    + text_clean[end:]
                )
                st.markdown(f"> {highlighted}")
            else:
                st.markdown(f"> 전체 리뷰: {text_clean}")
                st.markdown(f"> 추출된 스팬: **{predicted_span}**")

            st.divider()

            # 도넛 차트
            st.markdown("#### 📊 감정별 확률 분포 (Classifier 기준)")

            df = pd.DataFrame(
                [
                    {"sentiment": "positive", "probability": prob_dict["positive"]},
                    {"sentiment": "neutral", "probability": prob_dict["neutral"]},
                    {"sentiment": "negative", "probability": prob_dict["negative"]},
                ]
            )

            base = alt.Chart(df).encode(
                theta=alt.Theta("probability", stack=True),
            )

            pie = base.mark_arc(innerRadius=60).encode(
                color=alt.Color(
                    "sentiment",
                    scale=alt.Scale(
                        domain=["positive", "neutral", "negative"],
                        range=["#00BF63", "#FFDE59", "#EB1D26"],
                    ),
                    legend=alt.Legend(title="감정"),
                ),
                order=alt.Order("probability", sort="descending"),
                tooltip=[
                    "sentiment",
                    alt.Tooltip("probability", format=".1%"),
                ],
            )

            text = base.mark_text(
                radius=120,
                fontSize=16,
                fontWeight="bold",
            ).encode(
                text=alt.Text("probability", format=".1%"),
                order=alt.Order("probability", sort="descending"),
                color=alt.value("black"),
            )

            st.altair_chart(pie + text, use_container_width=True)

            st.markdown("#### 📄 감정별 확률 (표)")
            st.dataframe(
                df.assign(probability=lambda x: x["probability"].map(lambda v: f"{v:.3f}")),
                use_container_width=True,
            )

        elif analyze_btn and not review_text.strip():
            st.warning("먼저 ZARA 리뷰 텍스트를 입력해 주세요.")


# -----------------------------
# 탭 2: CSV 업로드 & 워드클라우드
# -----------------------------
with tab_csv:
    st.subheader("📂 CSV 파일 업로드")

    uploaded_file = st.file_uploader("ZARA 리뷰 CSV 파일을 업로드해 주세요 (.csv)", type=["csv"])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)

        st.markdown("#### 업로드된 데이터 미리보기")
        st.dataframe(df_raw.head(), use_container_width=True)

        # 텍스트 컬럼 선택
        text_col = st.selectbox(
            "리뷰 텍스트가 들어 있는 컬럼을 선택하세요:",
            options=list(df_raw.columns),
            index=list(df_raw.columns).index("text") if "text" in df_raw.columns else 0,
        )

        max_rows = st.slider( 
            "최대 몇 개의 리뷰를 분석할까요?",
            min_value=1,
            max_value=min(30000, len(df_raw)),
            value=min(300, len(df_raw)),
            step=50,
        )

        if st.button("CSV 분석 시작", type="primary", use_container_width=True):
            df = df_raw.copy()
            df = df[df[text_col].notna()].reset_index(drop=True)
            df = df.head(max_rows)

            st.info(f"{len(df)}개의 리뷰를 분석합니다.")

            sentiments = []
            spans = []

            with st.spinner("리뷰들을 분석하는 중입니다... (classifier + QA)"):
                for _, row in df.iterrows():
                    text_val = str(row[text_col])
                    pred_sent, _prob = classify_sentiment(text_val)
                    span = extract_span_with_qa(pred_sent, text_val)
                    sentiments.append(pred_sent)
                    spans.append(span)

            df["pred_sentiment"] = sentiments
            df["span"] = spans

            st.markdown("#### 분석 결과 미리보기")
            st.dataframe(df[[text_col, "pred_sentiment", "span"]].head(20), use_container_width=True)

            st.divider()

            # 감정별 개수
            st.markdown("#### 감정별 리뷰 개수")
            count_df = df["pred_sentiment"].value_counts().reindex(
                ["positive", "neutral", "negative"]
            ).fillna(0).astype(int).reset_index()
            count_df.columns = ["sentiment", "count"]
            st.dataframe(count_df, use_container_width=True)

            # 워드클라우드: 감정별 span 기반
            st.markdown("#### 🎨 감정별 워드클라우드 (QA 스팬 기반)")

            wc_col1, wc_col2, wc_col3 = st.columns(3)
            with wc_col1:
                st.markdown("**Positive**")
                make_wordcloud(df[df["pred_sentiment"] == "positive"]["span"].tolist(), "Positive")

            with wc_col2:
                st.markdown("**Neutral**")
                make_wordcloud(df[df["pred_sentiment"] == "neutral"]["span"].tolist(), "Neutral")

            with wc_col3:
                st.markdown("**Negative**")
                make_wordcloud(df[df["pred_sentiment"] == "negative"]["span"].tolist(), "Negative")

            st.markdown("#### ⬇️ 전체 결과 다운로드용 CSV")
            st.download_button(
                label="분석 결과 CSV 다운로드",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="zara_reviews_with_sentiment_and_span.csv",
                mime="text/csv",
            )
    else:
        st.info("CSV 파일을 업로드하면, 감정별 워드클라우드를 생성할 수 있습니다.")
