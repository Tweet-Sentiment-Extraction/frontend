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
# 2. 모델 로드 (QA DeBERTa)
#    - QA: ./model_saved (당신이 학습한 DeBERTa QA)
# =========================================

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


CANDIDATE_LABELS = ["positive", "neutral", "negative"]


# =========================================
# 3. QA 한 개로 감정 + 스팬 동시 결정
# =========================================
def qa_predict_label_and_span(review_text: str):
    """
    하나의 QA 모델로 감정(label)과 스팬(span)을 동시에 결정하는 함수.

    - 각 감정 라벨(positive/neutral/negative)을 question으로 넣고
      review_text 를 context로 넣어 QA를 3번 수행.
    - 각 라벨마다:
        * context 영역에서만 start/end 토큰을 선택
        * 해당 span의 score = start_logit + end_logit
    - score가 가장 큰 라벨을 최종 감정으로 선택하고,
      그때의 span을 최종 스팬으로 사용.
    - label score 들에 softmax를 씌워 pseudo-probability도 함께 반환.
    """
    label2score_span = {}

    for label in CANDIDATE_LABELS:
        encoding = qa_tokenizer(
            label,
            review_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        # question / context 구분을 위한 sequence_ids
        sequence_ids = encoding.sequence_ids(0)  # list 길이 = 토큰 수

        # 일부 모델에서는 token_type_ids 미사용 → 제거
        if "token_type_ids" in encoding:
            del encoding["token_type_ids"]

        with torch.no_grad():
            outputs = qa_model(**encoding)

        start_logits = outputs.start_logits[0]  # (seq_len,)
        end_logits = outputs.end_logits[0]      # (seq_len,)

        # context(token_type == 1)에 해당하는 토큰만 허용
        # sequence_ids 내에서 1인 위치가 context
        context_mask = torch.tensor(
            [1 if s == 1 else 0 for s in sequence_ids],
            dtype=torch.bool,
        )

        # question / special token 쪽은 매우 작은 값으로 마스킹
        start_logits = start_logits.masked_fill(~context_mask, -1e9)
        end_logits = end_logits.masked_fill(~context_mask, -1e9)

        # argmax로 start/end 선택
        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))
        if end_idx < start_idx:
            end_idx = start_idx

        # span score: start_logit + end_logit
        score = start_logits[start_idx].item() + end_logits[end_idx].item()

        span = qa_tokenizer.decode(
            encoding["input_ids"][0][start_idx : end_idx + 1],
            skip_special_tokens=True,
        ).strip()

        label2score_span[label] = (score, span)

    # 가장 score가 큰 라벨 선택
    best_label, (best_score, best_span) = max(
        label2score_span.items(), key=lambda x: x[1][0]
    )

    # label score 들을 softmax로 확률처럼 normalization
    scores_tensor = torch.tensor([label2score_span[l][0] for l in CANDIDATE_LABELS])
    probs_tensor = F.softmax(scores_tensor, dim=-1)

    prob_dict = {
        label: float(probs_tensor[i].item())
        for i, label in enumerate(CANDIDATE_LABELS)
    }

    return best_label, best_span, prob_dict

# =========================================
# 4. 생략
# =========================================

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

            # 1) QA 모델로 감정 + 스팬 동시 추론
            with st.spinner("QA 모델로 감정과 스팬을 추론하는 중입니다..."):
                pred_sentiment, predicted_span, prob_dict = qa_predict_label_and_span(
                    text_clean
                )

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

            # 진행률 표시 바 생성
            progress_bar = st.progress(0.0)

            with st.spinner("리뷰들을 분석하는 중입니다... (classifier + QA)"):
                total = len(df)
                for i, (_, row) in enumerate(df.iterrows(), start=1):
                    text_val = str(row[text_col])
                    pred_label, span, _prob = qa_predict_label_and_span(text_val)

                    sentiments.append(pred_label)
                    spans.append(span)

                    # 진행률 업데이트
                    progress_bar.progress(i / total)

            # 진행률 바 제거
            progress_bar.empty()

            # 완료 알림
            st.success("✅ CSV 리뷰 분석이 완료되었습니다!")

            df["pred_sentiment"] = sentiments
            df["span"] = spans


            st.markdown("#### 분석 결과 미리보기")
            st.dataframe(df[[text_col, "pred_sentiment", "span"]].head(20), use_container_width=True)

            st.divider()

            # 감정별 개수
            st.markdown("#### 감정별 리뷰 개수")
            count_df = (
                df["pred_sentiment"]
                .value_counts()
                .reindex(["positive", "neutral", "negative"])
                .fillna(0)
                .astype(int)
                .reset_index()
            )
            count_df.columns = ["sentiment", "count"]
            st.dataframe(count_df, use_container_width=True)

            # 워드클라우드: 감정별 span 기반
            st.markdown("#### 🎨 감정별 워드클라우드 (QA 스팬 기반)")

            st.markdown("**Positive**")
            make_wordcloud(
                df[df["pred_sentiment"] == "positive"]["span"].tolist(),
                "Positive"
            )

            st.markdown("**Neutral**")
            make_wordcloud(
                df[df["pred_sentiment"] == "neutral"]["span"].tolist(),
                "Neutral"
            )

            st.markdown("**Negative**")
            make_wordcloud(
                df[df["pred_sentiment"] == "negative"]["span"].tolist(),
                "Negative"
            )

            st.markdown("#### ⬇️ 전체 결과 다운로드용 CSV")
            st.download_button(
                label="분석 결과 CSV 다운로드",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="zara_reviews_with_sentiment_and_span.csv",
                mime="text/csv",
            )

    else:
        st.info("CSV 파일을 업로드하면, 감정별 워드클라우드를 생성할 수 있습니다.")
