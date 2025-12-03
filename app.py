# StreamlitTestApp/app.py
import streamlit as st
import pandas as pd
import altair as alt
import torch
import torch.nn.functional as F
import transformers 
import traceback
import numpy as np

# ---------------------------------------------------------
# 1. QA 모델 로드 (기존 utils에서 가져오기)
# ---------------------------------------------------------
try:
    from utils.extract import tokenizer as qa_tokenizer, model as qa_model
except ImportError:
    st.error("utils.extract 모듈을 찾을 수 없습니다. 폴더 구조를 확인해주세요.")
    st.stop()

# ---------------------------------------------------------
# 2. 감정 분류 모델 로드
# ---------------------------------------------------------
@st.cache_resource
def load_sentiment_classifier():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    except AttributeError:
        # 우회 경로 시도
        tokenizer = transformers.models.auto.AutoTokenizer.from_pretrained(model_name)
        model = transformers.models.auto.AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# ---------------------------------------------------------
# 3. QA 모델을 이용한 근거 추출 함수 (Offset Mapping 적용)
# ---------------------------------------------------------
def extract_span(text, sentiment, qa_model, qa_tokenizer):
    """
    수정된 함수: sequence_ids를 사용하여 질문 영역을 정확히 마스킹하고,
    모델이 [CLS] 토큰을 선택했을 때의 예외 처리를 강화함.
    """
    qa_model.eval()

    # 1. 토크나이징 (질문=sentiment, 본문=text)
    inputs = qa_tokenizer(
        sentiment,
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
    )

    offset_mapping = inputs.pop("offset_mapping")[0]
    
    # sequence_ids를 통해 질문(0)과 본문(1)을 명확히 구분
    # None: 특수 토큰([CLS], [SEP]), 0: 감정 단어, 1: 실제 텍스트
    sequence_ids = inputs.sequence_ids(0)

    # 디바이스 이동
    device = next(qa_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = qa_model(**inputs)
        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()

    # 2. 마스킹 (Masking) - 본문이 아닌 모든 곳(질문, 특수토큰)을 -1e10으로 가림
    # 이렇게 하면 모델이 강제로 본문 안에서만 답을 찾게 됨
    for i, seq_id in enumerate(sequence_ids):
        if seq_id != 1:  # 1번 시퀀스(본문)가 아니면
            start_logits[i] = -1e10
            end_logits[i] = -1e10

    # 3. Span 예측
    start_idx = int(torch.argmax(start_logits))
    end_idx = int(torch.argmax(end_logits))

    # 역전 보정
    if end_idx < start_idx:
        end_idx = start_idx

    # 4. Offset을 이용해 실제 텍스트 위치 찾기
    try:
        char_start = int(offset_mapping[start_idx][0])
        char_end = int(offset_mapping[end_idx][1])
        
        # 모델이 여전히 확신이 없어 (0,0)이나 이상한 위치를 가리킬 경우 방어 로직
        if char_start == 0 and char_end == 0 and sequence_ids[start_idx] != 1:
             # 본문이 아님 -> 실패로 간주하지 않고 가장 확률 높은 구간 재탐색 혹은 빈값 반환
             return text, 0.0, -1, -1
             
    except Exception:
        return text, 0.0, -1, -1

    # 5. 최종 결과 반환
    span_text = text[char_start:char_end]
    
    # 확률 계산
    prob = (torch.softmax(start_logits, dim=-1)[start_idx].item()
            + torch.softmax(end_logits, dim=-1)[end_idx].item()) / 2

    return span_text, prob, char_start, char_end
# ---------------------------------------------------------
# 메인 UI 로직
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Advanced Sentiment Analysis", page_icon="🧠", layout="wide")
    
    st.title("🧠 AI 트윗 감정 분석기")
    st.markdown("QA 모델을 활용하여 감정을 자동으로 예측하고 근거 구간을 시각화합니다.")

    with st.spinner("감정 분류 모델 로딩 중..."):
        try:
            cls_tokenizer, cls_model = load_sentiment_classifier()
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {e}")
            return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 텍스트 입력")
        input_text = st.text_area("분석할 영어 트윗을 입력하세요:", height=200, placeholder="Example: I really love this movie, it's fantastic!")
        analyze_btn = st.button("AI 정밀 분석 시작", type="primary", use_container_width=True)

    with col2:
        if analyze_btn and input_text:
            with st.spinner("AI가 2단계 분석을 수행 중입니다..."):
                try:
                    # [Step 1] 감정 분류 수행
                    inputs = cls_tokenizer(input_text, return_tensors="pt")
                    with torch.no_grad():
                        logits = cls_model(**inputs).logits
                    
                    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                    labels = ["negative", "neutral", "positive"]
                    
                    cls_results = [
                        {"label": labels[i], "score": float(score)} 
                        for i, score in enumerate(scores)
                    ]
                    
                    sorted_results = sorted(cls_results, key=lambda x: x['score'], reverse=True)
                    top_result = sorted_results[0]
                    top_sentiment = top_result['label']
                    top_score = top_result['score']

                    # =========================================================
                    # [수정] 변수 정의를 여기로 위로 올립니다. (이 부분이 핵심!)
                    # =========================================================
                    emoji_map = {"positive": "🥰 긍정", "negative": "😡 부정", "neutral": "😐 중립"}
                    color_map = {"positive": "#d4edda", "negative": "#f8d7da", "neutral": "#e2e3e5"}
                    
                    # [Step 2] QA 모델로 근거 추출 (인덱스 반환)
                    # (추가) 중립 감정일 경우 전체 문장을 근거로 처리 (모델 특성 반영)
                    if top_sentiment == "neutral":
                        predicted_span = input_text
                        span_confidence = 1.0
                        char_start = 0
                        char_end = len(input_text)
                        
                        # 이제 여기서 emoji_map을 사용할 수 있습니다.
                        st.success(f"최종 판정: {emoji_map.get(top_sentiment, top_sentiment)}")
                        st.info("ℹ️ '중립(Neutral)' 감정은 특정 단어가 아니라 문장 전체의 맥락으로 판단됩니다.")
                        
                        # 하이라이트 없이 전체 텍스트 표시
                        st.markdown(f"<div style='background-color:#e2e3e5; padding:15px; border-radius:10px;'>{input_text}</div>", unsafe_allow_html=True)
                    
                    else:
                        # 긍정/부정일 때만 정밀 추출 수행
                        predicted_span, span_confidence, char_start, char_end = extract_span(input_text, top_sentiment, qa_model, qa_tokenizer)

                        st.success(f"최종 판정: {emoji_map.get(top_sentiment, top_sentiment)}")
                        st.caption(f"감정 확신도: {top_score:.1%} | 근거 추출 확신도: {span_confidence:.1%}")

                        st.markdown("#### 🔍 판단 근거 (Highlight)")
                        highlight_bg = color_map.get(top_sentiment, "#fff3cd")
                        
                        # (수정) 유효한 인덱스일 때만 하이라이트
                        if char_start != -1 and char_end != -1 and len(predicted_span.strip()) > 0:
                            part1 = input_text[:char_start]
                            part2 = input_text[char_start:char_end] 
                            part3 = input_text[char_end:]
                            
                            highlighted_html = (
                                f"{part1}"
                                f"<span style='background-color: {highlight_bg}; color: black; padding: 2px 4px; border-radius: 4px; font-weight: bold; border: 1px solid rgba(0,0,0,0.1);'>{part2}</span>"
                                f"{part3}"
                            )
                            st.markdown(f"<div style='font-size:1.1em; line-height:1.6; background-color:#f9f9f9; padding:15px; border-radius:10px;'>{highlighted_html}</div>", unsafe_allow_html=True)
                        else:
                            st.warning("뚜렷한 근거 단어를 찾기 어렵거나, 문장 전체가 감정을 내포하고 있습니다.")
                            st.write(f"전체 텍스트: {input_text}")

                    st.divider()

                    st.markdown("#### 📊 감정 확률 분포")
            
                    df_chart = pd.DataFrame(cls_results)
                    domain = ["positive", "neutral", "negative"]
                    range_ = ["#00BF63", "#FFDE59", "#EB1D26"]
                    
                    base = alt.Chart(df_chart).encode(
                        theta=alt.Theta("score", stack=True),
                    )

                    pie = base.mark_arc(innerRadius=60).encode(
                        color=alt.Color(
                            "label",
                            scale=alt.Scale(domain=domain, range=range_),
                            legend=alt.Legend(title="감정"),
                        ),
                        order=alt.Order("score", sort="descending"),
                        tooltip=["label", alt.Tooltip("score", format=".1%")]
                    )
                    
                    text = base.mark_text(radius=140).encode(
                        text=alt.Text("score", format=".1%"),
                        order=alt.Order("score", sort="descending"),
                        color=alt.value("black")
                    )

                    st.altair_chart(pie + text, use_container_width=True)

                except Exception as e:
                    st.error("분석 중 오류가 발생했습니다.")
                    st.code(traceback.format_exc())

        elif analyze_btn:
            st.warning("텍스트를 입력해주세요!")

if __name__ == "__main__":
    main()