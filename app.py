import streamlit as st
import pandas as pd
import altair as alt
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import torch.nn.functional as F

# --- 1. 기본 설정 ---
st.set_page_config(
    page_title="Auto Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
)

# --- 2. 모델 로드 ---
MODEL_PATH = "./best_tweet_qa_model_distilbert"

@st.cache_resource
def load_model():
    try:
        # local_files_only=True: 로컬 폴더에서만 모델을 찾습니다.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH, local_files_only=True)
        return tokenizer, model
    except Exception as e:
        return None, str(e)

tokenizer, model = load_model()

# --- 3. 분석 함수 (자동 예측 로직) ---
def analyze_text(text):
    sentiments = ["positive", "negative", "neutral"]
    results = []

    for sentiment in sentiments:
        # 토크나이징
        inputs = tokenizer(text, sentiment, return_tensors="pt")
        
        # DistilBERT는 token_type_ids를 사용하지 않으므로 제거
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            outputs = model(**inputs)
        
        # 1. 로짓을 확률로 변환
        start_probs = F.softmax(outputs.start_logits, dim=-1)
        end_probs = F.softmax(outputs.end_logits, dim=-1)
        
        # 2. 가장 유력한 시작/끝 위치 찾기
        start_idx = torch.argmax(start_probs)
        end_idx = torch.argmax(end_probs)
        
        # 3. 확신도(Confidence) 계산
        confidence = (start_probs[0][start_idx] * end_probs[0][end_idx]).item()
        
        # 4. 텍스트 추출
        span = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx+1]).strip()
        
        results.append({
            "sentiment": sentiment,
            "confidence": confidence,
            "span": span
        })
    
    # 확신도 내림차순 정렬
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results

# --- 4. UI 구성 ---
st.title("🤖 AI 트윗 감정 자동 분석기")
st.markdown("QA 모델을 활용하여 **감정을 자동으로 예측**하고 **근거 구간**을 시각화합니다.")

if tokenizer is None:
    st.error("모델을 불러오지 못했습니다. 폴더 위치를 확인해주세요.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("텍스트 입력")
    input_text = st.text_area("분석할 트윗을 영어로 입력하세요:", height=200, placeholder="Example: I really enjoyed the movie, but the popcorn was terrible.")
    analyze_btn = st.button("AI 분석 시작", type="primary", use_container_width=True)

with col2:
    if analyze_btn and input_text:
        with st.spinner("AI가 분석 중입니다..."):
            results = analyze_text(input_text)
            
            best_result = results[0]
            predicted_sentiment = best_result['sentiment']
            predicted_span = best_result['span']
            
            # --- 결과 표시 ---
            st.subheader("분석 결과")
            
            # 1. 감정 예측 결과
            emoji_map = {"positive": "🥰 긍정", "negative": "😡 부정", "neutral": "😐 중립"}
            st.success(f"예측된 감정: **{emoji_map[predicted_sentiment]}** (확신도: {best_result['confidence']:.1%})")
            
            # 2. 하이라이트
            st.markdown("#### 🔍 판단 근거 (Highlight)")
            if predicted_span.lower() in input_text.lower():
                start = input_text.lower().find(predicted_span.lower())
                if start != -1:
                    orig_span = input_text[start : start + len(predicted_span)]
                    highlighted = input_text.replace(orig_span, f"**:red[{orig_span}]**")
                    st.markdown(f"> {highlighted}")
                else:
                    st.markdown(f"> {input_text}")
            else:
                st.write(predicted_span)
            
            st.divider()
            
            # 3. 원형 차트 (Donut Chart)
            st.markdown("#### 📊 확률 분포")
            
            # 데이터 가공
            df = pd.DataFrame(results)
            total_conf = df['confidence'].sum()
            if total_conf == 0: total_conf = 1
            df['probability'] = df['confidence'] / total_conf
            
            # 차트 베이스 설정
            base = alt.Chart(df).encode(
                theta=alt.Theta("probability", stack=True) # 각도 설정
            )

            # 도넛 차트 (Arc)
            pie = base.mark_arc(innerRadius=60).encode(
                color=alt.Color("sentiment", 
                                scale=alt.Scale(domain=['positive', 'neutral', 'negative'], 
                                                range=['#00BF63', '#FFDE59', '#EB1D26']),
                                legend=alt.Legend(title="감정")),
                order=alt.Order("probability", sort="descending"),
                tooltip=["sentiment", alt.Tooltip("probability", format=".1%")]
            )

            # 텍스트 라벨 (퍼센트 표시)
            text = base.mark_text(radius=140, fontSize=20, fontWeight="bold").encode(
                text=alt.Text("probability", format=".1%"),
                order=alt.Order("probability", sort="descending"),
                color=alt.value("black")  # 라벨 색상
            )

            # 차트 + 텍스트 결합
            st.altair_chart(pie + text, use_container_width=True)

    elif analyze_btn and not input_text:
        st.warning("텍스트를 먼저 입력해주세요!")