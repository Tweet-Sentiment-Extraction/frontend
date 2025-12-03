import streamlit as st
import pandas as pd
import altair as alt
import time
import random
from utils.extract import analyze_text

st.title("📈 실시간 감정 트렌드 분석")

keyword = st.text_input("키워드 입력 (예: BTS, movie, iphone15)")
limit = st.slider("수집할 트윗 수 (시뮬레이션)", 10, 100, 30)
run = st.button("트렌드 분석 시작")

# 데모용 가상 트윗 생성 함수
def generate_mock_tweets(keyword, count):
    # 실제 스크래핑 대신 키워드가 포함된 문장을 생성하여 모델 테스트
    templates = [
        f"I really love {keyword}! It's amazing.",
        f"{keyword} is the worst thing ever.",
        f"Just saw {keyword}, it was okay.",
        f"Why is everyone talking about {keyword}?",
        f"Can't wait for the new {keyword} update!",
        f"Honestly, {keyword} is disappointing.",
        f"{keyword} changed my life.",
        f"Not sure how I feel about {keyword}.",
        f"{keyword} is purely a marketing gimmick.",
        f"Best experience with {keyword} so far!"
    ]
    return [random.choice(templates) for _ in range(count)]

if run and keyword:
    st.info(f"🔍 '{keyword}' 관련 트윗을 수집하는 중... (시뮬레이션)")
    
    # 스크래핑 딜레이 시뮬레이션 (UX 확인용)
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01) 
        progress_bar.progress(i + 1)
    
    # 1. 가상 데이터 생성 (snscrape 대체)
    tweets = generate_mock_tweets(keyword, limit)
    
    st.success(f"트윗 {len(tweets)}개 수집 완료!")

    # 2. 감정 분석 (app.py와 동일한 모델 사용)
    records = []
    with st.spinner("AI가 감정을 분석 중입니다..."):
        for text in tweets:
            result = analyze_text(text)[0]  # 모델 예측
            records.append({
                "text": text,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            })

    df = pd.DataFrame(records)

    # 3. 결과 시각화
    st.subheader("📊 감정 분포")
    
    # 차트 색상 매핑
    color_scale = alt.Scale(domain=['positive', 'negative', 'neutral'],
                            range=['#28a745', '#dc3545', '#6c757d'])

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("sentiment", title="감정"),
            y=alt.Y("count()", title="트윗 수"),
            color=alt.Color("sentiment", scale=color_scale, legend=None),
            tooltip=["sentiment", "count()"]
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)

    st.subheader("📄 분석된 트윗 데이터")
    st.dataframe(df, use_container_width=True)