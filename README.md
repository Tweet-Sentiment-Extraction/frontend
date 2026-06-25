# Sentiment Span Extraction (DeBERTa QA)

**Duration:** Sep 2025 – Dec 2025  
Kaggle의 Tweet Sentiment Extraction 데이터를 사용한 프로젝트입니다.  
https://www.kaggle.com/c/tweet-sentiment-extraction  
<br>



## Overview

기존 감정 분석은 문장 전체를 긍정·부정·중립으로만 분류합니다.  
본 프로젝트는 트윗 텍스트에서 다음을 동시에 수행하는 QA 기반 모델을 구현했습니다.

- 감정 라벨 예측
- 감정 근거 span (selected_text) 추출  

<br>


## Approach

### QA 방식으로 문제 재정의

**[CLS] sentiment [SEP] tweet text [SEP]**

- Question: sentiment  
- Context: tweet text  
- Answer: 감정 근거 span  

→ start / end logits 예측 방식  

<br>


## Model

- Backbone: DeBERTa V3 Base
- Custom QA Head (start/end logits)
- Multi-Sample Dropout (5-branch)
- Stratified 5-Fold + Ensemble

**5-Fold CV Jaccard Score: 0.7198**
  
<br>


## Demo Features

- 실시간 문장 분석 (span highlight)
- CSV 일괄 감정 분석
- 감정 분포 그래프 및 Word Cloud
- 결과 CSV 다운로드
 
<br>

## 프로토타입 시연 화면

| 기능 | 시연 화면 |
| :--- | :---: |
| **메인 화면 - 실시간 트윗 분석** | 
<img width="1355" height="897" alt="스크린샷_2025-12-05_170944" src="https://github.com/user-attachments/assets/29fefc83-1006-4b5e-b6ff-82237f9a4ec8" /> |
| **데이터 분석 페이지 (CSV 업로드 분석)** | <img width="1353" height="898" alt="스크린샷_2025-12-05_172638" src="https://github.com/user-attachments/assets/ea123ba9-e295-4d62-b198-f5ec576dfa39" /><img width="1336" height="893" alt="스크린샷_2025-12-05_172717" src="https://github.com/user-attachments/assets/ca0e2308-a484-4d5f-a8a4-71e9900fd65c" /><img width="1354" height="902" alt="스크린샷_2025-12-05_171604" src="https://github.com/user-attachments/assets/204f14a1-871f-49b4-becc-a4ac44fa546c" /> |
| **트렌드 페이지 (모의 데이터 기반 시연)** | <img width="1343" height="899" alt="스크린샷_2025-12-05_171625" src="https://github.com/user-attachments/assets/af336ba0-a226-4fe3-a55e-4db8204a2c8a" /><img width="1352" height="857" alt="스크린샷_2025-12-05_171932" src="https://github.com/user-attachments/assets/c84be08f-6ede-4bb4-8648-a6ba86d6b739" /> |
  
## Key Points

- Transformer 기반 QA 모델 설계 및 파인튜닝
- Custom Loss Function 및 Span Alignment 구현
- K-Fold + Ensemble로 일반화 성능 개선
- epoch=3, max_length=128 최적화
- 모델부터 웹 서비스까지 End-to-End 구현
  
<br>

  
## Stack

Python, PyTorch, HuggingFace, Streamlit  

<br>


## Run

```bash
streamlit run app.py
