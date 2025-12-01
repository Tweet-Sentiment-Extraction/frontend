import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_PATH = "./model"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH, local_files_only=True)
    return tokenizer, model

tokenizer, model = load_model()


def get_span(text, sentiment):
    """
    DeBERTa 기반 QA span 추출을 안정적으로 수행하는 함수
    """

    encoded = tokenizer(
        sentiment,        # question (감정)
        text,             # context (문장)
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=256
    )

    offset_mapping = encoded["offset_mapping"][0]
    input_ids = encoded["input_ids"]

    # 모델 입력 준비
    encoded.pop("offset_mapping")
    if "token_type_ids" in encoded:
        encoded.pop("token_type_ids")

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

    # 1) 질문 영역은 offset=(0,0)
    # context 시작점 찾기
    context_start_index = None
    for i, (start, end) in enumerate(offset_mapping):
        if not (start == 0 and end == 0):
            context_start_index = i
            break

    if context_start_index is None:
        return text, 0.0, 0, len(text)

    # 2) 질문 영역 logits는 무시 (큰 음수로 마스킹)
    masked_start = start_logits.clone()
    masked_end = end_logits.clone()

    masked_start[:context_start_index] = -1e10
    masked_end[:context_start_index] = -1e10

    # 3) span 위치 찾기
    start_idx = int(torch.argmax(masked_start))
    end_idx = int(torch.argmax(masked_end))

    if end_idx < start_idx:
        end_idx = start_idx

    # 4) 문자 오프셋 변환
    char_start = int(offset_mapping[start_idx][0])
    char_end = int(offset_mapping[end_idx][1])

    if char_start == 0 and char_end == 0:
        return text, 0.0, 0, len(text)

    span = text[char_start:char_end]

    # 확신도
    prob = (
        torch.softmax(start_logits, -1)[start_idx].item() +
        torch.softmax(end_logits, -1)[end_idx].item()
    ) / 2

    return span, prob, char_start, char_end


def analyze_text(text):
    sentiments = ["positive", "negative", "neutral"]
    results = []

    for s in sentiments:
        span, conf, char_s, char_e = get_span(text, s)
        results.append({
            "sentiment": s,
            "span": span,
            "confidence": conf
        })

    return sorted(results, key=lambda x: x['confidence'], reverse=True)
