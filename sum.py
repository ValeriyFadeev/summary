# app.py
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from pydantic import BaseModel

# Инициализируем FastAPI
app = FastAPI()

# ======= Модель суммаризации загружаем при старте =======
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ai-forever/FRED-T5-1.7B"  # или любая другая

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
).to(device)

# ======= Функции суммаризации =======
def split_text(text, tokenizer, max_tokens=900, overlap=100):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    if total_tokens <= max_tokens:
        yield text
        return

    prev_chunk = None
    start = 0

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]

        current_chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        if prev_chunk is not None:
            if len(chunk_tokens) < 200:
                yield prev_chunk + " " + current_chunk
                prev_chunk = None
                break
            else:
                yield prev_chunk
                prev_chunk = current_chunk
        else:
            prev_chunk = current_chunk

        start = end - overlap

    if prev_chunk is not None:
        yield prev_chunk

def summarize_chunk(text):
    inputs = tokenizer(
        "summarize: " + text,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    summary_ids = model.generate(
        inputs.input_ids,
        max_length=300,
        min_length=100,
        num_beams=3,
        length_penalty=1.2,
        early_stopping=True,
        do_sample=True,
        top_k=30
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_long_text(text):
    chunks = split_text(text, tokenizer)
    summaries = []

    for chunk in tqdm(chunks, desc="Обработка чанков"):
        summary = summarize_chunk(chunk)
        summaries.append(summary)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_text = " ".join(summaries)
    final_summary = summarize_chunk(combined_text)

    final_summary = re.sub(r"\s+", " ", final_summary).strip()
    final_summary = ". ".join([s.capitalize() for s in final_summary.split(". ")])

    return final_summary

# ======= Модель для тела запроса =======
class SummarizationRequest(BaseModel):
    text: str

# ======= Эндпоинт =======
@app.post("/summarize")
def summarize_api(request_data: SummarizationRequest):
    """
    Принимает JSON вида:
      {
        "text": "Большой текст..."
      }
    Возвращает:
      {
        "summary": "Краткая выжимка"
      }
    """
    text = request_data.text
    summary = summarize_long_text(text)
    return {"summary": summary}
