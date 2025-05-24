import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from redis import Redis
from rq import Queue
from typing import Dict, Any

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")

app = FastAPI()
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

redis_conn = Redis(
    host="redis",
    port=6379,
    db=0,
    decode_responses=False,
    health_check_interval=30
)

task_queue = Queue("text_classification_queue", connection=redis_conn)


class TextRequest(BaseModel):
    text: str


def classify_text(text: str) -> Dict[str, Any]:
    try:
        # Tokenization
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt',
            add_special_tokens=True
        )

        # Inference
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            probs = F.softmax(outputs.logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        return {
            "label": "human" if pred.item() == 1 else "ai",
            "confidence": round(confidence.item(), 2)
        }

    except Exception as e:
        print(f"Classification failed: {e}")
        raise


@app.post("/predict")
async def predict(request: TextRequest):
    try:
        result = classify_text(request.text)
        return {
            "prediction": result["label"],
            "confidence": f"{result['confidence']}%",
            "status": "success"
        }
    except Exception as e:
        print(f"Prediction failed: {str(e)}")


@app.post("/enqueue_task")
async def enqueue_task(request: TextRequest) -> Dict[str, str]:
    """
    Endpoint for Telegram bot to submit tasks
    """
    job = task_queue.enqueue(
        classify_text,
        request.text,
        result_ttl=3600  # Store result for 1 hour
    )
    return {"job_id": job.id, "status": "queued"}


@app.get("/check_result/{job_id}")
async def check_result(job_id: str) -> Dict[str, Any]:
    """
    Endpoint for Telegram bot to check results
    """
    job = task_queue.fetch_job(job_id)

    if not job:
        return {"status": "not_found"}

    if job.is_failed:
        return {"status": "failed"}

    if job.is_finished:
        return {
            "status": "completed",
            "result": job.result
        }

    return {"status": "pending"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
