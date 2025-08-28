from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LetterRequest(BaseModel):
    tenantName: str
    address: str
    issueDate: str
    description: str

class LetterResponse(BaseModel):
    letter: str


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")  # accept either
if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN (or HF_API_TOKEN) in backend environment.")

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

# create a single reusable client
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)


def prompt_from(req: LetterRequest) -> str:
    #todo: improve prompt
    return (
        f"Compose a polite German landlord letter using the following structure.\n\n"
        f"Berlin, den {req.issueDate}\n\n"
        f"Sehr geehrte Damen und Herren,\n\n"
        f"hiermit möchte ich, {req.tenantName}, wohnhaft in {req.address}, "
        f"Sie darüber informieren, dass in meiner Wohnung folgende Mängel aufgetreten sind:\n\n"
        f"{req.description}\n\n"
        f"Ich bitte Sie, die Beseitigung dieser Mängel möglichst umgehend zu veranlassen.\n\n"
        f"Bitte teilen Sie uns bis zum {generate_deadline(req.issueDate)} mit, wann und durch wen die Mängel beseitigt werden, damit wir uns entsprechend einrichten können.\n\n"
        f"In Erwartung Ihrer fristgerechten Antwort verbleiben wir\n"
        f"Mit freundlichen Grüßen\n{req.tenantName}"
        f"Write the full letter in German. Translate the issues if needed. Only include the letter text, no explanations."
    )


def generate_with_hf(prompt: str) -> str:
    try:
        completion = hf_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt}
                ],
        )
        msg = completion.choices[0].message
        # works whether message is an object or dict
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        if not content:
            raise RuntimeError("Empty response from HF chat completion.")
        return content
    except Exception as e:
        raise RuntimeError(f"Hugging Face chat completion failed: {e}")

def generate_deadline(current_date: str) -> str:
    date_obj = datetime.strptime(current_date, "%Y-%m-%d")
    deadline_date = date_obj + timedelta(weeks=2)
    return deadline_date.strftime("%Y-%m-%d")

@app.post("/generate-letter-template", response_model=LetterResponse)
def generate_letter_template(req: LetterRequest):
    try:
        text = (
            f"Berlin, den {req.issueDate}\n\n"
            f"Sehr geehrte Damen und Herren,\n\n"
            f"hiermit möchte ich, {req.tenantName}, wohnhaft in {req.address}, "
            f"Sie darüber informieren, dass in meiner Wohnung folgende Mängel aufgetreten sind:\n\n"
            f"{req.description}\n\n"
            f"Ich bitte Sie, die Beseitigung dieser Mängel möglichst umgehend zu veranlassen.\n\n"
            f"Bitte teilen Sie uns bis zum {generate_deadline(req.issueDate)} mit, wann und durch wen die Mängel beseitigt werden, damit wir uns entsprechend einrichten können.\n\n"
            f"In Erwartung Ihrer fristgerechten Antwort verbleiben wir\n"
            f"Mit freundlichen Grüßen\n{req.tenantName}"
        )
        return {"letter": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/generate-letter-llm", response_model=LetterResponse)
def generate_letter_LLM(req: LetterRequest):
    prompt = prompt_from(req)
    try:
        text = generate_with_hf(prompt)
        return {"letter": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))