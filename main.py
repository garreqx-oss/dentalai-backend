from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel
import openai
import os
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dental AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ─── Models ───────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    transcription: str
    visit_type: str = "Kontrolna"
    patient_name: str = ""
    language: str = "pl"

class AnalyzeResponse(BaseModel):
    summary: str = ""
    diagnosis: str = ""
    treatment: str = ""
    recommendations: str = ""
    medications: str = ""
    next_visit: str = ""


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "Dental AI API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="pl")
):
    """Transkrybuje plik audio WAV używając Whisper."""
    logger.info(f"Transkrypcja: {file.filename}, język: {language}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Brak pliku audio")

    # Zapisz plik tymczasowo
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Wysyłanie do Whisper, rozmiar: {len(content)} bajtów")

        with open(tmp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="text"
            )

        transcription = response if isinstance(response, str) else response.text
        logger.info(f"Transkrypcja gotowa: {len(transcription)} znaków")

        return {"transcription": transcription}

    except openai.APIError as e:
        logger.error(f"Błąd OpenAI API: {e}")
        raise HTTPException(status_code=502, detail=f"Błąd Whisper: {str(e)}")
    except Exception as e:
        logger.error(f"Błąd transkrypcji: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd transkrypcji: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_visit(request: AnalyzeRequest):
    """Analizuje transkrypcję wizyty i generuje raport medyczny."""
    logger.info(f"Analiza wizyty: {request.patient_name}, typ: {request.visit_type}")

    if not request.transcription or len(request.transcription.strip()) < 10:
        raise HTTPException(status_code=400, detail="Transkrypcja jest zbyt krótka")

    prompt = f"""Jesteś asystentem lekarza dentysty. Przeanalizuj poniższą transkrypcję rozmowy między dentystą a pacjentem i wygeneruj strukturyzowany raport medyczny w języku polskim.

DANE WIZYTY:
- Pacjent: {request.patient_name or 'nieznany'}
- Typ wizyty: {request.visit_type}

TRANSKRYPCJA:
{request.transcription}

Wygeneruj raport w dokładnie tym formacie JSON (wypełnij każde pole na podstawie transkrypcji, jeśli informacja nie pada - napisz "Brak danych"):

{{
  "summary": "Krótkie podsumowanie wizyty (2-3 zdania)",
  "diagnosis": "Diagnoza i stan zdrowia jamy ustnej pacjenta",
  "treatment": "Opis przeprowadzonego leczenia podczas wizyty",
  "recommendations": "Zalecenia dla pacjenta po wizycie (higiena, dieta, itp.)",
  "medications": "Przepisane leki lub środki (nazwa, dawka, czas stosowania) lub 'Nie przepisano leków'",
  "next_visit": "Termin i cel następnej wizyty lub 'Do ustalenia'"
}}

Odpowiedz TYLKO JSON, bez żadnego dodatkowego tekstu."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś precyzyjnym asystentem medycznym specjalizującym się w stomatologii. Odpowiadasz wyłącznie w formacie JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        logger.info(f"Odpowiedź GPT: {content[:200]}...")

        import json
        data = json.loads(content)

        return AnalyzeResponse(
            summary=data.get("summary", ""),
            diagnosis=data.get("diagnosis", ""),
            treatment=data.get("treatment", ""),
            recommendations=data.get("recommendations", ""),
            medications=data.get("medications", ""),
            next_visit=data.get("next_visit", "")
        )

    except openai.APIError as e:
        logger.error(f"Błąd OpenAI API: {e}")
        raise HTTPException(status_code=502, detail=f"Błąd GPT: {str(e)}")
    except Exception as e:
        logger.error(f"Błąd analizy: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd analizy: {str(e)}")
