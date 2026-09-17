from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import json

from .services.ingestion import extract_text
from .services.classifier import classify_document
from .services.explain import explain_classification
from .database import SessionLocal
from .models import Document

BASE = Path(__file__).resolve().parent.parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="IPAM XAI Document Intelligence Prototype")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "templates" / "index.html").read_text(encoding="utf-8")


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    safe_name = Path(file.filename).name
    target = UPLOADS / safe_name

    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text(target)
    category, score = classify_document(text)
    explanation = explain_classification(text, category)
    explanation_text = json.dumps(explanation)

    db = SessionLocal()

    try:
        document = Document(
            filename=safe_name,
            category=category,
            extracted_text=text,
            confidence=score,
            explanation=explanation_text
        )

        db.add(document)
        db.commit()
        db.refresh(document)

    finally:
        db.close()

    return JSONResponse({
        "filename": safe_name,
        "category": category,
        "classification_score": score,
        "explanation": explanation,
        "text_preview": text[:1000],
        "document_id": document.id
    })
@app.get("/api/documents")
def get_documents():
    db = SessionLocal()

    try:
        documents = (
            db.query(Document)
            .order_by(Document.id.desc())
            .all()
        )

        return [
            {
                "id": document.id,
                "filename": document.filename,
                "category": document.category,
                "confidence": document.confidence,
                "explanation": document.explanation,
                "uploaded_at": (
                    document.uploaded_at.isoformat()
                    if document.uploaded_at
                    else None
                )
            }
            for document in documents
        ]

    finally:
        db.close()