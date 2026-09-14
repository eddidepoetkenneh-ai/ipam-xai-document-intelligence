# IPAM Explainable AI Document Intelligence Prototype

Prototype aligned with the Group Six proposal.

## Scope
- User authentication
- Document upload
- PDF/DOCX/image ingestion
- OCR hook using PyTesseract
- Document categorisation
- Semantic-search-ready architecture
- Explanation-ready API response
- PostgreSQL-ready configuration
- Simple web UI

## Run
1. Create a virtual environment.
2. Install `requirements.txt`.
3. Run `uvicorn app.main:app --reload`.
4. Open `/`.

The prototype deliberately uses a lightweight rule-based classifier fallback so the application can run before a trained classifier/model is supplied. Replace the fallback with the selected Hugging Face/Sentence-BERT model for the final AI evaluation.
