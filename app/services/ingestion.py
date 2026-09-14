from pathlib import Path
from pypdf import PdfReader
from docx import Document
import pytesseract
from PIL import Image

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return pytesseract.image_to_string(Image.open(path))
    raise ValueError("Unsupported format. Use PDF, DOCX, or a supported image.")
