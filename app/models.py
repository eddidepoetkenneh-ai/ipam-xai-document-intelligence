from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from datetime import datetime

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)
    extracted_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    explanation = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)