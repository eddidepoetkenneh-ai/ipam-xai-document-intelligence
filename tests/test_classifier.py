from app.services.classifier import classify_document

def test_transcript_classification():
    cat, score = classify_document("Student transcript course grade semester GPA result")
    assert cat == "Transcript"
    assert score > 0
