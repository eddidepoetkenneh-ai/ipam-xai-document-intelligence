import re

CATEGORIES = {
    "Transcript": ["transcript", "semester", "grade", "course", "gpa", "result"],
    "Admission": ["admission", "applicant", "application", "entry", "acceptance"],
    "Administrative Memo": ["memo", "memorandum", "department", "meeting", "office"],
    "Financial Record": ["invoice", "payment", "receipt", "fee", "financial", "amount"]
}

def classify_document(text: str):
    low = text.lower()
    scores = {cat: sum(len(re.findall(r"\b" + re.escape(w) + r"\b", low))
                       for w in words) for cat, words in CATEGORIES.items()}
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    if total == 0:
        return "Unclassified", 0.0
    return best, round(scores[best] / total, 3)
