def explain_classification(text: str, category: str):
    keywords = {
        "Transcript": ["transcript", "semester", "grade", "course", "gpa", "result"],
        "Admission": ["admission", "applicant", "application", "entry", "acceptance"],
        "Administrative Memo": ["memo", "memorandum", "department", "meeting", "office"],
        "Financial Record": ["invoice", "payment", "receipt", "fee", "financial", "amount"]
    }.get(category, [])
    low = text.lower()
    matched = [w for w in keywords if w in low]
    return {
        "method": "keyword-evidence fallback",
        "category": category,
        "supporting_terms": matched,
        "note": "Replace this fallback with the final LIME/SHAP or model-specific explanation component before final evaluation."
    }
