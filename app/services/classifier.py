import re


CATEGORIES = {

    "Transcript": {
        "transcript": 5,
        "academic transcript": 6,
        "semester": 2,
        "grade": 2,
        "grades": 2,
        "course": 2,
        "courses": 2,
        "gpa": 4,
        "cgpa": 4,
        "result": 2,
        "results": 2,
        "credit": 2,
        "credits": 2,
        "student number": 4,
        "matriculation": 4
    },

    "Admission": {
        "admission": 5,
        "admission letter": 6,
        "applicant": 3,
        "application": 3,
        "entry": 2,
        "acceptance": 4,
        "acceptance letter": 6,
        "enrollment": 3,
        "enrolment": 3,
        "offer letter": 5
    },

    "Administrative Memo": {
        "memo": 5,
        "memorandum": 6,
        "department": 2,
        "meeting": 2,
        "office": 2,
        "official": 2,
        "subject": 2,
        "to:": 2,
        "from:": 2,
        "cc:": 2,
        "date:": 2
    },

    "Financial Record": {
        "invoice": 6,
        "payment": 4,
        "receipt": 5,
        "fee": 3,
        "fees": 3,
        "financial": 4,
        "amount": 2,
        "total": 2,
        "balance": 3,
        "tuition": 4,
        "transaction": 4,
        "account number": 4
    }

}


def count_term(text: str, term: str) -> int:
    """
    Count occurrences of a term in the document text.
    """

    pattern = r"\b" + re.escape(term) + r"\b"

    return len(re.findall(pattern, text))


def classify_document(text: str):
    """
    Classify a document using weighted evidence.

    Returns:
        category, confidence
    """

    if not text or not text.strip():
        return "Unclassified", 0.0

    low = text.lower()

    scores = {}

    for category, keywords in CATEGORIES.items():

        score = 0

        for keyword, weight in keywords.items():

            occurrences = count_term(low, keyword)

            score += occurrences * weight

        scores[category] = score


    # Find category with highest score

    best_category = max(scores, key=scores.get)

    best_score = scores[best_category]

    total_score = sum(scores.values())


    # No meaningful evidence

    if best_score == 0 or total_score == 0:

        return "Unclassified", 0.0


    # Require stronger evidence before classification

    if best_score < 4:

        return "Unclassified", 0.0


    # Calculate relative confidence

    confidence = best_score / total_score


    return best_category, round(confidence, 3)