from sentence_transformers import SentenceTransformer, util
import re


# Load the semantic embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Descriptions representing the four supported document categories
CATEGORY_DESCRIPTIONS = {

    "Transcript": (
        "Academic student transcript containing courses, grades, "
        "semester results, GPA, credits, academic performance and "
        "student academic records."
    ),

    "Admission": (
        "University admission document containing applicant information, "
        "application details, admission decisions, acceptance, enrollment "
        "or offer of admission."
    ),

    "Administrative Memo": (
        "Institutional administrative memorandum containing official "
        "communication, departments, meetings, instructions, notices, "
        "office information and internal directives."
    ),

    "Financial Record": (
        "Institutional financial document containing payments, invoices, "
        "receipts, fees, tuition, balances, transactions, amounts and "
        "financial information."
    )
}


# Create embeddings for the category descriptions
CATEGORY_EMBEDDINGS = {

    category: model.encode(
        description,
        convert_to_tensor=True
    )

    for category, description
    in CATEGORY_DESCRIPTIONS.items()
}


# Weighted keyword evidence
KEYWORDS = {

    "Transcript": {

        "transcript": 5,
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
        "date": 2
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


def keyword_fallback(text: str):

    """
    Classify the document using weighted keyword evidence.

    This is used when semantic evidence is too weak.
    """

    low = text.lower()

    scores = {}

    for category, keywords in KEYWORDS.items():

        score = 0

        for keyword, weight in keywords.items():

            matches = len(
                re.findall(
                    r"\b" + re.escape(keyword) + r"\b",
                    low
                )
            )

            score += matches * weight

        scores[category] = score


    best_category = max(
        scores,
        key=scores.get
    )


    total = sum(scores.values())


    # No meaningful keyword evidence
    if total == 0:

        return "Unclassified", 0.0


    confidence = scores[best_category] / total


    return best_category, round(confidence, 3)


def classify_document(text: str):

    """
    Hybrid document classifier.

    1. Attempts semantic classification using sentence embeddings.
    2. If semantic evidence is weak, uses weighted keyword evidence.
    3. If neither method finds meaningful evidence, returns Unclassified.
    """

    if not text or not text.strip():

        return "Unclassified", 0.0


    # Limit the amount of text sent to the embedding model
    document_text = text[:5000]


    # Generate document embedding
    document_embedding = model.encode(
        document_text,
        convert_to_tensor=True
    )


    # Calculate semantic similarity against every category
    similarities = {}

    for category, category_embedding in CATEGORY_EMBEDDINGS.items():

        similarity = util.cos_sim(
            document_embedding,
            category_embedding
        ).item()

        similarities[category] = similarity


    # Select the strongest semantic category
    best_category = max(
        similarities,
        key=similarities.get
    )


    best_similarity = similarities[best_category]


    # ---------------------------------------------------------
    # HYBRID FALLBACK
    #
    # If semantic evidence is weak, use weighted keyword
    # evidence. This prevents legitimate IPAM documents from
    # being incorrectly marked as Unclassified.
    # ---------------------------------------------------------

    if best_similarity < 0.35:

        return keyword_fallback(text)


    # Convert semantic similarity to a bounded score
    confidence = max(
        0.0,
        min(1.0, best_similarity)
    )


    return best_category, round(confidence, 3)