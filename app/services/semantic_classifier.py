from sentence_transformers import SentenceTransformer, util


# Load the semantic embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


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


# Create embeddings for category descriptions
CATEGORY_EMBEDDINGS = {
    category: model.encode(description, convert_to_tensor=True)
    for category, description in CATEGORY_DESCRIPTIONS.items()
}


def classify_semantically(text: str):
    """
    Classify a document using semantic similarity.
    """

    if not text or not text.strip():
        return "Unclassified", 0.0

    document_embedding = model.encode(
        text[:5000],
        convert_to_tensor=True
    )

    similarities = {}

    for category, category_embedding in CATEGORY_EMBEDDINGS.items():

        similarity = util.cos_sim(
            document_embedding,
            category_embedding
        ).item()

        similarities[category] = similarity


    best_category = max(
        similarities,
        key=similarities.get
    )

    best_score = similarities[best_category]

    # Convert similarity into a bounded confidence value
    confidence = max(0.0, min(1.0, best_score))

    return best_category, round(confidence, 3)