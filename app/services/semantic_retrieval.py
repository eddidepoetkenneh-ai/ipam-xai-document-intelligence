from sentence_transformers import SentenceTransformer, util


# Load the same embedding model used by the document classifier.
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_document_embedding(text: str):
    """
    Convert document text into a semantic vector representation.
    """
    if not text or not text.strip():
        return None

    return model.encode(
        text[:5000],
        convert_to_tensor=True
    )


def semantic_search(query: str, documents: list, top_k: int = 5):
    """
    Search stored documents by semantic similarity.

    Duplicate database records with the same filename are reduced
    to one searchable document so that repeated test uploads do
    not appear multiple times in the search results.
    """

    if not query or not query.strip():
        return []

    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    # Keep only the most recent database record for each unique filename.
    unique_documents = {}

    for document in documents:
        if not document.filename:
            continue

        existing = unique_documents.get(document.filename)

        if existing is None or document.id > existing.id:
            unique_documents[document.filename] = document

    results = []

    for document in unique_documents.values():
        text = document.extracted_text

        if not text or not text.strip():
            continue

        document_embedding = model.encode(
            text[:5000],
            convert_to_tensor=True
        )

        similarity = util.cos_sim(
            query_embedding,
            document_embedding
        ).item()

        results.append({
            "id": document.id,
            "filename": document.filename,
            "category": document.category,
            "similarity": round(
                max(0.0, min(1.0, similarity)),
                3
            )
        })

    # Rank documents from highest to lowest semantic similarity.
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:top_k]