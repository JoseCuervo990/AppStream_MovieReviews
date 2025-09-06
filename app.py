import json
import math
from typing import List, Dict

import streamlit as st

# Lazy imports for heavy ML libraries
def load_zero_shot_model():
    try:
        from transformers import pipeline
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.error("Zero-shot model unavailable: {}. Install 'transformers' and 'torch'.".format(e))
        return None


def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error("Embedding model unavailable: {}. Install 'sentence-transformers' and its deps.".format(e))
        return None


@st.cache_resource
def get_models():
    return {
        "zero_shot": load_zero_shot_model(),
        "embed": load_embedding_model(),
    }


def predict_zero_shot(text: str, labels: List[str]):
    models = get_models()
    clf = models["zero_shot"]
    if clf is None:
        return {"error": "zero-shot model not loaded"}
    result = clf(text, candidate_labels=labels)
    return result


def predict_few_shot(text: str, examples: Dict[str, List[str]], k: int = 3):
    """Simple few-shot implemented via embedding k-NN over provided examples.
    Returns predicted label, scores, and nearest example texts.
    """
    models = get_models()
    embed_model = models["embed"]
    if embed_model is None:
        return {"error": "embedding model not loaded"}

    # Flatten examples
    labels = list(examples.keys())
    example_texts = []
    example_labels = []
    for lbl in labels:
        for ex in examples[lbl]:
            example_texts.append(ex)
            example_labels.append(lbl)

    if len(example_texts) == 0:
        return {"error": "no few-shot examples provided"}

    # Compute embeddings
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    ex_emb = embed_model.encode(example_texts, convert_to_numpy=True)
    q_emb = embed_model.encode([text], convert_to_numpy=True)

    sims = cosine_similarity(q_emb, ex_emb)[0]  # shape (n_examples,)
    # Get top-k
    topk_idx = sims.argsort()[::-1][:k]

    # Aggregate by label (sum of similarities)
    label_scores = {lbl: 0.0 for lbl in labels}
    nearest = []
    for idx in topk_idx:
        nearest.append({"text": example_texts[idx], "label": example_labels[idx], "score": float(sims[idx])})
        label_scores[example_labels[idx]] += float(sims[idx])

    # Normalize scores
    total = sum(label_scores.values())
    if total > 0:
        for lbl in label_scores:
            label_scores[lbl] /= total

    # Choose best
    pred_label = max(label_scores.items(), key=lambda x: x[1])[0]

    return {
        "prediction": pred_label,
        "label_scores": label_scores,
        "nearest": nearest,
    }


def load_default_examples(path: str = "few_shot_examples.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception:
        # fallback small defaults
        return {
            "positive": [
                "I loved this movie. The story was touching and the acting was superb.",
                "A wonderful, moving film with great performances.",
            ],
            "negative": [
                "This movie was boring and too long. I hated it.",
                "Poor script and bad acting; not worth watching.",
            ],
            "neutral": [
                "It was an average movie, some parts were good and some were bad.",
                "Nothing special — a standard genre film.",
            ],
        }


def main():
    st.set_page_config(page_title="Movie Review: Zero-shot & Few-shot demo", layout="centered")
    st.title("Movie Review Sentiment — Zero-shot vs Few-shot")

    st.markdown("Use the zero-shot classifier (NLI-based) or the few-shot k-NN (embedding) approach.")

    col1, col2 = st.columns([3, 1])
    with col1:
        review = st.text_area("Paste or write a movie review:", height=200)
    with col2:
        method = st.radio("Method", ["Zero-shot", "Few-shot"], index=0)
        labels_input = st.text_input("Candidate labels (comma separated, for zero-shot)", value="positive,negative,neutral")
        use_default_examples = st.checkbox("Use default few-shot examples", value=True)
        k = st.number_input("k (nearest examples)", min_value=1, max_value=10, value=3)

    if st.button("Analyze"):
        if not review or review.strip() == "":
            st.warning("Please enter a review to analyze.")
            return

        if method == "Zero-shot":
            labels = [l.strip() for l in labels_input.split(",") if l.strip()]
            with st.spinner("Running zero-shot classifier..."):
                res = predict_zero_shot(review, labels)
            if "error" in res:
                st.error(res["error"])
            else:
                st.subheader("Zero-shot results")
                # show label scores
                for lbl, score in zip(res["labels"], res["scores"]):
                    st.write(f"- {lbl}: {score:.3f}")
                # Show top label
                st.success(f"Predicted: {res['labels'][0]} (score {res['scores'][0]:.3f})")

        else:  # Few-shot
            if use_default_examples:
                examples = load_default_examples("c:/Users/Javi2/Analitica/movie_review_streamlit/few_shot_examples.json")
            else:
                st.info("Provide examples later — currently only default examples are supported in this demo.")
                examples = load_default_examples()

            with st.spinner("Running few-shot embedding k-NN..."):
                res = predict_few_shot(review, examples, k=int(k))

            if "error" in res:
                st.error(res["error"])
            else:
                st.subheader("Few-shot results")
                st.write(f"Predicted: {res['prediction']}")
                st.write("Label scores:")
                for lbl, score in res["label_scores"].items():
                    st.write(f"- {lbl}: {score:.3f}")

                st.write("Nearest examples used:")
                for n in res["nearest"]:
                    st.write(f"- ({n['label']}, {n['score']:.3f}) — {n['text']}")

    st.markdown("---")
    st.markdown("### Notes")
    st.markdown(
        "- Zero-shot uses a Natural Language Inference (NLI) model (e.g., facebook/bart-large-mnli).\n- Few-shot uses sentence embeddings + k-NN over a handful of examples.\n- Models will be downloaded on first run if not present locally."
    )


if __name__ == "__main__":
    main()
