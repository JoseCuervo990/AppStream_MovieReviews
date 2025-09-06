Javier DC movie APP

This small demo shows two approaches to sentiment-style labeling for movie reviews:

- Zero-shot classification using an NLI model (e.g., `facebook/bart-large-mnli`).
- Few-shot classification implemented as embedding-based k-NN using `all-MiniLM-L6-v2` embeddings.

How to run

1. Create or activate a Python environment (Python 3.8+ recommended).
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

Notes

- The first run may download transformer weights; this can take time and disk space.
- If you prefer a smaller zero-shot model, replace the model name in `app.py` (for example, use `valhalla/distilbart-mnli-12-1`).
- The app handles missing packages gracefully and will show an error if a model cannot be loaded.
