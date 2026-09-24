---
title: PDF Question Answering
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# PDF Question Answering (multimodal RAG)

Upload a PDF, inspect what was extracted (text, tables, figures, equations), and ask questions with cited answers.

**Pipeline:** Docling extraction → section-aware chunks → Chroma (MiniLM embeddings) + BM25 hybrid search → Gemini answer with figure/table images.

## Run locally

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
docling-tools models download layout tableformer code_formula
python app.py            # http://localhost:8000
```

Create a `.env` with:

```
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GEMINI_API_KEY=...
```

## Deploy on Hugging Face Spaces

1. Create a Space → SDK **Docker** → hardware **CPU basic** (free).
2. Push this folder to the Space repo.
3. Space **Settings → Variables and secrets**:
   - Secret `GEMINI_API_KEY` – your key
   - Secret `APP_PASSWORD` – optional; visitors must enter it (username can be anything)
   - Variable `MODEL_NAME` = `sentence-transformers/all-MiniLM-L6-v2`

Storage on the free tier is temporary: uploaded PDFs and the index are lost when the Space restarts.
