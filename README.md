# Resume RAG + Job Matcher

This folder contains the assignment implementation:

- `resume_rag.py`: document loading, section-aware chunking, embedding, metadata extraction, and ChromaDB indexing
- `job_matcher.py`: semantic + keyword hybrid retrieval, must-have filtering, scoring, and reasoning output

## 1) Setup

```bash
pip install -r requirements.txt
```

## 2) Dataset layout

Put resumes here (30+ recommended):

```text
data/resumes/
  john_doe.pdf
  priya_sharma.docx
  ...
```

Put job descriptions here (5+ recommended):

```text
data/job_descriptions/
  jd_ml_engineer.txt
  jd_data_scientist.txt
  ...
```

## 3) Build the RAG index

```bash
python resume_rag.py \
  --resume-dir data/resumes \
  --chroma-path .chroma_resume_index \
  --collection resume_chunks \
  --embedding-provider huggingface \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Switch embedding providers with:

- `--embedding-provider openai --embedding-model text-embedding-3-small`
- `--embedding-provider cohere --embedding-model embed-english-v3.0`

Environment vars (if needed):

- `OPENAI_API_KEY`
- `COHERE_API_KEY`

## 4) Run job matching

```bash
python job_matcher.py \
  --job-description "We need a Python ML engineer with 5+ years of experience in NLP, AWS, and Docker." \
  --chroma-path .chroma_resume_index \
  --collection resume_chunks \
  --top-k 10 \
  --candidate-pool 60 \
  --embedding-provider huggingface \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-path output_matches.json
```

## Output schema

```json
{
  "job_description": "...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "data/resumes/john_doe.pdf",
      "match_score": 92,
      "matched_skills": ["Python", "Machine Learning"],
      "relevant_excerpts": ["..."],
      "reasoning": "Strong match..."
    }
  ]
}
```

## Suggested notebook deliverable

Create a notebook named `experiments.ipynb` with:

1. Embedding model comparison (OpenAI/Cohere/HF)
2. Top-K sensitivity (`K=5,10,15`)
3. Hybrid weight tuning (`semantic=0.65`, `keyword=0.35`)
4. Retrieval latency benchmarks
5. Accuracy metrics on manually labeled relevant candidates
