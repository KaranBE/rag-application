import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.utils import embedding_functions

SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "sql",
    "machine learning", "deep learning", "nlp", "aws", "azure",
    "gcp", "docker", "kubernetes", "react", "node.js",
    "django", "flask", "pytorch", "tensorflow", "power bi", "tableau",
]

def get_embedding_function(provider: str, model_name: str, api_key: Optional[str]) -> object:
    provider = provider.lower()
    if provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for provider=openai")
        return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)
    if provider == "cohere":
        if not api_key:
            raise ValueError("COHERE_API_KEY is required for provider=cohere")
        return embedding_functions.CohereEmbeddingFunction(api_key=api_key, model_name=model_name)
    if provider == "huggingface":
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    raise ValueError("Unsupported provider. Use one of: openai, cohere, huggingface")

def extract_skills(text: str) -> List[str]:
    lowered = text.lower()
    out: List[str] = []
    for skill in SKILL_VOCAB:
        if re.search(rf"\b{re.escape(skill)}\b", lowered):
            out.append(skill.title())
    return sorted(set(out))

def extract_must_haves(job_description: str) -> List[Tuple[str, int]]:
    lowered = job_description.lower()
    requirements: List[Tuple[str, int]] = []
    for skill in SKILL_VOCAB:
        pattern = rf"(\d+)\+?\s*(?:years|yrs).{{0,30}}{re.escape(skill)}|{re.escape(skill)}.{{0,30}}(\d+)\+?\s*(?:years|yrs)"
        match = re.search(pattern, lowered)
        if match:
            years = next((int(group) for group in match.groups() if group), 0)
            requirements.append((skill.title(), years))
    return requirements

def keyword_overlap_score(jd_skills: Sequence[str], candidate_skills: Sequence[str]) -> float:
    if not jd_skills:
        return 0.0
    jd_set = {s.lower() for s in jd_skills}
    candidate_set = {s.lower() for s in candidate_skills}
    overlap = jd_set & candidate_set
    return len(overlap) / len(jd_set)

def normalize_distance_to_similarity(distance: float) -> float:
    similarity = max(0.0, 1.0 - float(distance))
    return min(1.0, similarity)

def build_reasoning(
    semantic_score: float,
    keyword_score: float,
    matched_skills: List[str],
    must_have_failures: List[str],
    sections: List[str],
) -> str:
    base = f"Semantic relevance={semantic_score:.2f}, keyword overlap={keyword_score:.2f}."
    if matched_skills:
        base += f" Matched skills: {', '.join(matched_skills[:8])}."
    if sections:
