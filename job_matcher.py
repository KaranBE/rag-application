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
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "react",
    "node.js",
    "django",
    "flask",
    "pytorch",
    "tensorflow",
    "power bi",
    "tableau",
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
        base += f" Strong signals from sections: {', '.join(sorted(set(sections))[:4])}."
    if must_have_failures:
        base += f" Missing must-have constraints: {', '.join(must_have_failures)}."
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid semantic + keyword job matching over indexed resumes")
    parser.add_argument("--job-description", required=True, help="Raw job description text")
    parser.add_argument("--chroma-path", default=".chroma_resume_index", help="Path where Chroma index is stored")
    parser.add_argument("--collection", default="resume_chunks", help="Chroma collection name")
    parser.add_argument("--embedding-provider", choices=["openai", "cohere", "huggingface"], default="huggingface")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--cohere-api-key", default=os.getenv("COHERE_API_KEY"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=60, help="Chunk retrieval pool size before candidate aggregation")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.openai_api_key if args.embedding_provider == "openai" else args.cohere_api_key
    embedding_function = get_embedding_function(args.embedding_provider, args.embedding_model, api_key)
    client = chromadb.PersistentClient(path=str(Path(args.chroma_path)))
    collection = client.get_collection(args.collection, embedding_function=embedding_function)

    jd_skills = extract_skills(args.job_description)
    must_haves = extract_must_haves(args.job_description)
    query_result = collection.query(query_texts=[args.job_description], n_results=max(args.top_k, args.candidate_pool), include=["documents", "metadatas", "distances"])

    docs = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]
    distances = query_result.get("distances", [[]])[0]

    candidate_bucket: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "candidate_name": "",
            "resume_path": "",
            "candidate_skills": [],
            "experience_years": 0,
            "semantic_scores": [],
            "sections": [],
            "relevant_excerpts": [],
        }
    )

    for doc, metadata, distance in zip(docs, metadatas, distances):
        resume_path = str(metadata.get("resume_path", ""))
        if not resume_path:
            continue

        bucket = candidate_bucket[resume_path]
        bucket["candidate_name"] = str(metadata.get("candidate_name", "Unknown Candidate"))
        bucket["resume_path"] = resume_path
        bucket["candidate_skills"] = [s.strip() for s in str(metadata.get("skills", "")).split(",") if s.strip()]
        bucket["experience_years"] = int(metadata.get("experience_years", "0"))
        bucket["semantic_scores"].append(normalize_distance_to_similarity(float(distance)))
        bucket["sections"].append(str(metadata.get("section", "general")))
        if len(bucket["relevant_excerpts"]) < 3:
            bucket["relevant_excerpts"].append(doc[:280].replace("\n", " "))

    ranked: List[Dict[str, object]] = []
    for _, bucket in candidate_bucket.items():
        candidate_skills = bucket["candidate_skills"]
        semantic_score = sum(bucket["semantic_scores"]) / max(len(bucket["semantic_scores"]), 1)
        keyword_score = keyword_overlap_score(jd_skills, candidate_skills)
        matched_skills = sorted(set(jd_skills) & set(candidate_skills))

        must_have_failures = []
        for skill, years in must_haves:
            if skill not in candidate_skills or int(bucket["experience_years"]) < years:
                must_have_failures.append(f"{years}+ years {skill}")
        must_have_penalty = 0.0 if not must_have_failures else 0.2

        hybrid_score = (0.65 * semantic_score) + (0.35 * keyword_score) - must_have_penalty
        match_score = max(0, min(100, round(hybrid_score * 100)))
        reasoning = build_reasoning(
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            matched_skills=matched_skills,
            must_have_failures=must_have_failures,
            sections=bucket["sections"],
        )

        ranked.append(
            {
                "candidate_name": bucket["candidate_name"],
                "resume_path": bucket["resume_path"],
                "match_score": match_score,
                "matched_skills": matched_skills[:10],
                "relevant_excerpts": bucket["relevant_excerpts"],
                "reasoning": reasoning,
            }
        )

    ranked = sorted(ranked, key=lambda x: int(x["match_score"]), reverse=True)[: args.top_k]
    output = {
        "job_description": args.job_description,
        "top_matches": ranked,
    }

    output_json = json.dumps(output, indent=2)
    if args.output_path:
        Path(args.output_path).write_text(output_json, encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
