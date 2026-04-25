import argparse
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency at runtime
    PdfReader = None

try:
    import docx  # python-docx
except Exception:  # pragma: no cover - optional dependency at runtime
    docx = None


SECTION_PATTERNS = [
    "summary",
    "professional summary",
    "experience",
    "work experience",
    "employment history",
    "projects",
    "education",
    "skills",
    "certifications",
    "achievements",
]

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


@dataclass
class ResumeChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Install with: pip install pypdf")
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_docx_file(path: Path) -> str:
    if docx is None:
        raise RuntimeError("python-docx is not installed. Install with: pip install python-docx")
    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())


def load_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix in {".docx", ".doc"}:
        return read_docx_file(path)
    raise ValueError(f"Unsupported file format: {path.name}")


def iter_resume_files(resume_dir: Path) -> Iterable[Path]:
    for extension in ("*.pdf", "*.docx", "*.doc", "*.txt"):
        yield from resume_dir.rglob(extension)


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sections(text: str) -> List[Tuple[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: List[Tuple[str, List[str]]] = [("general", [])]
    heading_pattern = re.compile(r"^[A-Za-z][A-Za-z &/()\-]{2,40}$")

    for line in lines:
        lower_line = line.lower()
        looks_like_heading = heading_pattern.match(line) and any(key in lower_line for key in SECTION_PATTERNS)
        if looks_like_heading:
            sections.append((lower_line, []))
        else:
            sections[-1][1].append(line)

    output: List[Tuple[str, str]] = []
    for section_name, section_lines in sections:
        if section_lines:
            output.append((section_name, "\n".join(section_lines)))
    return output


def chunk_section_text(section_name: str, section_text: str, max_chars: int = 950, overlap_chars: int = 180) -> List[str]:
    section_text = normalize_spaces(section_text)
    if len(section_text) <= max_chars:
        return [f"[{section_name}] {section_text}"]

    chunks: List[str] = []
    start = 0
    while start < len(section_text):
        end = min(start + max_chars, len(section_text))
        chunk_text = section_text[start:end]
        chunks.append(f"[{section_name}] {chunk_text}")
        if end == len(section_text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def extract_name(text: str, fallback_stem: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback_stem.replace("_", " ").title()
    first_line = lines[0]
    if len(first_line.split()) <= 5 and not any(char.isdigit() for char in first_line):
        return first_line
    return fallback_stem.replace("_", " ").title()


def extract_experience_years(text: str) -> int:
    lowered = text.lower()
    patterns = [
        r"(\d+)\+?\s*(?:years|yrs)\s+of\s+experience",
        r"experience\s*[:\-]?\s*(\d+)\+?\s*(?:years|yrs)",
        r"(\d+)\+?\s*(?:years|yrs)\s+experience",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return 0


def extract_education(text: str) -> str:
    lowered = text.lower()
    if "phd" in lowered:
        return "PhD"
    if "master" in lowered or "m.tech" in lowered or "mba" in lowered:
        return "Masters"
    if "bachelor" in lowered or "b.tech" in lowered or "b.e" in lowered:
        return "Bachelors"
    return "Not Specified"


def extract_skills(text: str) -> List[str]:
    lowered = text.lower()
    matched = []
    for skill in SKILL_VOCAB:
        if re.search(rf"\b{re.escape(skill)}\b", lowered):
            matched.append(skill.title())
    return sorted(set(matched))


def build_resume_metadata(path: Path, text: str) -> Dict[str, str]:
    skills = extract_skills(text)
    metadata: Dict[str, str] = {
        "candidate_name": extract_name(text, path.stem),
        "resume_path": str(path.as_posix()),
        "skills": ",".join(skills),
        "experience_years": str(extract_experience_years(text)),
        "education": extract_education(text),
    }
    return metadata


def build_resume_chunks(path: Path, text: str) -> List[ResumeChunk]:
    sections = split_sections(text)
    base_metadata = build_resume_metadata(path, text)
    output: List[ResumeChunk] = []

    for section_name, section_text in sections:
        for chunk_text in chunk_section_text(section_name, section_text):
            chunk_id = str(uuid.uuid4())
            metadata = {
                **base_metadata,
                "section": section_name,
            }
            output.append(ResumeChunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))
    return output


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


def get_or_create_collection(
    chroma_path: Path,
    collection_name: str,
    embedding_function: object,
) -> Collection:
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    return collection


def upsert_chunks(collection: Collection, chunks: Sequence[ResumeChunk]) -> None:
    if not chunks:
        return
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )


def write_summary(summary_path: Path, resume_count: int, chunk_count: int, collection_name: str, chroma_path: Path) -> None:
    summary = {
        "resumes_indexed": resume_count,
        "chunks_indexed": chunk_count,
        "collection_name": collection_name,
        "vector_store_path": str(chroma_path.as_posix()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build resume RAG index with ChromaDB")
    parser.add_argument("--resume-dir", default="data/resumes", help="Directory containing resume files")
    parser.add_argument("--chroma-path", default=".chroma_resume_index", help="Path for ChromaDB persistence")
    parser.add_argument("--collection", default="resume_chunks", help="Chroma collection name")
    parser.add_argument("--embedding-provider", choices=["openai", "cohere", "huggingface"], default="huggingface")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--cohere-api-key", default=os.getenv("COHERE_API_KEY"))
    parser.add_argument("--summary-path", default="rag_index_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_dir = Path(args.resume_dir)
    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    api_key = args.openai_api_key if args.embedding_provider == "openai" else args.cohere_api_key
    embedding_function = get_embedding_function(args.embedding_provider, args.embedding_model, api_key)
    collection = get_or_create_collection(Path(args.chroma_path), args.collection, embedding_function)

    resume_paths = list(iter_resume_files(resume_dir))
    if not resume_paths:
        raise RuntimeError(f"No supported resume files found in {resume_dir}")

    total_chunks = 0
    for resume_path in resume_paths:
        text = load_resume_text(resume_path)
        chunks = build_resume_chunks(resume_path, text)
        upsert_chunks(collection, chunks)
        total_chunks += len(chunks)

    write_summary(
        summary_path=Path(args.summary_path),
        resume_count=len(resume_paths),
        chunk_count=total_chunks,
        collection_name=args.collection,
        chroma_path=Path(args.chroma_path),
    )
    print(f"Indexed {len(resume_paths)} resumes into {args.collection} with {total_chunks} chunks.")


if __name__ == "__main__":
    main()
