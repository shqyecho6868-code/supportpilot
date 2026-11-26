"""Policy documents and chunking.

Documents are Markdown files in ``kb/policies`` with a small front-matter block
(``doc_id``, ``title``) and ``## Section`` headings. Sections are the unit of
citation ("Returns Policy § Electronics"), so a chunk never spans two sections;
a long section is split on paragraph boundaries.
"""

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[3] / "kb"
POLICY_DIR = KB_DIR / "policies"

MAX_CHUNK_WORDS = 220

_FRONT_MATTER = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.S)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str  # body without front matter


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    section: str
    chunk_index: int
    content: str

    @property
    def source(self) -> str:
        return f"{self.doc_title} § {self.section}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_document(raw: str) -> Document:
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise ValueError("policy document is missing its front matter")
    meta = dict(line.split(":", 1) for line in match["body"].splitlines() if ":" in line)
    return Document(
        doc_id=meta["doc_id"].strip(), title=meta["title"].strip(), text=raw[match.end() :].strip()
    )


def load_documents(directory: Path = POLICY_DIR) -> list[Document]:
    return [parse_document(p.read_text()) for p in sorted(directory.glob("*.md"))]


def _split_long(body: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        if current and len(" ".join([*current, paragraph]).split()) > MAX_CHUNK_WORDS:
            parts.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_document(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = re.split(r"^## +", doc.text, flags=re.M)
    for block in sections:
        if not block.strip():
            continue
        heading, _, body = block.partition("\n")
        heading = heading.strip()
        parts = _split_long(body.strip())
        for i, part in enumerate(parts):
            chunk_id = f"{doc.doc_id}#{_slug(heading)}" + (f"-{i + 1}" if len(parts) > 1 else "")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section=heading,
                    chunk_index=len(chunks),
                    # Keep the heading: it is what a naive Markdown splitter would produce.
                    content=f"## {heading}\n\n{part}",
                )
            )
    return chunks
