"""
Turn extracted_layout.json (from ingestion_pipeline_docling.py) into
section-aware RAG chunks in chunks.json.

Each chunk has:
    content         clean text to show the user / pass to the LLM
    embedding_text  content + "Document / Section" header, for the vector
    metadata        section path, pages, parent_id, image paths, ...
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIG
# ============================================================

INPUT_JSON = "extracted_layout.json"
OUTPUT_JSON = "chunks.json"
HIERARCHY_JSON: Optional[str] = None   # e.g. "hierarchy.json" to debug sections

CHUNK_SIZE = 800          # max characters per chunk
CHUNK_OVERLAP = 100       # characters repeated between split pieces
MIN_SPLIT_RATIO = 0.4     # never cut a piece shorter than this share of CHUNK_SIZE
MIN_CHUNK_CHARS = 50      # tiny trailing chunks merge into the previous one

HEADING_TYPES = ("title", "heading")
STANDALONE_TYPES = ("table", "figure")

SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*$")
NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$", re.S)
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")

# Split points, best first
BOUNDARIES = ["\n\n", ". ", "? ", "! ", "\n", " "]


# ============================================================
# TEXT
# ============================================================

def clean_text(text: Any) -> str:
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def unwrap_lines(text: str) -> str:
    """Undo PDF line wrapping: 'multi-\\nplicative' -> 'multiplicative'."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def item_text(item: Dict[str, Any]) -> str:
    """Searchable text for one extracted item."""
    item_type = item.get("type")

    if item_type == "table":
        title = clean_text(item.get("title"))
        data = clean_text(item.get("table_data"))
        return "\n".join(x for x in (f"Table: {title}" if title else "", data) if x)

    if item_type == "figure":
        title = clean_text(item.get("title"))
        return f"Figure: {title}" if title else ""

    if item_type == "equation":
        return clean_text(item.get("text_fallback"))

    return unwrap_lines(clean_text(item.get("content")))


# ============================================================
# SPLITTING
# ============================================================

def find_split(text: str, start: int, end: int) -> int:
    """Best boundary in text[start:end]: paragraph > sentence > line > word."""
    min_pos = start + int((end - start) * MIN_SPLIT_RATIO)

    for sep in BOUNDARIES:
        pos = text.rfind(sep, min_pos, end)
        if pos != -1:
            return pos + len(sep)

    return end


def overlap_start(text: str, end: int, overlap: int) -> int:
    """Step back ~overlap chars, then forward to the next word start."""
    pos = max(0, end - overlap)
    if pos > 0 and not text[pos - 1].isspace():
        space = text.find(" ", pos, end)
        pos = space + 1 if space != -1 else end
    return pos


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = find_split(text, start, end)

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= len(text):
            break

        next_start = overlap_start(text, end, chunk_overlap)
        start = next_start if next_start > start else end

    return pieces


def split_table(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split a markdown table by rows, repeating title + header in each piece."""
    lines = text.splitlines()

    title = []
    while lines and not lines[0].lstrip().startswith("|"):
        title.append(lines.pop(0))

    if len(lines) < 3:
        return split_text(text, chunk_size)

    header = title + lines[:2]
    header_len = sum(len(h) + 1 for h in header)

    pieces, rows, size = [], [], header_len
    for row in lines[2:]:
        if rows and size + len(row) + 1 > chunk_size:
            pieces.append("\n".join(header + rows))
            rows, size = [], header_len
        rows.append(row)
        size += len(row) + 1

    if rows:
        pieces.append("\n".join(header + rows))

    return pieces


# ============================================================
# SECTIONS
# ============================================================

def build_sections(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group items under headings and give each section a breadcrumb path.

    Handles both "3.2.1\\nTitle" and a bare "3.2.1" heading followed by "Title".
    """
    sections: List[Dict[str, Any]] = []
    stack: Dict[int, str] = {}       # depth -> title, for numbered headings
    pending_number: Optional[str] = None
    current: Optional[Dict[str, Any]] = None

    for index, item in enumerate(items):
        item_type = item.get("type")

        if item_type in HEADING_TYPES:
            text = clean_text(item.get("content"))

            if SECTION_NUMBER_RE.fullmatch(text):
                pending_number = text
                continue

            match = NUMBERED_HEADING_RE.match(text)
            number = match.group(1) if match else pending_number
            name = unwrap_lines(match.group(2) if match else text)
            pending_number = None

            title = f"{number} {name}" if number else name

            if number:
                depth = number.count(".") + 1
                stack = {d: t for d, t in stack.items() if d < depth}
                stack[depth] = title
                path = " > ".join(stack[d] for d in sorted(stack))
            else:
                stack = {}
                path = title

            current = {
                "parent_id": f"section_{len(sections) + 1:04d}",
                "type": item_type,
                "section_title": title,
                "section_path": path,
                "source": item.get("source", ""),
                "page": item.get("page"),
                "items": [],
            }
            sections.append(current)
            continue

        # Text before the first heading gets its own section
        if current is None:
            current = {
                "parent_id": "section_0000",
                "type": "preamble",
                "section_title": "",
                "section_path": "",
                "source": item.get("source", ""),
                "page": item.get("page"),
                "items": [],
            }
            sections.append(current)

        text = item_text(item)
        if not text:
            continue
        if item_type == "paragraph" and PAGE_NUMBER_RE.fullmatch(text):
            continue

        current["items"].append({**item, "item_index": index, "text": text})

    return [s for s in sections if s["items"]]


# ============================================================
# CHUNKS
# ============================================================

def to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_embedding_text(document: str, section_path: str, content: str) -> str:
    header = f"Document: {document}\n"
    if section_path:
        header += f"Section: {section_path}\n"
    return f"{header}\n{content}"


def make_chunk(
    section: Dict[str, Any],
    items: List[Dict[str, Any]],
    content: str,
    chunk_type: str,
) -> Dict[str, Any]:

    pages = [p for p in (to_int(i.get("page")) for i in items) if p is not None]
    image_paths = [i["image_path"] for i in items if i.get("image_path")]

    chunk = {
        "parent_id": section["parent_id"],
        "item_indices": [i["item_index"] for i in items],
        "type": chunk_type,
        "source": section["source"],
        "section_title": section["section_title"],
        "section_path": section["section_path"],
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "content": content,
    }
    if image_paths:
        chunk["image_paths"] = image_paths

    return chunk


def chunk_section(
    section: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """Pack consecutive text items up to chunk_size; tables and figures
    become their own chunks."""

    chunks: List[Dict[str, Any]] = []
    buffer: List[Dict[str, Any]] = []

    def flush():
        if not buffer:
            return
        joined = "\n\n".join(i["text"] for i in buffer)
        for piece in split_text(joined, chunk_size, chunk_overlap):
            chunks.append(make_chunk(section, list(buffer), piece, "text"))
        buffer.clear()

    for item in section["items"]:
        item_type = item.get("type")

        if item_type in STANDALONE_TYPES:
            flush()
            pieces = (
                split_table(item["text"], chunk_size)
                if item_type == "table"
                else split_text(item["text"], chunk_size, chunk_overlap)
            )
            for piece in pieces:
                chunks.append(make_chunk(section, [item], piece, item_type))
            continue

        size = sum(len(i["text"]) + 2 for i in buffer)
        if buffer and size + len(item["text"]) > chunk_size:
            flush()
        buffer.append(item)

    flush()

    # Merge a tiny trailing text chunk into the previous text chunk
    if (
        len(chunks) >= 2
        and chunks[-1]["type"] == chunks[-2]["type"] == "text"
        and len(chunks[-1]["content"]) < MIN_CHUNK_CHARS
    ):
        last = chunks.pop()
        prev = chunks[-1]
        prev["content"] += "\n\n" + last["content"]
        prev["item_indices"] += last["item_indices"]
        prev["page_end"] = last["page_end"] or prev["page_end"]

    return chunks


def stable_id(source: str, section_path: str, content: str) -> str:
    """Same content -> same id across runs, so vector DB upserts stay in sync."""
    stem = Path(source).stem or "document"
    digest = hashlib.sha1(
        f"{source}\x00{section_path}\x00{content}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{stem}_{digest}"


def create_chunks(
    sections: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:

    # First heading of each source is its document title
    documents: Dict[str, str] = {}
    for section in sections:
        if section["section_title"]:     # skip the untitled preamble
            documents.setdefault(section["source"], section["section_title"])

    chunks: List[Dict[str, Any]] = []
    seen = set()

    for section in sections:
        for chunk in chunk_section(section, chunk_size, chunk_overlap):
            chunk_id = stable_id(chunk["source"], chunk["section_path"], chunk["content"])

            # Skip exact duplicates (repeated boilerplate)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            document = documents.get(chunk["source"]) or Path(chunk["source"]).stem
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": len(chunks),
                **chunk,
                "embedding_text": build_embedding_text(
                    document, chunk["section_path"], chunk["content"]
                ),
                "char_count": len(chunk["content"]),
            })

    return chunks


# ============================================================
# I/O + REPORTING
# ============================================================

def load_json(path: str) -> List[Dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Input file not found: {file.resolve()}")

    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain an array of objects.")
    return data


def save_json(data: Any, path: str):
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {file.resolve()}")


def print_summary(sections: List[Dict[str, Any]], chunks: List[Dict[str, Any]]):
    print(f"\nSections: {len(sections)}")
    for section in sections:
        print(f"  p{section['page']!s:>3}  {section['section_path'] or '(preamble)'}"
              f"  [{len(section['items'])} items]")

    lengths = [c["char_count"] for c in chunks] or [0]
    print(f"\nChunks: {len(chunks)}  "
          f"(chars min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)})")
    for chunk_type, count in Counter(c["type"] for c in chunks).most_common():
        print(f"  {chunk_type:8}: {count}")


# ============================================================
# MAIN
# ============================================================

def main(input_json: str = INPUT_JSON, output_json: str = OUTPUT_JSON):
    items = load_json(input_json)
    print(f"Loaded {len(items)} items from {input_json}")

    sections = build_sections(items)
    chunks = create_chunks(sections)

    print_summary(sections, chunks)

    if HIERARCHY_JSON:
        save_json(sections, HIERARCHY_JSON)
    save_json(chunks, output_json)
    return chunks


if __name__ == "__main__":
    main()
