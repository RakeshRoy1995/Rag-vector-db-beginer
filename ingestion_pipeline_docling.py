"""
Multimodal PDF extraction with Docling (MIT license, runs locally, free).

Produces extracted_layout.json in the same format as ingestion_pipeline_pdf.py,
so chunking_pipeline.py works unchanged:

    title / heading / paragraph / caption  -> "content"
    table                                  -> "title", "table_data", "image_path"
    figure                                 -> "title", "image_path"
    equation                               -> "text_fallback" (LaTeX), "image_path"

Items come out in true reading order (multi-column aware), so tables,
figures and equations land under the correct section heading.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)


# ============================================================
# CONFIG
# ============================================================

DOCS_PATH = "docs"
OUTPUT_JSON = "extracted_layout-docling.json"
IMAGE_DIR = "images"

IMAGE_SCALE = 2.0             # 2.0 ≈ 144 DPI crops

DO_OCR = False                # True for scanned PDFs (slower)
DO_FORMULA_LATEX = True       # Convert equations to LaTeX (downloads a small model)
DO_TABLE_STRUCTURE = True     # TableFormer cell structure

# Pre-downloaded models (`docling-tools models download`); avoids the
# HuggingFace cache, which needs symlink rights on Windows
MODELS_DIR = Path.home() / ".cache" / "docling" / "models"

MAX_NAME_CHARS = 60

FIGURE_LABEL_MARGIN = 80     # pt around a picture where labels live
MAX_LABEL_CHARS = 40          # Longer text is real content

NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")           # Keep image paths short (Windows 260-char limit)

# Page furniture that should never be embedded
SKIP_LABELS = {
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
}

CAPTION_PREFIX_RE = re.compile(
    r"^\s*(figure|fig\.?|table)\s*\d+[a-z]?\s*[:.\-]?\s*",
    re.IGNORECASE,
)


# ============================================================
# CONVERTER
# ============================================================

def build_converter() -> DocumentConverter:

    options = PdfPipelineOptions()

    if MODELS_DIR.exists():
        options.artifacts_path = MODELS_DIR

    options.do_ocr = DO_OCR
    options.do_table_structure = DO_TABLE_STRUCTURE
    options.do_formula_enrichment = DO_FORMULA_LATEX
    options.images_scale = IMAGE_SCALE
    options.generate_picture_images = True
    options.generate_table_images = True
    options.generate_page_images = DO_FORMULA_LATEX  # needed to crop equations

    # pypdfium keeps word spacing in headings that docling-parse loses
    # ("Scaled Dot-Product Attention" vs "ScaledDot-ProductAttention")
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


# ============================================================
# HELPERS
# ============================================================

def page_of(item) -> Optional[int]:
    return item.prov[0].page_no if item.prov else None


def safe_name(text: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return name[:MAX_NAME_CHARS] or "untitled"


def to_box(item, doc: DoclingDocument) -> Optional[tuple]:
    """(page, l, t, r, b) in top-left coordinates."""
    if not item.prov:
        return None
    prov = item.prov[0]
    page = doc.pages.get(prov.page_no)
    if page is None:
        return None
    bb = prov.bbox.to_top_left_origin(page_height=page.size.height)
    return prov.page_no, bb.l, bb.t, bb.r, bb.b


def is_figure_label(
    node,
    text: str,
    doc: DoclingDocument,
    picture_boxes: List[tuple],
) -> bool:
    """Short, unnumbered text inside or right next to a picture is a
    label drawn in the figure (axis title, box name), not a heading."""

    if len(text) > MAX_LABEL_CHARS or NUMBERED_HEADING_RE.match(text):
        return False

    box = to_box(node, doc)
    if box is None:
        return False

    page, l, t, r, b = box
    m = FIGURE_LABEL_MARGIN

    for p_page, pl, pt, pr, pb in picture_boxes:
        if p_page != page:
            continue
        horizontal = min(r, pr) - max(l, pl) > 0
        vertical = t < pb + m and b > pt - m
        if horizontal and vertical:
            return True

    return False


def table_to_markdown(table: TableItem) -> str:
    """Compact markdown (no cell padding, no caption)."""

    rows = []
    for row in table.data.grid:
        cells = [
            " ".join((cell.text or "").split()).replace("|", "/")
            for cell in row
        ]
        # Drop rows repeated by row-spanning cells
        if rows and cells == rows[-1]:
            continue
        rows.append(cells)

    if not rows:
        return ""

    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("|" + "---|" * len(rows[0]))
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def caption_title(item, doc: DoclingDocument) -> str:
    """'Figure 1: The Transformer' -> 'The Transformer'"""
    caption = item.caption_text(doc) or ""
    return CAPTION_PREFIX_RE.sub("", caption).strip()


def save_image(image, path: Path) -> Optional[str]:
    if image is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path.as_posix()


def crop_item(item, doc: DoclingDocument):
    """Crop an item's region from its rendered page image."""
    if not item.prov:
        return None

    prov = item.prov[0]
    page = doc.pages.get(prov.page_no)

    if page is None or page.image is None or page.image.pil_image is None:
        return None

    image = page.image.pil_image
    scale = image.width / page.size.width

    bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)

    pad = 4
    box = (
        max(0, int((bbox.l - pad) * scale)),
        max(0, int((bbox.t - pad) * scale)),
        min(image.width, int((bbox.r + pad) * scale)),
        min(image.height, int((bbox.b + pad) * scale)),
    )

    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    return image.crop(box)


# ============================================================
# ITEM CONVERSION
# ============================================================

def convert_document(
    doc: DoclingDocument,
    pdf_file: str,
    image_dir: Path,
) -> List[Dict[str, Any]]:

    stem = Path(pdf_file).stem
    items: List[Dict[str, Any]] = []

    # Captions already attached to a table/figure (avoid duplicates)
    attached_captions = {
        ref.cref
        for node in list(doc.tables) + list(doc.pictures)
        for ref in node.captions
    }

    # Picture boxes, to drop labels drawn inside figures
    picture_boxes = [
        box for box in (to_box(p, doc) for p in doc.pictures) if box
    ]

    counters = {"table": 0, "figure": 0, "equation": 0}

    for node, _level in doc.iterate_items():

        page = page_of(node)
        base = {"source": pdf_file, "page": page}

        # ---------------- Tables ----------------
        if isinstance(node, TableItem):

            markdown = table_to_markdown(node)

            # Drop empty "tables" (usually misdetected figures)
            if not re.search(r"\w", markdown):
                continue

            title = caption_title(node, doc)
            index = counters["table"]
            counters["table"] += 1

            path = image_dir / "tables" / (
                f"{stem}_p{page}_table_{index}_{safe_name(title)}.png"
            )

            item = {
                "type": "table",
                **base,
                "table_data": markdown,
                "image_path": save_image(node.get_image(doc), path),
            }
            if title:
                item["title"] = title

            items.append(item)
            print(f"    p{page} table   : {title[:60] or '(no caption)'}")
            continue

        # ---------------- Figures ----------------
        if isinstance(node, PictureItem):

            title = caption_title(node, doc)
            index = counters["figure"]
            counters["figure"] += 1

            path = image_dir / (
                f"{stem}_p{page}_figure_{index}_{safe_name(title)}.png"
            )

            items.append({
                "type": "figure",
                **base,
                "title": title,
                "image_path": save_image(node.get_image(doc), path),
            })
            print(f"    p{page} figure  : {title[:60] or '(no caption)'}")
            continue

        # ---------------- Text-like ----------------
        if not isinstance(node, TextItem):
            continue

        if node.label in SKIP_LABELS:
            continue

        text = (node.text or "").replace("\r\n", "\n").strip()
        if not text:
            continue

        # Axis / box labels drawn in a figure are not content
        if node.label in (
            DocItemLabel.TITLE,
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.TEXT,
        ) and is_figure_label(node, text, doc, picture_boxes):
            continue

        if isinstance(node, (TitleItem, SectionHeaderItem)):

            # A real heading is one line; extra lines are figure text
            # merged in by the layout model
            text = text.splitlines()[0].strip()

        if isinstance(node, TitleItem):
            items.append({"type": "title", **base, "content": text})
            continue

        if isinstance(node, SectionHeaderItem):
            items.append({"type": "heading", **base, "content": text})
            continue

        if node.label == DocItemLabel.FORMULA:

            index = counters["equation"]
            counters["equation"] += 1

            path = image_dir / "equations" / (
                f"{stem}_p{page}_equation_{index}.png"
            )

            items.append({
                "type": "equation",
                **base,
                "text_fallback": text,
                "image_path": save_image(crop_item(node, doc), path),
            })
            continue

        if node.label == DocItemLabel.CAPTION:
            if node.self_ref in attached_captions:
                continue
            items.append({"type": "caption", **base, "content": text})
            continue

        # text, list_item, footnote, code, reference, ...
        items.append({"type": "paragraph", **base, "content": text})

    return items


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_pdf_data(
    docs_path: str = DOCS_PATH,
    image_dir: str = IMAGE_DIR,
) -> List[Dict[str, Any]]:

    pdf_files = sorted(Path(docs_path).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{docs_path}'")

    converter = build_converter()
    data: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        result = converter.convert(pdf_path)

        items = convert_document(
            result.document,
            pdf_path.name,
            Path(image_dir),
        )

        print(f"  -> {len(items)} items")
        data.extend(items)

    return data


# ============================================================
# UTILITIES
# ============================================================

def print_summary(data: List[Dict[str, Any]]):

    counts: Dict[str, int] = {}
    for item in data:
        counts[item["type"]] = counts.get(item["type"], 0) + 1

    print("\n" + "=" * 65)
    print(f"Total items extracted: {len(data)}")
    for item_type, count in sorted(counts.items()):
        print(f"  {item_type:10}: {count}")


def save_json(data: List[Dict[str, Any]], output_json: str = OUTPUT_JSON):

    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(data)} items -> {path.resolve()}")


def main(
    docs_path: str = DOCS_PATH,
    output_json: str = OUTPUT_JSON,
):
    data = extract_pdf_data(docs_path)
    print_summary(data)
    save_json(data, output_json)
    return data


if __name__ == "__main__":
    main()
