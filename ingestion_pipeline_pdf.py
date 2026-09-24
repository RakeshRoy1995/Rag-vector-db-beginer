import os
import base64
import html
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import fitz  # PyMuPDF


# ============================================================
# CONFIG
# ============================================================

IMAGE_SCALE = 2
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
FIGURE_PADDING_X = 8
FIGURE_PADDING_Y = 8
EQUATION_PADDING = 6
MIN_FIGURE_WIDTH = 80
MIN_FIGURE_HEIGHT = 40
MAX_FIGURE_WIDTH_RATIO = 0.95

MAX_EQUATION_CHARS = 300
MIN_EQUATION_CHARS = 12
MIN_MATH_SCORE = 3


# ============================================================
# HELPERS
# ============================================================

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def crop_to_base64(page, rect, scale: int = IMAGE_SCALE) -> Optional[str]:
    rect = fitz.Rect(rect)
    rect &= page.rect
    if rect.is_empty:
        return None
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    return image_to_base64(pix.tobytes("png"))


def table_to_html(table) -> str:
    rows = table.extract()
    if not rows:
        return "<table></table>"

    html_rows = []
    header = rows[0]
    header_html = "".join(f"<th>{html.escape(str(cell or ''))}</th>" for cell in header)
    html_rows.append(f"<thead><tr>{header_html}</tr></thead>")

    body = []
    for row in rows[1:]:
        cells = "".join(f"<td>{html.escape(str(cell or ''))}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    html_rows.append("<tbody>" + "".join(body) + "</tbody>")

    return "<table>" + "".join(html_rows) + "</table>"


def is_valid_table(table) -> bool:
    try:
        rows = table.extract()
        if not rows:
            return False
        rows = [r for r in rows if any(str(c or "").strip() for c in r)]
        if len(rows) < MIN_TABLE_ROWS:
            return False
        if max(len(r) for r in rows) < MIN_TABLE_COLS:
            return False
        non_empty = sum(1 for r in rows for c in r if str(c or "").strip())
        return non_empty >= 4
    except Exception:
        return False

# ============================================================
# LAYOUT ANALYSIS – Get rich text blocks
# ============================================================

def get_rich_blocks(page) -> List[Dict]:
    """
    Extract text blocks with layout information:
    - bbox
    - text
    - average font size
    - is_bold
    """
    blocks = []
    data = page.get_text("dict", sort=True)

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        bbox = block.get("bbox")
        if not bbox:
            continue

        rect = fitz.Rect(bbox)
        lines = []
        font_sizes = []
        is_bold = False

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                text = span.get("text", "")
                line_text += text
                size = span.get("size", 0)
                if size > 0:
                    font_sizes.append(size)
                # bold flag
                if span.get("flags", 0) & 2**4:
                    is_bold = True
            if line_text.strip():
                lines.append(line_text.strip())

        text = "\n".join(lines).strip()
        if not text:
            continue

        avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10.0

        blocks.append({
            "bbox": rect,
            "text": text,
            "font_size": avg_size,
            "is_bold": is_bold,
            "y0": rect.y0,
            "x0": rect.x0,
        })

    # Sort top-to-bottom, left-to-right
    blocks.sort(key=lambda b: (round(b["y0"], 1), b["x0"]))
    return blocks

# ============================================================
# CLASSIFY BLOCK (Layout Analysis)
# ============================================================

def classify_block(block: Dict, page_height: float, median_font: float) -> str:
    """
    Simple but effective layout classification.
    Returns one of: title, heading, paragraph, caption, equation
    """
    text = block["text"].strip()
    size = block["font_size"]
    is_bold = block["is_bold"]
    y0 = block["y0"]

    # Very top of page + large font → title
    if y0 < page_height * 0.15 and size > median_font * 1.4:
        return "title"

    # Section heading patterns
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", text) or re.match(r"^[A-Z][A-Za-z\s]{3,40}$", text):
        if size >= median_font * 1.1 or is_bold:
            return "heading"

    # Caption
    if re.match(r"^(Figure|Fig\.|Table)\s+\d+", text, re.IGNORECASE):
        return "caption"

    # Equation detection
    if is_likely_equation(text):
        return "equation"

    return "paragraph"


def math_score(text: str) -> int:
    score = 0
    strong = [
        r"MultiHead\s*\(", r"Attention\s*\(", r"FFN\s*\(", r"softmax\s*\(",
        r"LayerNorm\s*\(", r"PE\s*\(", r"d_{?model}?", r"d_{?k}?", r"d_{?v}?",
        r"W\^?[QKVO]", r"head_?\d", r"∈\s*R", r"\\sqrt", r"\\frac"
    ]
    for p in strong:
        if re.search(p, text, re.IGNORECASE):
            score += 2

    medium = [
        r"[A-Za-z]_{?[0-9i-n]+}?", r"[A-Za-z]\^{[0-9i-nQKVO]+}",
        r"=\s*[A-Za-z\\(]", r"[∑∫∏√∞≈≠≤≥∈∉→←]"
    ]
    for p in medium:
        if re.search(p, text):
            score += 1

    math_chars = sum(1 for c in text if c in "=+-*/^_()[]{}\\∈×÷∑∫")
    density = math_chars / max(len(text), 1)
    if density > 0.22:
        score += 2
    elif density > 0.15:
        score += 1
    return score


def is_likely_equation(text: str) -> bool:
    text = text.strip()
    if len(text) > MAX_EQUATION_CHARS or len(text) < MIN_EQUATION_CHARS:
        return False
    if text.count(". ") >= 2:
        return False
    if text.lower().startswith(("the ", "in this", "we ", "this ", "our ", "figure", "table")):
        return False
    return math_score(text) >= MIN_MATH_SCORE


# ============================================================
# EXTRACT EQUATIONS
# ============================================================

def extract_equations(page, blocks: List[Dict], pdf_file: str, page_number: int):
    equations = []
    eq_rects = []

    for block in blocks:
        if classify_block(block, page.rect.height, 11.0) != "equation":
            continue

        rect = fitz.Rect(block["bbox"])
        rect.x0 -= EQUATION_PADDING
        rect.x1 += EQUATION_PADDING
        rect.y0 -= EQUATION_PADDING
        rect.y1 += EQUATION_PADDING
        rect &= page.rect

        b64 = crop_to_base64(page, rect)
        if not b64:
            continue

        print(f"    Equation: {block['text'][:65]}...")
        equations.append({
            "type": "equation",
            "source": pdf_file,
            "page": page_number,
            "text_fallback": block["text"],
            "image_base64": b64,
        })
        eq_rects.append(rect)

    return equations, eq_rects


# ============================================================
# FIGURES
# ============================================================

def find_figure_captions(blocks: List[Dict]) -> List[Dict]:
    captions = []
    pattern = re.compile(r"^\s*(figure|fig\.?)\s*\d+[a-zA-Z]?\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
    for block in blocks:
        for line in block["text"].splitlines():
            m = pattern.match(line.strip())
            if m and m.group(2).strip():
                captions.append({
                    "bbox": block["bbox"],
                    "title": m.group(2).strip(),
                    "caption": line.strip(),
                })
    return captions


def get_image_rects(page) -> List[fitz.Rect]:
    rects = []
    try:
        for img in page.get_image_info(xrefs=True):
            bbox = img.get("bbox")
            if bbox:
                r = fitz.Rect(bbox)
                if not r.is_empty:
                    rects.append(r)
    except Exception:
        pass
    return rects


def get_drawing_rects(page) -> List[fitz.Rect]:
    rects = []
    try:
        for d in page.get_drawings():
            if "rect" in d:
                r = fitz.Rect(d["rect"])
                if not r.is_empty and not (r.width < 3 and r.height < 3):
                    rects.append(r)
    except Exception:
        pass
    return rects


def union_rects(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects:
        return None
    result = fitz.Rect(rects[0])
    for r in rects[1:]:
        result |= fitz.Rect(r)
    return result


def find_figure_region(page, caption_bbox) -> Optional[fitz.Rect]:
    page_rect = page.rect
    visual = get_image_rects(page) + get_drawing_rects(page)
    if not visual:
        return None

    candidates = []
    cap_cy = (caption_bbox.y0 + caption_bbox.y1) / 2

    for rect in visual:
        if rect.width < MIN_FIGURE_WIDTH or rect.height < MIN_FIGURE_HEIGHT:
            continue
        if rect.width < page_rect.width * 0.05:
            continue

        if rect.y1 < caption_bbox.y0:
            dist = caption_bbox.y0 - rect.y1
        elif rect.y0 > caption_bbox.y1:
            dist = rect.y0 - caption_bbox.y1
        else:
            dist = 0

        if dist > page_rect.height * 0.40:
            continue

        center_dist = abs((rect.y0 + rect.y1) / 2 - cap_cy)
        candidates.append({"rect": rect, "distance": dist, "center_distance": center_dist})

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["distance"], x["center_distance"]))
    best = candidates[0]["rect"]
    selected = [best]

    for c in candidates[1:]:
        r = c["rect"]
        exp = fitz.Rect(best)
        exp.x0 -= 20
        exp.x1 += 20
        exp.y0 -= 20
        exp.y1 += 20
        if exp.intersects(r) or abs(r.y0 - best.y1) < 20 or abs(best.y0 - r.y1) < 20:
            selected.append(r)
            best |= r

    fig = union_rects(selected)
    if not fig:
        return None

    fig.x0 -= FIGURE_PADDING_X
    fig.x1 += FIGURE_PADDING_X
    fig.y0 -= FIGURE_PADDING_Y
    fig.y1 += FIGURE_PADDING_Y
    fig &= page_rect

    if fig.width > page_rect.width * MAX_FIGURE_WIDTH_RATIO:
        return None
    return fig


def extract_figures(
    page,
    blocks: List[Dict],
    pdf_file: str,
    page_number: int,
    image_dir: str = "images",
) -> List[Dict]:

    results = []

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    for index, cap in enumerate(find_figure_captions(blocks)):
        print(f"    Figure found: {cap['title']}")

        region = find_figure_region(page, cap["bbox"])

        if not region:
            continue

        # Create a safe unique filename
        pdf_name = Path(pdf_file).stem
        safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", cap["title"])

        image_name = (
            f"{pdf_name}_page_{page_number}_figure_{index}_{safe_title}.png"
        )

        image_path = image_dir / image_name

        # Save cropped PDF region as PNG
        pix = page.get_pixmap(
            clip=region,
            dpi=150,
            alpha=False
        )

        pix.save(str(image_path))

        item = {
            "type": "figure",
            "source": pdf_file,
            "page": page_number,
            "title": cap["title"],
            "image_path": str(image_path),
        }

        results.append(item)

    return results

# ============================================================
# TABLES
# ============================================================

def find_valid_tables(page) -> List:
    try:
        tables = page.find_tables(use_layout=True).tables
    except Exception as e:
        print(f"    Table detection error: {e}")
        return []

    valid = []
    for i, t in enumerate(tables, 1):
        if is_valid_table(t):
            print(f"    Valid table #{i} found")
            valid.append(t)
        else:
            print(f"    Ignored false table candidate #{i}")
    return valid


def find_table_caption(blocks: List[Dict], table_rect) -> Optional[str]:
    pattern = re.compile(r"^\s*table\s*\d+[a-zA-Z]?\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
    candidates = []
    for block in blocks:
        if block["bbox"].y1 > table_rect.y0:
            continue
        dist = table_rect.y0 - block["bbox"].y1
        if dist > 100:
            continue
        for line in block["text"].splitlines():
            m = pattern.match(line.strip())
            if m:
                candidates.append((dist, m.group(1).strip()))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def extract_tables(
    page,
    blocks: List[Dict],
    pdf_file: str,
    page_number: int,
    image_dir: str = "images/tables",
):
    docs = []
    valid = find_valid_tables(page)

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_file).stem

    for index, table in enumerate(valid):
        rect = fitz.Rect(table.bbox)

        # Find table title
        title = find_table_caption(blocks, rect)

        if title:
            safe_title = "".join(
                c if c.isalnum() or c in "_-" else "_"
                for c in title
            )
        else:
            safe_title = f"table_{index}"

        # Stable image ID
        image_id = (
            f"{pdf_name}_page_{page_number}_table_{index}"
        )

        image_path = image_dir / f"{image_id}_{safe_title}.png"

        # Crop table and save as PNG
        pix = page.get_pixmap(
            clip=rect,
            dpi=150,
            alpha=False,
        )

        pix.save(str(image_path))

        item = {
            "type": "table",
            "source": pdf_file,
            "page": page_number,
            "table_data": table_to_html(table),
            "image_id": image_id,
            "image_path": str(image_path),
        }

        if title:
            item["title"] = title
            print(f"      Table title: {title}")

        print(f"      Table image: {image_path}")

        docs.append(item)

    rects = [fitz.Rect(t.bbox) for t in valid]

    return docs, rects

# ============================================================
# MAIN EXTRACTION WITH LAYOUT ANALYSIS
# ============================================================

def extract_pdf_data(docs_path: str = "docs") -> List[Dict[str, Any]]:
    print("\n" + "=" * 65)
    print("Layout Analysis + LaTeX-free Extraction")
    print("=" * 65)

    document_data = []
    pdf_files = [f for f in os.listdir(docs_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{docs_path}'")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_path, pdf_file)
        print(f"\nProcessing: {pdf_file}")

        doc = fitz.open(pdf_path)
        try:
            for page_num, page in enumerate(doc, start=1):
                print(f"\n  Page {page_num}")

                # 1. Rich layout blocks
                blocks = get_rich_blocks(page)

                # Calculate median font size for classification
                sizes = [b["font_size"] for b in blocks if b["font_size"] > 0]
                median_font = sorted(sizes)[len(sizes)//2] if sizes else 11.0

                # 2. Tables
                table_docs, table_rects = extract_tables(page, blocks, pdf_file, page_num)

                # 3. Figures
                figure_docs = extract_figures(page, blocks, pdf_file, page_num)

                # 4. Equations
                equation_docs, equation_rects = extract_equations(page, blocks, pdf_file, page_num)

                # 5. Classify remaining text blocks
                exclude_rects = table_rects + equation_rects

                for block in blocks:
                    # Skip if overlapping with table/equation
                    skip = False
                    for er in exclude_rects:
                        inter = block["bbox"] & er
                        if not inter.is_empty:
                            area = block["bbox"].width * block["bbox"].height
                            if area > 0 and (inter.width * inter.height) / area >= 0.45:
                                skip = True
                                break
                    if skip:
                        continue

                    label = classify_block(block, page.rect.height, median_font)

                    # We only keep useful text types
                    if label in ("title", "heading", "paragraph", "caption"):
                        document_data.append({
                            "type": label,          # title / heading / paragraph / caption
                            "source": pdf_file,
                            "page": page_num,
                            "content": block["text"],
                            "font_size": round(block["font_size"], 1),
                        })

                document_data.extend(table_docs)
                document_data.extend(figure_docs)
                document_data.extend(equation_docs)

        finally:
            doc.close()

    print(f"\nTotal items extracted: {len(document_data)}")
    return document_data


# ============================================================
# UTILITIES
# ============================================================

def print_sample(data: List[Dict], limit: int = 12):
    print("\n" + "=" * 65)
    print("SAMPLE OUTPUT")
    print("=" * 65)

    for item in data[:limit]:
        print("\n" + "-" * 55)
        print(f"Type   : {item['type']}")
        print(f"Page   : {item['page']}")

        if item["type"] in ("title", "heading", "paragraph", "caption"):
            print(f"Content: {item['content'][:120]}...")
        elif item["type"] == "table":
            print(f"Title  : {item.get('title', '-')}")
            print(f"HTML   : {item['table_data'][:100]}...")
        elif item["type"] == "figure":
            print(f"Title  : {item.get('title', '-')}")
        elif item["type"] == "equation":
            print(f"Fallback: {item.get('text_fallback', '')[:80]}...")


def process_and_save(data: List[Dict[str, Any]], output_json: Optional[str] = None):
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(data)} items → {path.resolve()}")
    return data


# ============================================================
# MAIN
# ============================================================

def main(docs_path: str = "docs", output_json: Optional[str] = "extracted_layout.json"):
    data = extract_pdf_data(docs_path)
    print_sample(data)
    process_and_save(data, output_json)
    return data


if __name__ == "__main__":
    main()