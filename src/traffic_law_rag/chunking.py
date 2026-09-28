#!/usr/bin/env python3
"""Clause-level chunker for Vietnamese legal documents.

Parses Vietnamese legal texts (Luật, Nghị định, Thông tư) into semantic
chunks aligned to the legal hierarchy: Phần → Chương → Mục → Điều →
Khoản → Điểm.  Each chunk carries contextualised metadata and an
embedding-ready text field.

Output schema per chunk::

    {
        "id":                       str,
        "content":                  str,
        "contextualized_content":   str,
        "embed_text":               str,
        "metadata":                 dict
    }
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from .tokenizer import count_tokens
from .config import LEGAL_DOCS_DIR, PROCESSED_DIR, MAX_TOKENS_PER_CHUNK


# ============================================================================
# Configuration
# ============================================================================

INPUT_FOLDER = str(LEGAL_DOCS_DIR)
OUTPUT_FILE = str(PROCESSED_DIR / "semantic_chunks.json")

TOKEN_FALLBACK_RATIO = 1.35


# ============================================================================
# Regex patterns for legal structure elements
# ============================================================================

PART_RE = re.compile(r"^Phần\s+(?:thứ\s+)?(.+)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^Chương\s+([IVXLC]+|\d+)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^Mục\s+(\d+[A-Z]?)\.\s*(.*)$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^Điều\s+(\d+[a-zA-Z]?)\.\s*(.*)$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^(\d+[a-zA-Z]?)\.\s+(.+)$", re.IGNORECASE)
POINT_RE = re.compile(r"^([a-zđ])\)\s*(.*)$", re.IGNORECASE)


# ============================================================================
# Text helpers
# ============================================================================

def clean_text(value: str) -> str:
    """Normalise whitespace and strip a text fragment."""
    value = (value or "").replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s+\n", "\n", value)
    value = re.sub(r"\n\s+", "\n", value)
    return value.strip()


def normalize_spaces(value: str) -> str:
    """Collapse all whitespace into single spaces."""
    return re.sub(r"\s+", " ", value or "").strip()


def unique(values: list[str]) -> list[str]:
    """Return *values* with duplicates removed, preserving order."""
    return list(dict.fromkeys(v for v in values if v))


def parse_int(value: str) -> Optional[int]:
    """Safe integer conversion; returns ``None`` on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimate_tokens(text: str) -> int:
    """Token count via the shared BGE-M3 tokenizer helper."""
    return count_tokens(text)


def split_sentences(text: str) -> list[str]:
    """Split *text* on sentence-ending punctuation or newlines."""
    parts = re.split(r"(?<=[.;:!?])\s+|\n+", clean_text(text))
    return [part.strip() for part in parts if part.strip()]


def split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split free text by sentence/word boundaries.

    Used as a fallback when no legal sub-points are available.
    """
    text = clean_text(text)
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(" ".join(current).strip())
        current = []

    for sentence in split_sentences(text):
        if estimate_tokens(sentence) > max_tokens:
            flush()
            words = sentence.split()
            block: list[str] = []
            for word in words:
                candidate = " ".join(block + [word])
                if block and estimate_tokens(candidate) > max_tokens:
                    chunks.append(" ".join(block).strip())
                    block = [word]
                else:
                    block.append(word)
            if block:
                chunks.append(" ".join(block).strip())
            continue

        candidate = " ".join(current + [sentence])
        if current and estimate_tokens(candidate) > max_tokens:
            flush()
        current.append(sentence)

    flush()
    return chunks


# ============================================================================
# Document metadata extraction
# ============================================================================

def normalize_document_metadata(doc_name: str) -> tuple[str, str, str]:
    """Derive ``(code, id, type)`` from the document filename."""
    base = os.path.splitext(os.path.basename(doc_name))[0]
    parts = base.split("_")
    if len(parts) >= 3:
        number = parts[0]
        year = parts[1]
        type_code = parts[2].upper()
        document_id = f"{number}/{year}/{type_code}"
        document_code = "_".join(parts[:3])
        if type_code.startswith("ND"):
            return document_code, document_id, "Nghị định"
        if type_code.startswith("TT"):
            return document_code, document_id, "Thông tư"
        if type_code.startswith("QH"):
            return document_code, document_id, "Luật"
        return document_code, document_id, type_code
    return base, base, ""


def title_case_vietnamese(text: str) -> str:
    """Capitalise only the first character (Vietnamese title convention)."""
    text = normalize_spaces(text).lower()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def extract_document_name(
    lines: list[str], document_type: str, document_id: str
) -> str:
    """Heuristically extract the document title from its first lines."""
    if not document_type:
        return document_id

    type_upper = document_type.upper()
    for index, line in enumerate(lines[:40]):
        if normalize_spaces(line).upper() != type_upper:
            continue
        title_lines: list[str] = []
        for next_line in lines[index + 1 : index + 8]:
            clean = normalize_spaces(next_line)
            if not clean:
                continue
            if re.match(
                r"^(Căn cứ|Quốc hội|Chính phủ|Bộ trưởng|Điều|Chương|Phần)\b",
                clean,
                re.IGNORECASE,
            ):
                break
            if len(clean) > 180:
                break
            title_lines.append(clean)
        if title_lines:
            return (
                f"{document_type} "
                f"{title_case_vietnamese(' '.join(title_lines))}"
            )

    return f"{document_type} {document_id}".strip()


# ============================================================================
# Semantic enrichment (aliases & roles)
# ============================================================================

def detect_aliases(text: str) -> list[str]:
    """Return synonym keywords relevant to the chunk content."""
    lower = text.lower()
    aliases: list[str] = []

    checks = [
        (("vi phạm hành chính",),
         ["hành vi vi phạm", "xử lý vi phạm", "xử phạt hành chính"]),
        (("xử phạt", "phạt tiền"),
         ["xử phạt hành chính", "mức phạt", "xử lý vi phạm"]),
        (("tước quyền", "đình chỉ"),
         ["tước giấy phép", "đình chỉ giấy phép", "thu hồi giấy phép"]),
        (("trừ điểm",),
         ["trừ điểm giấy phép lái xe", "điểm GPLX"]),
        (("đèn đỏ", "đèn tín hiệu", "tín hiệu giao thông",
          "tín hiệu đèn", "hiệu lệnh của đèn", "hiệu lệnh đèn"),
         ["vượt đèn đỏ", "không chấp hành đèn tín hiệu",
          "vi phạm tín hiệu đèn giao thông"]),
        (("tốc độ", "quá tốc độ", "bắn tốc độ", "chạy quá tốc độ"),
         ["chạy quá tốc độ", "vượt tốc độ", "lỗi tốc độ"]),
        (("nồng độ cồn", "rượu", "bia", "cồn",
          "đồ uống có cồn", "thổi nồng độ cồn"),
         ["uống rượu bia", "say rượu",
          "vi phạm nồng độ cồn", "nồng độ cồn"]),
        (("mũ bảo hiểm", "không đội mũ", "đội mũ bảo hiểm"),
         ["không đội mũ bảo hiểm", "đội mũ bảo hiểm",
          "lỗi mũ bảo hiểm"]),
        (("làn đường", "phần đường", "lấn làn", "sai làn", "đè vạch"),
         ["đi sai làn", "không đúng phần đường",
          "đè vạch kẻ đường", "lấn làn"]),
        (("quá tải", "tải trọng", "quá khổ", "chở hàng"),
         ["chở quá tải", "vượt tải trọng", "xe quá khổ"]),
        (("giấy phép lái xe", "bằng lái", "gplx",
          "không có bằng", "giấy phép, chứng chỉ hành nghề"),
         ["GPLX", "bằng lái xe", "không mang bằng lái",
          "không có giấy phép lái xe", "thu hồi bằng lái"]),
        (("tạm giữ", "lập biên bản", "tuần tra", "kiểm soát"),
         ["xử lý vi phạm", "thủ tục xử phạt",
          "tạm giữ phương tiện", "tạm giữ xe",
          "giữ xe", "giữ bằng"]),
        (("khắc phục hậu quả", "buộc"),
         ["biện pháp khắc phục hậu quả"]),
        (("tái phạm",),
         ["vi phạm lại", "tái phạm"]),
        (("vi phạm nhiều lần",),
         ["nhiều lần vi phạm", "vi phạm nhiều lần"]),
        (("biển số", "biển số xe", "gắn biển số"),
         ["che biển số", "dán biển số", "sửa biển số",
          "decal biển số", "thay đổi biển số"]),
        (("thiết bị kỹ thuật nghiệp vụ", "hình ảnh",
          "camera", "ghi hình", "thông báo vi phạm"),
         ["phạt nguội", "hình ảnh vi phạm",
          "bắn tốc độ", "phạt qua camera"]),
        (("tịch thu",),
         ["tịch thu phương tiện", "tịch thu xe",
          "tịch thu xe máy", "tịch thu ô tô"]),
    ]

    for terms, values in checks:
        if any(term in lower for term in terms):
            aliases.extend(values)

    return unique(aliases)


def detect_chunk_role(text: str, article_title: str) -> str:
    """Classify the semantic role of a chunk (penalty, procedural, …)."""
    combined = f"{article_title} {text}".lower()
    title = article_title.lower()

    if any(t in title for t in [
        "nguyên tắc", "giải thích từ ngữ",
        "phạm vi", "đối tượng áp dụng",
    ]):
        return "principle"
    if any(t in title for t in ["thẩm quyền", "trách nhiệm"]):
        return "procedural"
    if any(t in combined for t in [
        "phạt tiền", "trừ điểm", "tước quyền",
        "đình chỉ", "tịch thu", "cảnh cáo",
    ]):
        return "actionable_penalty"
    if any(t in combined for t in [
        "thẩm quyền", "trách nhiệm", "thủ tục",
        "lập biên bản", "tạm giữ", "tuần tra", "kiểm soát",
    ]):
        return "procedural"
    if any(t in combined for t in [
        "tái phạm", "tình tiết tăng nặng",
        "tình tiết giảm nhẹ", "vi phạm nhiều lần",
    ]):
        return "factor"

    return "general"


# ============================================================================
# Hierarchical context container
# ============================================================================

class Context(dict):
    """Dictionary subclass exposing legal hierarchy fields as properties."""

    @property
    def part(self) -> str:
        return self.get("part", "")

    @property
    def part_title(self) -> str:
        return self.get("part_title", "")

    @property
    def chapter(self) -> str:
        return self.get("chapter", "")

    @property
    def chapter_title(self) -> str:
        return self.get("chapter_title", "")

    @property
    def section(self) -> str:
        return self.get("section", "")

    @property
    def section_title(self) -> str:
        return self.get("section_title", "")

    @property
    def article(self) -> str:
        return self.get("article", "")

    @property
    def article_title(self) -> str:
        return self.get("article_title", "")

    @property
    def clause(self) -> str:
        return self.get("clause", "")

    @property
    def clause_text(self) -> str:
        return self.get("clause_text", "")


def make_context() -> Context:
    """Return a fresh, empty legal-hierarchy context."""
    return Context(
        part="", part_title="",
        chapter="", chapter_title="",
        section="", section_title="",
        article="", article_title="",
        clause="", clause_text="",
    )


# ============================================================================
# Chunk construction
# ============================================================================

def chunk_location_label(ctx: Context, point_label: str = "") -> str:
    """Build a human-readable location string (e.g. Điều 5 – Khoản 2)."""
    parts: list[str] = []
    if ctx.article:
        parts.append(f"Điều {ctx.article}: {ctx.article_title}".strip())
    if ctx.clause:
        clause = f"Khoản {ctx.clause}"
        if point_label:
            clause += f" Điểm {point_label}"
        parts.append(clause)
    return " - ".join(parts)


def embed_header(
    document_type: str, document_id: str,
    ctx: Context, point_label: str = "",
) -> str:
    """Build the bracketed header prepended to ``embed_text``."""
    parts: list[str] = []
    if document_type and document_id:
        parts.append(f"{document_type} {document_id}")
    if ctx.article:
        parts.append(f"Điều {ctx.article}: {ctx.article_title}".strip())
    if ctx.clause:
        clause = f"Khoản {ctx.clause}"
        if point_label:
            clause += f" Điểm {point_label.lower()}"
        parts.append(clause)
    return " > ".join(part for part in parts if part)


def build_chunk(
    *,
    document_code: str,
    document_id: str,
    document_type: str,
    document_name: str,
    ctx: Context,
    content: str,
    point_label: str = "",
    id_suffix: str = "",
) -> dict[str, Any]:
    """Assemble a single chunk dictionary with all required fields."""
    content = clean_text(content)
    point_value = point_label.lower() if point_label else None

    # Build chunk ID  --  e.g. 168_2024_ND_D5_K2_A
    id_parts = [document_code]
    if ctx.article:
        id_parts.append(f"D{ctx.article}")
    if ctx.clause:
        id_parts.append(f"K{ctx.clause}")
    if id_suffix:
        id_parts.append(id_suffix.upper())
    elif point_label:
        id_parts.append(point_label.upper())

    location = chunk_location_label(ctx, point_label.upper())
    contextualized_content = (
        f"{document_name} ({document_id}) - {location}:\n"
        f"Nội dung: {content}"
    ).strip()

    # Detect synonym keywords from chunk content only (not article title)
    # to avoid noisy cross-contamination.
    aliases = detect_aliases(content)
    alias_suffix = f" (Từ khóa: {', '.join(aliases)})" if aliases else ""
    embed_text = (
        f"[{embed_header(document_type, document_id, ctx, point_label)}] "
        f"{content}{alias_suffix}"
    ).strip()

    parent_id = None
    if ctx.article:
        if ctx.clause:
            if point_label:
                parent_id = f"{document_code}_D{ctx.article}_K{ctx.clause}"
            else:
                parent_id = f"{document_code}_D{ctx.article}"

    metadata = {
        "document_id": document_id,
        "document_type": document_type,
        "status": "active",
        "article": parse_int(ctx.article),
        "clause": parse_int(ctx.clause),
        "point": point_value,
        "parent_id": parent_id,
        "heading_title": ctx.article_title,
        "chunk_role": detect_chunk_role(content, ctx.article_title),
        "retrieval_weight": 0.5,
        "aliases": aliases,
        "token_count": estimate_tokens(contextualized_content),
    }

    return {
        "id": "_".join(part for part in id_parts if part),
        "content": content,
        "contextualized_content": contextualized_content,
        "embed_text": embed_text,
        "metadata": metadata,
    }


def point_range_label(points: list[tuple[str, str]]) -> str:
    """Return a compact label like ``a`` or ``a-d`` for a group of points."""
    if not points:
        return ""
    if len(points) == 1:
        return points[0][0]
    return f"{points[0][0]}-{points[-1][0]}"


def content_for_point_group(
    clause_intro: str, points: list[tuple[str, str]],
) -> str:
    """Combine clause introduction text with one or more sub-points."""
    lines: list[str] = []
    if clause_intro:
        lines.append(clause_intro)
    if len(points) == 1:
        lines.append(points[0][1])
    else:
        for key, text in points:
            lines.append(f"Điểm {key}) {text}" if text else f"Điểm {key})")
    return "\n".join(line for line in lines if line).strip()


def split_clause_without_points(
    base_chunk_args: dict[str, Any], text: str,
) -> list[dict[str, Any]]:
    """Split a clause that has no sub-points into token-limited chunks."""
    first = build_chunk(**base_chunk_args, content=text)
    if first["metadata"]["token_count"] <= MAX_TOKENS_PER_CHUNK:
        return [first]

    # Leave room for contextual header overhead.
    content_limit = max(80, MAX_TOKENS_PER_CHUNK - 90)
    parts = split_long_text(text, content_limit)
    chunks = [
        build_chunk(**base_chunk_args, content=part, id_suffix=f"P{idx}")
        for idx, part in enumerate(parts, start=1)
    ]

    # Retry with tighter limit if header overhead still pushes over.
    if any(c["metadata"]["token_count"] > MAX_TOKENS_PER_CHUNK for c in chunks):
        parts = split_long_text(text, max(50, content_limit - 80))
        chunks = [
            build_chunk(**base_chunk_args, content=part, id_suffix=f"P{idx}")
            for idx, part in enumerate(parts, start=1)
        ]

    if len(chunks) > 1:
        for idx, chunk in enumerate(chunks, start=1):
            chunk["metadata"]["part_index"] = idx
            chunk["metadata"]["part_total"] = len(chunks)

    return chunks


def split_clause_by_points(
    base_chunk_args: dict[str, Any],
    clause_intro: str,
    points: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Group sub-points into token-limited chunks."""
    chunks: list[dict[str, Any]] = []
    current: list[tuple[str, str]] = []

    def candidate(group: list[tuple[str, str]]) -> dict[str, Any]:
        label = point_range_label(group)
        content = content_for_point_group(clause_intro, group)
        return build_chunk(**base_chunk_args, content=content, point_label=label)

    for point in points:
        test_group = current + [point]
        test_chunk = candidate(test_group)
        if current and test_chunk["metadata"]["token_count"] > MAX_TOKENS_PER_CHUNK:
            chunks.append(candidate(current))
            current = [point]
            continue
        current = test_group

    if current:
        chunks.append(candidate(current))

    if len(chunks) > 1:
        for idx, chunk in enumerate(chunks, start=1):
            chunk["metadata"]["part_index"] = idx
            chunk["metadata"]["part_total"] = len(chunks)

    return chunks


# ============================================================================
# Main parser
# ============================================================================

def parse_text(text: str, doc_name: str = "") -> list[dict[str, Any]]:
    """Parse a full legal document into a list of semantic chunks."""
    lines = [clean_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    document_code, document_id, document_type = normalize_document_metadata(doc_name)
    document_name = extract_document_name(lines, document_type, document_id)
    ctx = make_context()

    chunks: list[dict[str, Any]] = []
    article_intro: list[str] = []
    current_point_key: Optional[str] = None
    current_point_lines: list[str] = []
    clause_points: list[tuple[str, str]] = []
    pending_title: Optional[str] = None

    def base_args() -> dict[str, Any]:
        return {
            "document_code": document_code,
            "document_id": document_id,
            "document_type": document_type,
            "document_name": document_name,
            "ctx": Context(ctx),
        }

    def flush_point() -> None:
        nonlocal current_point_key, current_point_lines
        if current_point_key:
            clause_points.append(
                (current_point_key, clean_text(" ".join(current_point_lines)))
            )
        current_point_key = None
        current_point_lines = []

    def flush_article_intro() -> None:
        nonlocal article_intro
        if not article_intro or not ctx.article:
            article_intro = []
            return
        intro = clean_text("\n".join(article_intro))
        saved_clause = ctx["clause"]
        saved_clause_text = ctx["clause_text"]
        ctx["clause"] = ""
        ctx["clause_text"] = ""
        chunks.extend(split_clause_without_points(base_args(), intro))
        ctx["clause"] = saved_clause
        ctx["clause_text"] = saved_clause_text
        article_intro = []

    def flush_clause() -> None:
        nonlocal clause_points
        flush_point()
        if not ctx.clause:
            return

        clause_intro = clean_text(ctx.clause_text)
        if clause_points:
            chunks.extend(
                split_clause_by_points(base_args(), clause_intro, clause_points)
            )
        elif clause_intro:
            chunks.extend(split_clause_without_points(base_args(), clause_intro))

        ctx["clause"] = ""
        ctx["clause_text"] = ""
        clause_points = []

    def flush_article() -> None:
        flush_clause()
        flush_article_intro()

    # ---- Line-by-line parsing ----
    for raw_line in lines:
        line = clean_text(raw_line)

        # Phần (Part)
        part_match = PART_RE.match(line)
        if part_match:
            flush_article()
            ctx["part"] = part_match.group(1).strip()
            ctx["part_title"] = ""
            ctx["chapter"] = ctx["chapter_title"] = ""
            ctx["section"] = ctx["section_title"] = ""
            ctx["article"] = ctx["article_title"] = ""
            pending_title = "part"
            continue

        # Chương (Chapter)
        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            flush_article()
            ctx["chapter"] = chapter_match.group(1).strip()
            ctx["chapter_title"] = ""
            ctx["section"] = ctx["section_title"] = ""
            ctx["article"] = ctx["article_title"] = ""
            pending_title = "chapter"
            continue

        # Mục (Section)
        section_match = SECTION_RE.match(line)
        if section_match:
            flush_article()
            ctx["section"] = section_match.group(1).strip()
            ctx["section_title"] = section_match.group(2).strip()
            ctx["article"] = ctx["article_title"] = ""
            pending_title = None if ctx.section_title else "section"
            continue

        # Điều (Article)
        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush_article()
            ctx["article"] = article_match.group(1).strip()
            ctx["article_title"] = article_match.group(2).strip()
            ctx["clause"] = ""
            ctx["clause_text"] = ""
            article_intro = []
            pending_title = None if ctx.article_title else "article"
            continue

        # Pending title lines (next line after a bare heading)
        if pending_title == "part":
            ctx["part_title"] = line
            pending_title = None
            continue
        if pending_title == "chapter":
            ctx["chapter_title"] = line
            pending_title = None
            continue
        if pending_title == "section":
            ctx["section_title"] = line
            pending_title = None
            continue
        if pending_title == "article":
            ctx["article_title"] = line
            pending_title = None
            continue

        # Khoản (Clause)
        clause_match = CLAUSE_RE.match(line)
        if clause_match and ctx.article and "|" not in line[:40]:
            flush_clause()
            flush_article_intro()
            ctx["clause"] = clause_match.group(1).strip()
            ctx["clause_text"] = clause_match.group(2).strip()
            pending_title = None
            continue

        # Điểm (Point / sub-clause)
        point_match = POINT_RE.match(line)
        if point_match and ctx.clause:
            flush_point()
            current_point_key = point_match.group(1).strip()
            current_point_lines = [point_match.group(2).strip()]
            pending_title = None
            continue

        # Continuation lines
        if current_point_key:
            current_point_lines.append(line)
        elif ctx.clause:
            ctx["clause_text"] = normalize_spaces(f"{ctx.clause_text} {line}")
        elif ctx.article:
            article_intro.append(line)

    flush_article()
    return chunks


# ============================================================================
# File I/O
# ============================================================================

def read_docx_text(path: str) -> str:
    """Extract plain text from a .docx file (mammoth → python-docx fallback)."""
    mammoth_err = None
    try:
        import mammoth
        with open(path, "rb") as file:
            result = mammoth.extract_text(file)
        if result.value.strip():
            return result.value
    except Exception as exc:
        mammoth_err = exc

    docx_err = None
    try:
        from docx import Document
        document = Document(path)
        lines: list[str] = []
        for paragraph in document.paragraphs:
            text = clean_text(paragraph.text)
            if text:
                lines.append(text)
        for table in document.tables:
            for row in table.rows:
                cells = [clean_text(cell.text) for cell in row.cells]
                cells = [cell for cell in cells if cell]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)
    except Exception as exc:
        docx_err = exc

    filename = os.path.basename(path)
    print(f"  Warning: Failed to extract text from docx file '{filename}'")
    if isinstance(mammoth_err, ImportError) and isinstance(docx_err, ImportError):
        print(
            "    Reason: Both 'mammoth' and 'python-docx' libraries are "
            "missing. Install them: pip install mammoth python-docx"
        )
    else:
        if mammoth_err:
            print(f"    mammoth extraction error: {mammoth_err}")
        if docx_err:
            print(f"    python-docx extraction error: {docx_err}")
    return ""


def read_text_file(path: str) -> str:
    """Read a document file (.txt, .docx, .doc) and return its text."""
    ext = os.path.splitext(path)[1].lower()
    filename = os.path.basename(path)

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()

    if ext == ".docx":
        return read_docx_text(path)

    if ext == ".doc":
        try:
            import textract
            raw = textract.process(path)
            return raw.decode("utf-8", errors="ignore")
        except ImportError:
            print(f"  Warning: Failed to read doc file '{filename}'")
            print(
                "    Reason: 'textract' library is missing. "
                "Install it: pip install textract"
            )
            return ""
        except Exception as exc:
            print(f"  Warning: Failed to read doc file '{filename}': {exc}")
            return ""

    raise ValueError(f"Unsupported file format: {ext}")


def discover_documents(input_folder: str) -> list[str]:
    """Find all legal document files in *input_folder*, preferring .docx."""
    paths: list[str] = []
    for pattern in ("*.txt", "*.docx", "*.doc"):
        paths.extend(glob.glob(os.path.join(input_folder, pattern)))

    preferred: dict[str, str] = {}
    priority = {".docx": 3, ".txt": 2, ".doc": 1}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1].lower()
        current = preferred.get(stem)
        if current is None or priority.get(ext, 0) > priority.get(
            os.path.splitext(current)[1].lower(), 0
        ):
            preferred[stem] = path
    return sorted(preferred.values())


def process_documents(input_folder: str, output_file: str) -> int:
    """Read all documents, chunk them, and write the JSON output."""
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return 1

    documents = discover_documents(input_folder)
    if not documents:
        print(f"Error: No documents found in {input_folder}")
        return 1

    print(f"Found {len(documents)} document(s) to process:")
    for path in documents:
        print(f"  - {os.path.basename(path)}")

    all_chunks: list[dict[str, Any]] = []
    for path in documents:
        name = os.path.basename(path)
        print(f"\nProcessing: {name}")
        text = read_text_file(path)
        if not text.strip():
            print("  Warning: could not read text")
            continue
        chunks = parse_text(text, doc_name=name)
        print(f"  Generated {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("Error: No chunks generated")
        return 1

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=2)

    over_limit = sum(
        1
        for chunk in all_chunks
        if chunk["metadata"]["token_count"] > MAX_TOKENS_PER_CHUNK
    )
    print(f"\nSuccess! Generated {len(all_chunks)} chunks")
    print(
        f"Chunks over {MAX_TOKENS_PER_CHUNK} tokens: {over_limit} "
        "(single long point chunks are allowed)"
    )
    print(f"Output saved to: {output_file}")
    return 0


# ============================================================================
# CLI entry point
# ============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments and run the chunking pipeline."""
    parser = argparse.ArgumentParser(
        description="Chunk Vietnamese legal documents by clause.",
    )
    parser.add_argument("--input", default=INPUT_FOLDER)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args(argv)
    return process_documents(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
