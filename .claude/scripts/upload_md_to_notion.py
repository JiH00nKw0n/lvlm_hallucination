#!/usr/bin/env python3
"""Upload a markdown file to a Notion page with full formatting.

Usage:
    python upload_md_to_notion.py <input.md> <page_id_or_url>

Supports: headings, tables, inline equations ($...$), block equations ($$...$$),
bold (**...**), bullet/numbered lists, dividers (---), callouts (> ...).

Requires: pip install notion-client
Token: reads from ~/.mcp.json (mcpServers.notion.env.OPENAPI_MCP_HEADERS)
"""

import argparse
import json
import os
import re
import time

from notion_client import Client


# ------------------------------------------------------------------ #
# Token                                                               #
# ------------------------------------------------------------------ #

def _get_token() -> str:
    """Extract Notion token from ~/.mcp.json."""
    mcp_path = os.path.expanduser("~/.mcp.json")
    with open(mcp_path) as f:
        config = json.load(f)

    # Try multiple server names
    for name in ("notion-wecoverai", "notion"):
        srv = config.get("mcpServers", {}).get(name, {})
        headers_str = srv.get("env", {}).get("OPENAPI_MCP_HEADERS", "")
        if headers_str:
            headers = json.loads(headers_str)
            return headers["Authorization"].split("Bearer ")[1]

    raise RuntimeError("Notion token not found in ~/.mcp.json")


# ------------------------------------------------------------------ #
# Markdown -> Notion blocks                                          #
# ------------------------------------------------------------------ #

def _split_equations(text: str, bold: bool = False) -> list[dict]:
    """Split text on $...$ into text and equation rich_text parts."""
    parts: list[dict] = []
    segments = re.split(r'\$([^$]+)\$', text)
    for i, seg in enumerate(segments):
        if i % 2 == 0:
            if seg:
                rt: dict = {"type": "text", "text": {"content": seg}}
                if bold:
                    rt["annotations"] = {"bold": True}
                parts.append(rt)
        else:
            parts.append({"type": "equation", "equation": {"expression": seg}})
    return parts


def parse_rich_text(text: str) -> list[dict]:
    """Convert markdown text with **...** and $...$ to Notion rich_text.

    Bold is processed first so that **$\\alpha$ definition**: works correctly.
    Then equations are split within each bold/non-bold segment.
    """
    parts: list[dict] = []
    # Split on **...** first (bold may contain $...$)
    bold_segments = re.split(r'\*\*(.+?)\*\*', text)
    for i, seg in enumerate(bold_segments):
        if i % 2 == 1:
            # Bold segment — split equations inside it
            parts.extend(_split_equations(seg, bold=True))
        else:
            # Non-bold segment — split equations
            parts.extend(_split_equations(seg, bold=False))
    return [p for p in parts if p.get("text", {}).get("content", "") or p.get("equation")]


def md_to_blocks(md_text: str) -> list[dict]:
    """Parse markdown into Notion block objects."""
    blocks: list[dict] = []
    lines = md_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith('### '):
            blocks.append({"type": "heading_3", "heading_3": {"rich_text": parse_rich_text(stripped[4:])}})
            i += 1; continue
        if stripped.startswith('## '):
            blocks.append({"type": "heading_2", "heading_2": {"rich_text": parse_rich_text(stripped[3:])}})
            i += 1; continue
        if stripped.startswith('# '):
            blocks.append({"type": "heading_1", "heading_1": {"rich_text": parse_rich_text(stripped[2:])}})
            i += 1; continue
        if stripped == '---':
            blocks.append({"type": "divider", "divider": {}})
            i += 1; continue
        if stripped.startswith('$$'):
            expr = stripped[2:]
            if expr.endswith('$$'):
                expr = expr[:-2]
            else:
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('$$'):
                    expr += ' ' + lines[i].strip()
                    i += 1
                if i < len(lines):
                    expr += ' ' + lines[i].strip().rstrip('$').rstrip('$')
            blocks.append({"type": "paragraph", "paragraph": {"rich_text": [{"type": "equation", "equation": {"expression": expr.strip()}}]}})
            i += 1; continue
        img_match = re.match(r'^!\[([^\]]*)\]\((\S+)\)$', stripped)
        if img_match:
            alt, url = img_match.group(1), img_match.group(2)
            blocks.append({"type": "image", "image": {"type": "external", "external": {"url": url}, "caption": parse_rich_text(alt) if alt else []}})
            i += 1; continue
        if stripped.startswith('> '):
            blocks.append({"type": "callout", "callout": {"rich_text": parse_rich_text(stripped[2:]), "icon": {"type": "emoji", "emoji": "\u25b6\ufe0f"}}})
            i += 1; continue
        if stripped.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row_text = lines[i].strip()
                if re.match(r'^\|[\s\-:|]+\|$', row_text):
                    i += 1; continue
                cells_text = [c.strip() for c in row_text.split('|')[1:-1]]
                rows.append({"type": "table_row", "table_row": {"cells": [parse_rich_text(c) for c in cells_text]}})
                i += 1
            if rows:
                blocks.append({"type": "table", "table": {"table_width": len(rows[0]["table_row"]["cells"]), "has_column_header": True, "children": rows}})
            continue
        if stripped.startswith('- '):
            blocks.append({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": parse_rich_text(stripped[2:])}})
            i += 1; continue
        m = re.match(r'^(\d+)\.\s+', stripped)
        if m:
            blocks.append({"type": "numbered_list_item", "numbered_list_item": {"rich_text": parse_rich_text(stripped[len(m.group(0)):])}})
            i += 1; continue
        blocks.append({"type": "paragraph", "paragraph": {"rich_text": parse_rich_text(stripped)}})
        i += 1
    return blocks


# ------------------------------------------------------------------ #
# Upload                                                             #
# ------------------------------------------------------------------ #

def extract_page_id(page_id_or_url: str) -> str:
    """Extract page ID from URL or raw ID."""
    # URL format: https://www.notion.so/workspace/Page-Title-<32hex>
    m = re.search(r'([0-9a-f]{32})\s*$', page_id_or_url.replace('-', ''))
    if m:
        raw = m.group(1)
        return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"
    # Already formatted UUID
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-', page_id_or_url):
        return page_id_or_url
    raise ValueError(f"Cannot parse page ID from: {page_id_or_url}")


def upload(md_path: str, page_id_or_url: str, batch_size: int = 10):
    token = _get_token()
    client = Client(auth=token)
    page_id = extract_page_id(page_id_or_url)

    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()

    blocks = md_to_blocks(md_text)
    print(f"Parsed {len(blocks)} blocks from {md_path}")

    total = len(blocks)
    sent = 0
    for i in range(0, total, batch_size):
        batch = blocks[i:i + batch_size]
        for attempt in range(5):
            try:
                client.blocks.children.append(block_id=page_id, children=batch)
                sent += len(batch)
                print(f"  Batch {i // batch_size}: {len(batch)} blocks ({sent}/{total})")
                break
            except Exception as e:
                err = str(e)
                if "503" in err or "service_unavailable" in err:
                    wait = 3 * (attempt + 1)
                    print(f"  Batch {i // batch_size}: 503, retry in {wait}s ({attempt + 1}/5)")
                    time.sleep(wait)
                else:
                    print(f"  Batch {i // batch_size}: FAILED - {err[:200]}")
                    break
        time.sleep(0.3)

    print(f"\nDone: {sent}/{total} blocks uploaded to {page_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload markdown to Notion")
    parser.add_argument("input", help="Path to markdown file")
    parser.add_argument("page_id", help="Notion page ID or URL")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    upload(args.input, args.page_id, args.batch_size)


if __name__ == "__main__":
    main()