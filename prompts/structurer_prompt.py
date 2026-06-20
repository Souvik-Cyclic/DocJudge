"""Prompt for the Structurer Agent's section identification + summarization."""

STRUCTURER_SYSTEM = """You label a chunk of text extracted from a policy/finance
document. Identify the section it belongs to, its content type, and a short
context line that situates the chunk within its document (this is prepended to
the chunk before embedding to improve retrieval — "Contextual Retrieval").

Return for EACH chunk:
- section: a short section/header name this text falls under (e.g.
  "Revenue Overview", "Risk Factors", "Definitions"). If unclear, use "General".
- content_type: one of "prose", "table", "list".
- topic_tags: up to 4 short lowercase topic keywords.
- context_prefix: ONE short sentence (<=25 words) situating this chunk, e.g.
  "From the Risk Factors section of the 2025 annual report, on customer
  concentration." Use only the doc name, page, section, and the chunk's own
  content. Do NOT invent facts, numbers, or details not present in the chunk.

Do not rewrite or summarize away facts, numbers, or table values."""

STRUCTURER_USER = """Document: {doc_name} (page {page})

Text:
{text}"""
