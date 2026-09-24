"""
Answer questions about the PDFs, using retrieval_pipeline_pdf.PdfRetriever.

1. Rewrite   follow-up question -> standalone question (when there is chat history)
2. Retrieve  hybrid search over chunks.json / Chroma
3. Generate  grounded answer with [n] citations; figure / table images are sent
             to the model too when the provider supports vision (Gemini)

Usage:
    python answer_generation_pdf.py "What is scaled dot-product attention?"
    python answer_generation_pdf.py                 # interactive chat
    python answer_generation_pdf.py --provider ollama "How many heads?"
"""

import argparse
import base64
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from retrieval_pipeline_pdf import PdfRetriever, citation, format_context

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

PROVIDER = "gemini"                 # "gemini" (free tier, GEMINI_API_KEY) or "ollama" (local)
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = "llama3.1"

TOP_K = 5
NEIGHBOURS = 1                      # surrounding chunks per hit, for fuller context
SEND_IMAGES = True                  # attach figure/table images (vision models only)
MAX_IMAGES = 3
HISTORY_TURNS = 4                   # previous Q/A pairs kept for follow-ups

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions about technical documents using ONLY the numbered sources provided.

Rules:
- Use only facts stated in the sources (text, tables, equations, or attached images). Do not use outside knowledge.
- Cite every claim with its source number in square brackets, e.g. [1] or [2][3].
  Facts read from an attached image are cited with that image's source number too, never as [Figure N].
- Quote exact numbers, names and formulas as they appear in the sources.
- If sources disagree, say so and cite both.
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise: answer directly first, then add supporting detail if useful."""

REWRITE_PROMPT = """Rewrite the user's latest question as a standalone search query, resolving references
like "it", "that table" or "the second one" using the conversation. Return only the query.

Conversation:
{history}

Latest question: {question}

Standalone query:"""


# ============================================================
# LLM
# ============================================================

def get_llm(provider: str = PROVIDER):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # reads GEMINI_API_KEY / GOOGLE_API_KEY from .env
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)

    raise ValueError(f"Unknown provider: {provider}")


def supports_images(provider: str) -> bool:
    return provider == "gemini"


# ============================================================
# PROMPT BUILDING
# ============================================================

def image_block(path: str) -> Optional[Dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return None
    data = base64.b64encode(file.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def build_user_message(
    question: str,
    results: List[Dict[str, Any]],
    with_images: bool,
) -> HumanMessage:

    text = f"Sources:\n\n{format_context(results)}\n\nQuestion: {question}"

    if not with_images:
        return HumanMessage(content=text)

    blocks: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    attached: set = set()   # split tables share one image

    for n, r in enumerate(results, start=1):
        for path in r.get("image_paths") or []:
            if len(attached) >= MAX_IMAGES or path in attached:
                continue
            block = image_block(path)
            if block:
                blocks.append({"type": "text", "text": f"Image for source [{n}] ({r['type']}):"})
                blocks.append(block)
                attached.add(path)

    return HumanMessage(content=blocks)


def format_history(history: List[BaseMessage]) -> str:
    lines = []
    for m in history[-HISTORY_TURNS * 2:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content if isinstance(m.content, str) else ''}")
    return "\n".join(lines)


# ============================================================
# ANSWERER
# ============================================================

class PdfAnswerer:

    def __init__(self, provider: str = PROVIDER, retriever: Optional[PdfRetriever] = None):
        self.provider = provider
        self.llm = get_llm(provider)
        self.retriever = retriever or PdfRetriever()
        self.history: List[BaseMessage] = []   # plain Q/A text, no sources

    def standalone_question(self, question: str) -> str:
        if not self.history:
            return question
        prompt = REWRITE_PROMPT.format(history=format_history(self.history), question=question)
        rewritten = self.llm.invoke(prompt).text.strip()
        return rewritten or question

    def answer(
        self,
        question: str,
        k: int = TOP_K,
        chunk_type: Optional[str] = None,
    ) -> Dict[str, Any]:

        query = self.standalone_question(question)
        results = self.retriever.search(query, k=k, chunk_type=chunk_type, neighbours=NEIGHBOURS)

        if not results:
            answer = NO_ANSWER
        else:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                *self.history[-HISTORY_TURNS * 2:],
                build_user_message(
                    question,
                    results,
                    with_images=SEND_IMAGES and supports_images(self.provider),
                ),
            ]
            answer = self.llm.invoke(messages).text.strip()

        self.history += [HumanMessage(content=question), AIMessage(content=answer)]

        return {
            "question": question,
            "search_query": query,
            "answer": answer,
            "sources": [
                {
                    "n": n,
                    "citation": citation(r),
                    "type": r["type"],
                    "chunk_id": r["chunk_id"],
                    "image_paths": r.get("image_paths", []),
                }
                for n, r in enumerate(results, start=1)
            ],
        }

    def reset(self):
        self.history.clear()


# ============================================================
# OUTPUT
# ============================================================

def print_answer(result: Dict[str, Any]):
    if result["search_query"] != result["question"]:
        print(f"(searched: {result['search_query']})")

    print("\n" + result["answer"])

    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['type']:6} {s['citation']}")
        for path in s["image_paths"]:
            print(f"         image: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Windows consoles default to cp1252 and choke on math symbols
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Answer questions about the PDFs")
    parser.add_argument("question", nargs="?", help="question (omit for interactive chat)")
    parser.add_argument("--provider", choices=["gemini", "ollama"], default=PROVIDER)
    parser.add_argument("-k", type=int, default=TOP_K, help="chunks to retrieve")
    parser.add_argument("--type", choices=["text", "table", "figure"], help="only search this chunk type")
    args = parser.parse_args()

    answerer = PdfAnswerer(provider=args.provider)

    if args.question:
        print_answer(answerer.answer(args.question, k=args.k, chunk_type=args.type))
        return

    print("\nAsk about your PDFs. Commands: 'reset' clears history, 'exit' quits.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            answerer.reset()
            print("History cleared.")
            continue
        print_answer(answerer.answer(question, k=args.k, chunk_type=args.type))


if __name__ == "__main__":
    main()
