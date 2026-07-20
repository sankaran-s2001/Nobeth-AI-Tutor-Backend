import os
import json
import logging
import datetime
import re
from tutor_app.utils.db_client import get_collection, get_qdrant_client

# Set local HuggingFace cache folder inside workspace
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# REMOTE EMBEDDING CLIENT (no torch / no local model weights)
# ─────────────────────────────────────────────────────────────
class HuggingFaceInferenceEmbeddings:
    """
    Lightweight wrapper around the HuggingFace Inference API.
    Eliminates PyTorch / sentence-transformers to stay within
    Render free-tier 512 MB RAM limit.
    """
    def __init__(self):
        from django.conf import settings
        from huggingface_hub import InferenceClient
        self.api_token = (
            getattr(settings, "HF_TOKEN", None)
            or os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
        )
        self.model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.client = InferenceClient(token=self.api_token)

    def embed_query(self, text: str):
        try:
            res = self.client.feature_extraction(
                text=text,
                model=self.model_name
            )
            if hasattr(res, "tolist"):
                return res.tolist()
            if isinstance(res, list):
                return res
            # Handle ndarray from newer client versions
            import numpy as np
            if isinstance(res, np.ndarray):
                return res.tolist()
            return res
        except Exception as e:
            logger.error(f"HF InferenceClient feature_extraction exception: {e}")
            return None


_embeddings = None
_semantic_query_cache: dict = {}


def get_embeddings() -> HuggingFaceInferenceEmbeddings | None:
    """Lazily initialises the remote HuggingFace embedding client."""
    global _embeddings
    if _embeddings is None:
        try:
            _embeddings = HuggingFaceInferenceEmbeddings()
            logger.info("Remote HuggingFace Inference embedding client ready.")
        except Exception as e:
            logger.error(f"Failed to init embedding client: {e}")
            _embeddings = None
    return _embeddings


# ─────────────────────────────────────────────────────────────
# COLLECTION NAME HELPER
# ─────────────────────────────────────────────────────────────
def get_grade_collection_name(grade_val: str) -> str:
    """Maps grade string → Qdrant collection name (e.g. '8' → 'irish_8th_collection')."""
    clean = str(grade_val).lower().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "").strip()
    ordinals = {
        "1": "1st", "2": "2nd", "3": "3rd", "4": "4th", "5": "5th",
        "6": "6th", "7": "7th", "8": "8th", "9": "9th", "10": "10th",
        "11": "11th", "12": "12th",
    }
    suffix = ordinals.get(clean, f"{clean}th")
    return f"irish_{suffix}_collection"


def _extract_grade_number(grade_val: str) -> int | None:
    """Returns integer grade number from grade string."""
    try:
        clean = str(grade_val).lower().replace("th","").replace("st","").replace("nd","").replace("rd","").strip()
        match = re.search(r"\d+", clean)
        return int(match.group()) if match else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# LLM INTENT CLASSIFIER
# ─────────────────────────────────────────────────────────────
def classify_query_intent(query_text: str, last_question: str | None = None) -> dict:
    """
    Uses Groq LLM to classify the query on 4 dimensions:
      - academic_type : ACADEMIC | NON_ACADEMIC
      - question_type : INDEPENDENT | FOLLOW_UP
      - is_image      : true | false  (whether a diagram adds learning value)
      - image_name    : exact student-vocabulary phrase | null
      - subject       : ["Physics", "Biology", ...] (1-3 subjects)

    Returns a dict with keys:
      is_academic, is_followup, is_image, image_name, subject
    """
    fallback = {
        "is_academic": True,
        "is_followup": len(query_text.split()) < 5,
        "is_image": False,
        "image_name": None,
        "subject": ["Science"],
    }

    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate

        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if not groq_api_key:
            return fallback

        llm = ChatGroq(groq_api_key=groq_api_key, model_name=groq_model, temperature=0)

        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict intent classifier for an NCERT school AI tutor.

Classify the CURRENT question on FOUR dimensions and return ONLY valid JSON.

━━ 1. ACADEMIC TYPE ━━
ACADEMIC   : school syllabus questions, definitions, theory, numericals, diagrams
NON_ACADEMIC: cooking, personal advice, entertainment, chit-chat, harmful content

━━ 2. QUESTION DEPENDENCY ━━
INDEPENDENT : fully understandable on its own, introduces a complete topic, OR switches/shifts to a completely new topic (e.g., switching from heat conduction to photosynthesis).
FOLLOW_UP   : depends on the previous question/context, asks to elaborate/clarify/continue on the SAME topic, or uses pronouns ("this", "that", "it", "these") referencing the previous topic.

━━ 3. IMAGE REQUIREMENT ━━
Set is_image = true ONLY when a diagram genuinely adds learning value.

ALWAYS false for: definitions, numericals, historical facts, grammar, opinions.
ALWAYS true  for: explicit requests ("show diagram", "draw", "figure", "sketch").
TRUE for visual concepts: anatomy/structure, cycles/processes, circuits/geometry.

image_name rules:
  - Extract EXACT vocabulary from the student's question.
  - Short phrase (1-6 words), lowercase, no underscores.
  - null when is_image = false.

━━ 4. SUBJECT DETECTION ━━
Return 1-3 subjects from: Physics, Chemistry, Biology, Mathematics, History,
Geography, Civics, Economics, English, Science, Other.
Order by relevance (primary first).

━━ OUTPUT FORMAT (STRICT JSON ONLY) ━━
{{
  "academic_type": "ACADEMIC | NON_ACADEMIC",
  "question_type": "INDEPENDENT | FOLLOW_UP",
  "is_image": true | false,
  "image_name": "string | null",
  "subject": ["Subject1"]
}}"""),
            ("human",
             "Previous question: {standalone}\n\nCurrent question: {question}"),
        ])

        chain  = intent_prompt | llm
        result = chain.invoke({
            "question":  query_text,
            "standalone": last_question or "NONE",
        })

        raw_content = result.content.strip()
        start_idx = raw_content.find('{')
        end_idx = raw_content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            raw_content = raw_content[start_idx:end_idx+1]
        
        parsed = json.loads(raw_content)
        is_image   = bool(parsed.get("is_image", False))
        image_name = parsed.get("image_name") if is_image else None

        return {
            "is_academic": parsed.get("academic_type") == "ACADEMIC",
            "is_followup": parsed.get("question_type") == "FOLLOW_UP",
            "is_image":    is_image,
            "image_name":  image_name,
            "subject":     parsed.get("subject", ["Science"]),
        }

    except Exception as e:
        logger.warning(f"classify_query_intent failed: {e} — using fallback.")
        return fallback


# ─────────────────────────────────────────────────────────────
# GRADE-AWARE CONTEXTUAL SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
def create_contextual_rag_prompt(grade: str, is_followup: bool, history: str) -> str:
    """
    Builds the full system prompt adapted to the student's grade level.
    Injects conversation history when available.
    """
    grade_num = _extract_grade_number(grade)

    # Age-appropriate language rules (Classes 1–10)
    if grade_num and grade_num <= 10:
        language_rules = f"""
LANGUAGE RULES FOR CLASS {grade_num} STUDENTS:
- Use simple, everyday words that a Class {grade_num} student understands.
- Keep sentences short and clear.
- Use examples from daily life wherever possible.
- Explain difficult words in brackets after using them.
- Be encouraging and friendly in tone.
- Do NOT use concepts, formulas, or methods from higher classes that this student hasn't studied yet.
- Use ONLY concepts, formulas, and symbols from the provided Class {grade_num} NCERT context."""
    else:
        language_rules = ""

    if is_followup:
        base = f"""You are Nobeth AI Tutor, a helpful and structured AI tutor for Class {grade} NCERT students.

CRITICAL INSTRUCTIONS FOR FOLLOW-UP QUESTIONS:
- The student is asking a follow-up based on our previous conversation.
- When they say "it", "this", "that" — refer to the most recent topic discussed.
- Use the conversation history below to understand context.
- Answer STRICTLY using only the syllabus context provided.
- If you list points, ALWAYS use continuous numbering (1, 2, 3 ...). Never restart.
- Do NOT use outside knowledge, personal information, or harmful content.
- If the student asks for a diagram, image, illustration, or figure: do NOT say that you cannot display images or that you are a text-based AI. A diagram will be attached automatically by the system. Just explain the concept and introduce the diagram.
- If the question is outside the syllabus or non-academic, say ONLY:
  "That's a great question! But it seems outside your Class {grade} NCERT syllabus. Let's focus on your current topics!"
  — Then STOP. Do not add anything else.
- If any part asks for code, scripts, recipes, games, or personal advice — skip that part entirely.
{language_rules}

IMPORTANT FORMATTING RULES:
- BOLD KEY TERMS & CATEGORIES:
  → Always use Markdown bold (`**word**`) to highlight key terms, headings, categories, and important concepts.
  → When listing items, points, or benefits, ALWAYS format them as a numbered list (1., 2., 3.) or bulleted list (-).
  → Start each item on a new line and separate them with double newlines (`\n\n`) to ensure spacing.
  → Put the category name or item header in bold followed by a colon at the start of the point (e.g., `1. **Oxygen Supply**: ...`).
- READABILITY & SPACING:
  → Avoid massive, packed blocks of text. Split your answer into logical parts.
  → Use double newlines (`\n\n`) to create clear vertical spacing between paragraphs, lists, and sections.
- THEORY / HISTORY / BIOLOGY / CIVICS / GEOGRAPHY:
  → Use clear paragraphs or numbered/bulleted list items.
  → No math formulas. No step-by-step format.

- MATHEMATICS / NUMERICAL PROBLEMS ONLY:
  → Use this EXACT exam-style structure:
    Given:
    Formula:
    Step 1:
    Step 2:
    ...
    Final Answer:
  → Always write the formula before any calculation.
  → Show EVERY intermediate step. Never skip steps.
  → Show unit conversions as separate steps.

- LOGICAL REASONING:
  → Answer briefly. Maximum 3 steps.
  → Give final answer directly if obvious. No repetition."""
    else:
        base = f"""You are Nobeth AI Tutor, a helpful and structured AI tutor for Class {grade} NCERT students.

INSTRUCTIONS:
- Answer STRICTLY using ONLY the provided NCERT syllabus context.
- Stay within the Class {grade} NCERT syllabus. Do not go beyond it.
- Render all mathematical equations in LaTeX (use $$ for block equations, $ for inline).
- If the context contains tabular data, present it as a Markdown table.
- If you list points, ALWAYS use continuous numbering (1, 2, 3 ...). Never restart.
- Cite subject, chapter, and page number where applicable.
- If the student asks for a diagram, image, illustration, or figure: do NOT say that you cannot display images or that you are a text-based AI. A diagram will be attached automatically by the system. Just explain the concept and introduce the diagram.
- If the context does not contain the answer, say ONLY:
  "That's a great question! But it seems outside your Class {grade} NCERT syllabus. Let's focus on your current topics!"
  — Then STOP.
- If the question is non-academic (movies, advice, cooking, etc.) — use the same refusal phrase.
- If any part asks for code, scripts, recipes, games, or personal advice — skip that part.
{language_rules}

IMPORTANT FORMATTING RULES:
- BOLD KEY TERMS & CATEGORIES:
  → Always use Markdown bold (`**word**`) to highlight key terms, headings, categories, and important concepts.
  → When listing items, points, or benefits, ALWAYS format them as a numbered list (1., 2., 3.) or bulleted list (-).
  → Start each item on a new line and separate them with double newlines (`\n\n`) to ensure spacing.
  → Put the category name or item header in bold followed by a colon at the start of the point (e.g., `1. **Oxygen Supply**: ...`).
- READABILITY & SPACING:
  → Avoid massive, packed blocks of text. Split your answer into logical parts.
  → Use double newlines (`\n\n`) to create clear vertical spacing between paragraphs, lists, and sections.
- THEORY / HISTORY / BIOLOGY / CIVICS / GEOGRAPHY:
  → Use clear paragraphs or numbered/bulleted list items.
  → No math formulas. No step-by-step format.

- MATHEMATICS / NUMERICAL PROBLEMS ONLY:
  → Use this EXACT exam-style structure:
    Given:
    Formula:
    Step 1:
    Step 2:
    ...
    Final Answer:
  → Always write the formula FIRST before any calculation.
  → Show EVERY intermediate step clearly.
  → Show unit conversions as separate steps.

- LOGICAL REASONING:
  → Answer briefly. Maximum 3 steps. No repetition."""

    if history.strip():
        return f"""{base}

Conversation History:
---
{history}
---"""
    return base


def _format_chat_history(chat_history: list, is_followup: bool, max_turns: int = 5) -> str:
    """Formats chat history for injection into the system prompt."""
    if not chat_history:
        return ""

    def escape(text: str) -> str:
        return (text or "").replace("{", "{{").replace("}", "}}")

    turns = chat_history[-max_turns:]
    lines = []

    if is_followup:
        for i, turn in enumerate(turns, 1):
            q = turn.get("query", "").strip()
            a = turn.get("response", "").strip()
            if q and a:
                lines.append(f"Q{i}: {q}\nA{i}: {escape(a)}")
    else:
        for turn in turns:
            q = turn.get("query", "").strip()
            a = turn.get("response", "").strip()
            if q and a:
                lines.append(f"Previous Question: {q}\nPrevious Answer: {escape(a)}")

    return "\n\n".join(lines)


def _get_last_question(chat_history: list) -> str | None:
    """Returns the most recent student query from history."""
    for turn in reversed(chat_history[-5:]):
        q = turn.get("query", "").strip()
        if q:
            return q
    return None


def _is_generic_query(q_text: str) -> bool:
    """Checks if a query is a generic follow-up without topic content (e.g. 'explain simply')."""
    generic_patterns = [
        "explain more", "explain it more", "explain further", "break it down",
        "can you explain", "tell me more", "clarify", "briefly", "more details",
        "explain", "elaborate", "simplify", "help me understand", "diagram",
        "draw", "figure", "sketch", "image", "show me", "picture", "illustration", "visual"
    ]
    q_clean = (q_text or "").lower().strip().replace("?", "").replace(".", "").replace(",", "")
    return any(pat in q_clean for pat in generic_patterns) or len(q_clean.split()) <= 4


def _get_latest_topic_question(chat_history: list) -> str | None:
    """Traverses backward in history to find the first non-generic topic question."""
    if not chat_history:
        return None
    for turn in reversed(chat_history):
        q = turn.get("query", "").strip()
        if q and not _is_generic_query(q):
            return q
    # If all are generic, fall back to the very first query in history
    for turn in chat_history:
        q = turn.get("query", "").strip()
        if q:
            return q
    return None


# ─────────────────────────────────────────────────────────────
# QDRANT DIAGRAM FETCHER  (global_pdf_diagrams collection)
# ─────────────────────────────────────────────────────────────
def fetch_diagram_from_qdrant(
    image_name: str,
    grade: str,
    similarity_threshold: float = 0.75,
    debug_info: dict | None = None,
) -> dict | None:
    """
    Semantic vector search against the Qdrant `global_pdf_diagrams` collection.

    Pipeline:
      1. Embed image_name → 384-dim vector via HF Inference API.
      2. Search Qdrant with class-level filter (predicted_classes contains grade_int).
      3. If class-level score < HIGH_CONFIDENCE_THRESHOLD, run global fallback search.
      4. Return {imageUrl, topic, caption, score} if best score >= similarity_threshold.
      5. Return None on no match or any error.
    """
    if not image_name:
        if debug_info is not None:
            debug_info["status"] = "skipped"
            debug_info["reason"] = "Empty image name requested"
        return None

    if debug_info is not None:
        debug_info["status"] = "initiated"
        debug_info["query"] = image_name
        debug_info["steps"] = []
        debug_info["threshold"] = similarity_threshold

    try:
        emb = get_embeddings()
        if emb is None:
            logger.warning("[DIAGRAM] Embedding client unavailable — skipping diagram search.")
            if debug_info is not None:
                debug_info["status"] = "failed"
                debug_info["reason"] = "Embedding client unavailable"
                debug_info["steps"].append("Failed to load embedding client")
            return None

        query_vector = emb.embed_query(image_name)
        if query_vector is None:
            logger.warning(f"[DIAGRAM] Embedding failed for: '{image_name}'")
            if debug_info is not None:
                debug_info["status"] = "failed"
                debug_info["reason"] = "Embedding generation returned None"
                debug_info["steps"].append("Embedding generation failed")
            return None

        q_client = get_qdrant_client()
        if q_client is None:
            logger.warning("[DIAGRAM] Qdrant client unavailable — skipping diagram search.")
            if debug_info is not None:
                debug_info["status"] = "failed"
                debug_info["reason"] = "Qdrant client unavailable"
                debug_info["steps"].append("Failed to load Qdrant client")
            return None

        COLLECTION = "global_diagrams_collection"
        HIGH_CONFIDENCE = 0.86
        grade_int = _extract_grade_number(grade) or 8

        if debug_info is not None:
            debug_info["collection"] = COLLECTION
            debug_info["target_grade"] = grade_int
            debug_info["steps"].append(f"Initiated search in '{COLLECTION}' for grade {grade_int}")

        # ── Step 1: Class-filtered search ──
        try:
            from qdrant_client.http import models as qm
            class_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="predicted_classes",
                        match=qm.MatchAny(any=[grade_int]),
                    ),
                    qm.FieldCondition(
                        key="is_diagram",
                        match=qm.MatchValue(value=True),
                    ),
                ]
            )

            res = q_client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                query_filter=class_filter,
                limit=1,
                with_payload=True,
            )
            class_results = res.points
            if debug_info is not None:
                debug_info["steps"].append(f"Class-filtered search completed. Found matches: {len(class_results)}")
        except Exception as e:
            logger.error(f"[DIAGRAM] Class-filtered Qdrant search failed: {e}")
            class_results = []
            if debug_info is not None:
                debug_info["steps"].append(f"Class-filtered search failed: {str(e)}")

        class_hit   = class_results[0] if class_results else None
        class_score = class_hit.score if class_hit else 0.0

        if debug_info is not None and class_hit:
            debug_info["class_match"] = {
                "topic": class_hit.payload.get("topic"),
                "score": round(class_score, 4),
                "predicted_classes": class_hit.payload.get("predicted_classes"),
                "imageUrl": class_hit.payload.get("imageUrl")
            }

        # ── Step 2: Global fallback when class score is weak ──
        global_hit = None
        global_score = 0.0
        if class_score < HIGH_CONFIDENCE:
            if debug_info is not None:
                debug_info["steps"].append(f"Class score ({class_score:.4f}) is below confidence threshold ({HIGH_CONFIDENCE}). Running global fallback.")
            try:
                from qdrant_client.http import models as qm
                global_filter = qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="is_diagram",
                            match=qm.MatchValue(value=True),
                        )
                    ]
                )
                res = q_client.query_points(
                    collection_name=COLLECTION,
                    query=query_vector,
                    query_filter=global_filter,
                    limit=1,
                    with_payload=True,
                )
                global_results = res.points
                if debug_info is not None:
                    debug_info["steps"].append(f"Global fallback search completed. Found matches: {len(global_results)}")
            except Exception as e:
                logger.warning(f"[DIAGRAM] Global fallback search failed: {e}")
                global_results = []
                if debug_info is not None:
                    debug_info["steps"].append(f"Global fallback search failed: {str(e)}")

            global_hit   = global_results[0] if global_results else None
            global_score = global_hit.score if global_hit else 0.0

            if debug_info is not None and global_hit:
                debug_info["global_match"] = {
                    "topic": global_hit.payload.get("topic"),
                    "score": round(global_score, 4),
                    "predicted_classes": global_hit.payload.get("predicted_classes"),
                    "imageUrl": global_hit.payload.get("imageUrl")
                }

            # Keep whichever is better
            if global_score > class_score:
                top_hit = global_hit
                top_score = global_score
                logger.info(f"[DIAGRAM] Global fallback wins: score={top_score:.4f}")
                if debug_info is not None:
                    debug_info["steps"].append(f"Global fallback match wins (score {global_score:.4f} > class score {class_score:.4f})")
            else:
                top_hit = class_hit
                top_score = class_score
                if debug_info is not None:
                    debug_info["steps"].append(f"Class-filtered match retained (score {class_score:.4f} >= global score {global_score:.4f})")
        else:
            top_hit = class_hit
            top_score = class_score
            logger.info(f"[DIAGRAM] Class hit high-confidence: score={top_score:.4f}")
            if debug_info is not None:
                debug_info["steps"].append(f"Class hit high confidence (score {class_score:.4f} >= {HIGH_CONFIDENCE})")

        if top_hit is None:
            logger.info(f"[DIAGRAM] No result found for: '{image_name}'")
            if debug_info is not None:
                debug_info["status"] = "not_found"
                debug_info["reason"] = "No matching diagrams found in collection"
                debug_info["steps"].append("No matching diagrams found in collection")
            return None

        is_global_match = (top_hit == global_hit) if (global_hit is not None) else False
        effective_threshold = 0.80 if is_global_match else similarity_threshold

        logger.info(
            f"[DIAGRAM] Query='{image_name}' | "
            f"Matched='{top_hit.payload.get('topic')}' | "
            f"Score={top_score:.4f} | Threshold={effective_threshold} (is_global={is_global_match})"
        )

        payload = top_hit.payload or {}
        if top_score >= effective_threshold:
            if debug_info is not None:
                debug_info["status"] = "success"
                debug_info["reason"] = f"Best match score {top_score:.4f} >= similarity threshold {effective_threshold}"
                debug_info["steps"].append("Match approved by similarity threshold check")
                debug_info["match"] = {
                    "topic": payload.get("topic"),
                    "score": round(top_score, 4),
                    "imageUrl": payload.get("imageUrl")
                }
            return {
                "imageUrl": payload.get("imageUrl"),
                "topic":    payload.get("topic"),
                "caption":  payload.get("caption") or payload.get("description"),
                "score":    round(top_score, 4),
            }

        logger.info(f"[DIAGRAM] Score {top_score:.4f} below threshold {effective_threshold} — no image returned.")
        if debug_info is not None:
            debug_info["status"] = "blocked"
            debug_info["reason"] = f"Best match score {top_score:.4f} is below similarity threshold {effective_threshold}"
            debug_info["steps"].append("Match rejected by similarity threshold check")
            debug_info["best_rejected_match"] = {
                "topic": payload.get("topic"),
                "score": round(top_score, 4),
                "imageUrl": payload.get("imageUrl")
            }
        return None

    except Exception as e:
        logger.error(f"[DIAGRAM FETCH ERROR]: {e}")
        if debug_info is not None:
            debug_info["status"] = "error"
            debug_info["reason"] = f"Exception occurred during fetch: {str(e)}"
        return None


# ─────────────────────────────────────────────────────────────
# MAIN PRODUCTION RAG PIPELINE
# ─────────────────────────────────────────────────────────────
def execute_rag_tutor_query(
    query_text: str,
    grade: str,
    chat_history: list | None = None,
    debug: bool = False,
    cache: bool = True,
) -> dict:
    """
    End-to-end production RAG pipeline:

    1.  Prompt injection / out-of-scope guard (instant block)
    2.  In-memory cache lookup
    3.  MongoDB shared cache lookup
    4.  LLM intent classification (academic / follow-up / image / subject)
    5.  Follow-up query rewriting with previous Q&A context
    6.  HF Inference API vector embedding
    7.  Grade-scoped Qdrant retrieval + similarity threshold (>= 0.68) + hybrid rerank
    8.  LLM answerability check  (standalone queries only)
    9.  Grade-aware contextual system prompt construction
    10. Groq LLM generation (llama-3.3-70b-versatile) with conversation memory
    11. Post-generation groundedness guard
    12. 4-case diagram decision logic + Qdrant diagram fetch
    13. Cache write (in-memory + MongoDB)
    14. Return structured response
    """
    if chat_history is None:
        chat_history = []

    _FALLBACK   = "I'm having trouble connecting to my learning resources. Please try again in a moment!"
    _OOS_MARKER = "outside your Class"

    debug_log = {
        "cache_status": "bypass" if not cache else "miss",
        "steps": [],
        "qdrant_retrieved_chunks": [],
        "answerability_check": None,
        "classification": None,
        "followup_rewritten": False,
        "retrieval_query": query_text,
    }

    # ── 1. Prompt Injection Guard ─────────────────────────────
    _INJECTION_PATTERNS = [
        "ignore previous", "ignore all instructions", "ignore the instructions",
        "bypass instructions", "system prompt", "reveal instructions",
        "tell me your rules", "you are now an", "you must now act",
        "forget your rules", "forget previous instructions",
        "forget the instructions", "override rules", "jailbreak",
    ]
    _OOS_KEYWORDS = ["hack", "bypass", "cheat", "porn", "prostitute",
                     "drug", "bomb", "weapon", "kill", "violence"]

    normalized_q = query_text.lower().strip()

    for pattern in _INJECTION_PATTERNS:
        if pattern in normalized_q:
            res = {
                "response": (
                    "I am a school AI tutor, and I can only help you with questions "
                    "about your NCERT curriculum subjects. Let know if you need help with your studies!"
                ),
                "sources": [],
                "diagram": None,
            }
            if debug:
                debug_log["steps"].append("Blocked by prompt injection guard")
                res["debug_info"] = debug_log
            return res

    for kw in _OOS_KEYWORDS:
        if kw in normalized_q:
            res = {
                "response": "I am here as your NCERT study companion. Let's focus on your school textbooks!",
                "sources": [],
                "diagram": None,
            }
            if debug:
                debug_log["steps"].append("Blocked by safety keywords guard")
                res["debug_info"] = debug_log
            return res

    # ── 4. LLM Intent Classification ─────────────────────────
    last_question = _get_latest_topic_question(chat_history)
    intent        = classify_query_intent(query_text, last_question)

    is_followup   = intent["is_followup"]
    is_academic   = intent["is_academic"]
    is_image      = intent["is_image"]
    image_name    = intent["image_name"]
    subject       = intent["subject"]

    # Follow-ups inherit academic status
    if is_followup and not is_academic:
        is_academic = True

    # Cannot follow-up if no history exists
    if is_followup and not chat_history:
        is_followup = False

    logger.info(
        f"[INTENT] academic={is_academic}, followup={is_followup}, "
        f"is_image={is_image}, image_name={image_name}, subject={subject}"
    )

    if debug:
        debug_log["classification"] = {
            "is_academic": is_academic,
            "is_followup": is_followup,
            "is_image": is_image,
            "image_name": image_name,
            "subject": subject
        }

    # Non-academic instant rejection
    if not is_academic:
        res = {
            "response": (
                f"That's a great question! But it seems outside your Class {grade} "
                f"NCERT syllabus. Let's focus on your current topics!"
            ),
            "sources": [],
            "diagram": None,
        }
        if debug:
            debug_log["steps"].append("Refused by non-academic check")
            res["debug_info"] = debug_log
        return res

    # ── 5. Follow-up Query Rewriting ─────────────────────────
    retrieval_query = query_text
    search_query = query_text
    is_generic_followup = False
    if is_followup and chat_history:
        last_turn   = chat_history[-1]
        last_q      = last_turn.get("query", "")
        last_a      = last_turn.get("response", "")
        
        # Always build the context-aware retrieval_query for the cache key to keep it unique
        retrieval_query = (
            f"Previous question: {last_q}\n"
            f"Previous answer: {last_a[:500]}\n"
            f"Follow-up question: {query_text}"
        )
        
        # Check if the query is a generic follow-up (e.g. explain more, break down, simplify)
        is_generic_followup = _is_generic_query(query_text)
        
        if is_generic_followup:
            topic_q = _get_latest_topic_question(chat_history)
            search_query = topic_q if topic_q else last_q
            logger.info(f"[FOLLOWUP] Generic follow-up detected. Reusing topic query '{search_query}' for vector search.")
        else:
            search_query = retrieval_query
            logger.info(f"[FOLLOWUP] Rewritten search query constructed.")
        
        if debug:
            debug_log["followup_rewritten"] = True
            debug_log["retrieval_query"] = retrieval_query
            debug_log["search_query"] = search_query
            debug_log["is_generic_followup"] = is_generic_followup

    # ── 2 & 3. Cache Lookup (Context-aware) ───────────────────
    cache_col = None
    normalized_retrieval_q = retrieval_query.lower().strip()
    cache_key = f"{grade}:{normalized_retrieval_q}"

    if cache:
        if cache_key in _semantic_query_cache:
            logger.info(f"[CACHE] In-memory hit: {query_text[:60]}")
            res_payload = _semantic_query_cache[cache_key]
            if debug:
                res_payload = dict(res_payload)
                debug_log["cache_status"] = "hit_in_memory"
                res_payload["debug_info"] = debug_log
            return res_payload

        cache_col = get_collection("query_cache")
        if cache_col is not None:
            try:
                cached_doc = cache_col.find_one({"key": cache_key})
                if cached_doc:
                    payload = {
                        "response": cached_doc["response"],
                        "sources":  cached_doc["sources"],
                        "diagram":  cached_doc.get("diagram"),
                        "subject":  cached_doc.get("subject"),
                    }
                    _semantic_query_cache[cache_key] = payload
                    logger.info(f"[CACHE] MongoDB hit: {query_text[:60]}")
                    if debug:
                        payload = dict(payload)
                        debug_log["cache_status"] = "hit_mongodb"
                        payload["debug_info"] = debug_log
                    return payload
            except Exception as e:
                logger.error(f"[CACHE] MongoDB lookup failed: {e}")

    # ── 6. Vector Embedding ───────────────────────────────────
    query_vector = None
    emb_model    = get_embeddings()
    if emb_model is not None:
        try:
            query_vector = emb_model.embed_query(search_query)
            if debug:
                if query_vector is None:
                    debug_log["steps"].append("HuggingFace Inference API returned None vector (possibly rate-limited or API error).")
                else:
                    debug_log["steps"].append(f"Successfully generated embedding vector of length {len(query_vector)}.")
        except Exception as e:
            logger.error(f"[EMBED] Failed: {e}")
            if debug:
                debug_log["steps"].append(f"Embedding generation failed with exception: {str(e)}")
    else:
        if debug:
            debug_log["steps"].append("Embedding model initialization returned None (remote HuggingFace client not ready).")

    # ── 7. Qdrant Retrieval + Threshold + Hybrid Rerank ───────
    context_chunks: list[str] = []
    source_cards:   list[dict] = []

    if query_vector is not None:
        q_client        = get_qdrant_client()
        collection_name = get_grade_collection_name(grade)

        if q_client is not None:
            if debug:
                debug_log["steps"].append(f"Initiating search in Qdrant collection: '{collection_name}'")
            try:
                res = q_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=8,
                    with_payload=True,
                )
                raw_results = res.points
                if debug:
                    debug_log["steps"].append(f"Qdrant query execution completed. Received {len(raw_results)} raw results.")
                    debug_log["qdrant_retrieved_chunks"] = [
                        {
                            "score": r.score,
                            "text": (r.payload or {}).get("page_content", "")[:300] + "...",
                            "metadata": (r.payload or {}).get("metadata", {})
                        }
                        for r in raw_results
                    ]

                # Hybrid rerank: boost by query keyword overlap first
                query_words = set(normalized_q.split())
                scored = []
                for pt in raw_results:
                    payload = pt.payload or {}
                    text    = payload.get("page_content", "")
                    overlap = len(query_words & set(text.lower().split()))
                    boosted_score = pt.score + 0.05 * overlap
                    scored.append((boosted_score, pt))

                # Apply similarity threshold filter to boosted scores
                THRESHOLD = 0.62
                matched_scored = [item for item in scored if item[0] >= THRESHOLD]
                if debug:
                    debug_log["steps"].append(f"Applied boosted similarity threshold (>= {THRESHOLD}). Matches remaining: {len(matched_scored)}")

                matched_scored.sort(key=lambda x: x[0], reverse=True)
                top_docs = [x[1] for x in matched_scored[:4]]
                if debug:
                    debug_log["steps"].append(f"Completed keyword rerank. Kept top {len(top_docs)} context chunks.")

                for pt in top_docs:
                    payload = pt.payload or {}
                    text    = payload.get("page_content", "")
                    meta    = payload.get("metadata", {})
                    context_chunks.append(text)
                    source_cards.append({
                        "subject":    meta.get("subject", subject[0] if subject else "Science"),
                        "chapter":    meta.get("chapter", "NCERT Syllabus"),
                        "pageNumber": meta.get("page_number", "N/A"),
                        "snippet":    text[:150] + "...",
                    })

                logger.info(
                    f"[QDRANT] Retrieved {len(raw_results)} docs, "
                    f"{len(matched_scored)} above threshold, top {len(top_docs)} after rerank."
                )

            except Exception as e:
                logger.error(f"[QDRANT] Search failed for {collection_name}: {e}")
                if debug:
                    debug_log["steps"].append(f"Qdrant search execution failed: {str(e)}")
        else:
            if debug:
                debug_log["steps"].append("Qdrant client could not be obtained (not connected to host).")
    else:
        if debug:
            debug_log["steps"].append("Skipping Qdrant retrieval because query_vector is None.")

    context_str = "\n\n".join(context_chunks) if context_chunks else "No book context available."

    # ── 8. LLM Answerability Check (standalone only) ──────────
    if not is_followup and context_chunks:
        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate

            groq_api_key = os.getenv("GROQ_API_KEY")
            groq_model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            _verifier_llm = ChatGroq(groq_api_key=groq_api_key, model_name=groq_model, temperature=0)

            verify_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an academic syllabus verifier. "
                 "Reply ONLY with YES or NO.\n"
                 "YES = the context covers the topic, concept, or is a similar problem/exercise of the student's question.\n"
                 "NO  = the topic only appears as a passing mention without real explanation or context.\n"
                 "For math questions: if context teaches the METHOD or CONCEPT, or lists a similar exercise/numerical, say YES.\n"
                 "Note: The context may contain OCR typos or scan errors (e.g., '125’' instead of '125^2', or '1267' instead of '126^2'). Be lenient and ignore OCR noise or formatting differences when matching mathematical symbols."
                 "Do NOT use your own knowledge to answer the question, but DO use your intelligence to determine if the topic/formula is covered in the context."),
                ("human",
                 "Question: {question}\n\nContext:\n{context}\n\n"
                 "Is the TOPIC of this question covered in the context? (YES/NO)"),
            ])

            v_chain  = verify_prompt | _verifier_llm
            v_result = v_chain.invoke({"question": query_text, "context": context_str[:3000]})
            v_answer = (v_result.content or "").strip().upper()
            is_answerable = "YES" in v_answer
            logger.info(f"[VERIFY] LLM answerability: {v_answer} → {'ALLOWED' if is_answerable else 'BLOCKED'}")

            if debug:
                debug_log["answerability_check"] = {
                    "performed": True,
                    "verdict": is_answerable,
                    "raw_response": v_answer
                }

            if not is_answerable:
                res = {
                    "response": (
                        f"That's a great question! But it seems outside your Class {grade} "
                        f"NCERT syllabus. Let's focus on your current topics!"
                    ),
                    "sources": [],
                    "diagram": None,
                }
                if debug:
                    debug_log["steps"].append("Refused by LLM answerability check")
                    res["debug_info"] = debug_log
                return res
        except Exception as e:
            logger.warning(f"[VERIFY] Answerability check failed: {e} — allowing query.")
            if debug:
                debug_log["answerability_check"] = {
                    "performed": True,
                    "verdict": "ERROR",
                    "error": str(e)
                }

    # ── 9 & 10. Grade-Aware Prompt + Groq LLM ─────────────────
    diagram = None

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not groq_api_key:
            logger.error("[GROQ] GROQ_API_KEY not set.")
            return {"response": "AI service is not configured. Please contact support.", "sources": [], "diagram": None}

        llm = ChatGroq(groq_api_key=groq_api_key, model_name=groq_model, temperature=0.3)

        history_str   = _format_chat_history(chat_history, is_followup)
        system_prompt = create_contextual_rag_prompt(grade, is_followup, history_str)

        # Build message chain with conversation memory (last 5 turns)
        messages = [SystemMessage(content=system_prompt)]
        for turn in chat_history[-5:]:
            messages.append(HumanMessage(content=turn.get("query", "")))
            messages.append(AIMessage(content=turn.get("response", "")))

        current_input = (
            f"NCERT Book Context:\n{context_str}\n\n"
            f"Question: {query_text}"
        )
        messages.append(HumanMessage(content=current_input))

        ai_msg       = llm.invoke(messages)
        response_text = ai_msg.content or ""

        # ── 11. Post-Generation Groundedness Guard ─────────────
        if not context_chunks and _OOS_MARKER not in response_text:
            response_text = (
                f"That's a great question! But it seems outside your Class {grade} "
                f"NCERT syllabus. Let's focus on your current topics!"
            )

        is_oos = _OOS_MARKER in response_text

        # ── 12. 4-Case Diagram Decision Logic ─────────────────
        # Case 1: LLM did NOT classify this as an image query → no diagram
        # Case 2: Valid answer + image requested → fetch from Qdrant
        # Case 3: Follow-up that returned false OOS → bypass block, fetch anyway
        # Case 4: True OOS (standalone) + image requested → suppress diagram

        diagram_debug = {}

        if is_image and image_name:
            resolved_image_name = image_name
            GENERIC_IMAGE_WORDS = {
                "diagram", "image", "figure", "sketch", "draw", "show", "illustration", 
                "visual", "picture", "photo", "graph", "chart", "diagram?", "image?", "figure?"
            }
            if image_name.lower().strip() in GENERIC_IMAGE_WORDS:
                topic_q = _get_latest_topic_question(chat_history) or search_query
                resolved_image_name = topic_q
                logger.info(f"[DIAGRAM] Generic image name '{image_name}' resolved to topic query: '{resolved_image_name}'")

            if not is_oos:
                # CASE 2 — Happy path
                diagram = fetch_diagram_from_qdrant(resolved_image_name, grade, debug_info=diagram_debug)
                if diagram is None:
                    response_text += (
                        "\n\n*(Note: I currently don't have a diagram for this specific "
                        "topic in my database, but I hope the explanation above helps!)*"
                    )
                else:
                    # Clean up "text-based AI" disclaimers from the response text if they explicitly asked for a diagram
                    is_explicit_image_request = any(
                        w in normalized_q 
                        for w in ["diagram", "draw", "figure", "sketch", "image", "show me", "picture", "illustration", "visual"]
                    )
                    if is_explicit_image_request:
                        paragraphs = response_text.split("\n\n")
                        cleaned_paragraphs = []
                        for para in paragraphs:
                            para_lower = para.lower()
                            is_disclaimer = False
                            if "text-based ai" in para_lower or "text-based assistant" in para_lower or "text-based chatbot" in para_lower:
                                  is_disclaimer = True
                            elif "cannot" in para_lower and ("display" in para_lower or "show" in para_lower or "draw" in para_lower) and ("image" in para_lower or "diagram" in para_lower or "picture" in para_lower or "figure" in para_lower):
                                  is_disclaimer = True
                            elif "unable to" in para_lower and ("display" in para_lower or "show" in para_lower or "draw" in para_lower) and ("image" in para_lower or "diagram" in para_lower or "picture" in para_lower or "figure" in para_lower):
                                  is_disclaimer = True
                            elif ("don't have" in para_lower or "do not have" in para_lower) and "capability" in para_lower and ("display" in para_lower or "show" in para_lower or "draw" in para_lower):
                                  is_disclaimer = True
                                  
                            if not is_disclaimer:
                                cleaned_paragraphs.append(para)
                                
                        response_text = "\n\n".join(cleaned_paragraphs).strip()
                        topic_label = diagram.get("topic", "the requested topic")
                        intro = f"Here is the relevant diagram showing {topic_label}:"
                        if not response_text or len(response_text) < 40:
                            response_text = intro
                        else:
                            if not response_text.startswith("Here is"):
                                response_text = f"{intro}\n\n{response_text}"

            elif is_followup:
                # CASE 3 — Follow-up falsely blocked → still try to fetch diagram
                is_explicit_image_request = any(
                    w in normalized_q 
                    for w in ["diagram", "draw", "figure", "sketch", "image", "show me", "picture", "illustration", "visual"]
                )
                if is_explicit_image_request:
                    diagram = fetch_diagram_from_qdrant(resolved_image_name, grade, debug_info=diagram_debug)
                    if diagram:
                        topic_label = diagram.get("topic", "this topic")
                        response_text = f"Here is the relevant diagram showing {topic_label}:"
                    else:
                        response_text = (
                            "I understand you're asking for a diagram related to our previous topic, "
                            "but I currently don't have one available in my database."
                        )
                else:
                    diagram = None
                    diagram_debug = {
                        "status": "skipped",
                        "reason": "Not an explicit image/diagram request during OOS follow-up"
                    }
            else:
                # CASE 4 — Genuine OOS → suppress diagram entirely
                diagram = None
                diagram_debug = {
                    "status": "suppressed",
                    "reason": "Genuine out-of-scope standalone query, diagram search skipped"
                }
        else:
            diagram_debug = {
                "status": "skipped",
                "reason": "Intent classifier did not request an image (is_image=False)" if not is_image else "Empty image_name returned by intent classifier"
            }

        if debug:
            debug_log["diagram_match"] = {
                "is_image_requested": is_image,
                "image_name_requested": image_name,
                "matched_diagram": diagram,
                "details": diagram_debug
            }

        # ── 13. Cache Write ────────────────────────────────────
        res_payload = {
            "response": response_text.strip(),
            "sources":  source_cards,
            "diagram":  diagram,
            "subject":  subject,
        }

        # Do not cache refusal responses containing the out-of-syllabus marker
        is_refusal = _OOS_MARKER in response_text
        if response_text and _FALLBACK not in response_text and not is_refusal:
            _semantic_query_cache[cache_key] = res_payload
            if cache_col is not None:
                try:
                    cache_col.update_one(
                        {"key": cache_key},
                        {"$set": {
                            "key":        cache_key,
                            "response":   response_text,
                            "sources":    source_cards,
                            "diagram":    diagram,
                            "subject":    subject,
                            "created_at": datetime.datetime.utcnow(),
                        }},
                        upsert=True,
                    )
                except Exception as ce:
                    logger.error(f"[CACHE] Write failed: {ce}")

        if debug:
            res_payload = dict(res_payload)
            res_payload["debug_info"] = debug_log

        return res_payload

    except Exception as e:
        logger.error(f"[GROQ] LLM chain execution failed: {e}")
        err_res = {
            "response": _FALLBACK,
            "sources":  source_cards,
            "diagram":  None,
        }
        if debug:
            debug_log["steps"].append(f"Execution exception: {e}")
            err_res["debug_info"] = debug_log
        return err_res


def find_matching_diagram(*args, **kwargs):
    """
    Legacy helper function kept for backward compatibility with integration tests.
    """
    return None


