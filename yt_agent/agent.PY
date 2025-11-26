# AGENT4_fixed_final.py
# Final robust RAG pipeline (user-prompted video id & question), GROQ compatible, debug-friendly.
# Save this as AGENT4_fixed_final.py and run inside your venv.

import os
import inspect
from typing import List, Any
import json  # added for final printing

# --- CONFIG ---
# Prefer setting GROQ_API_KEY as environment variable. If you must, put it here.
GROQ_API_KEY_INLINE = "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"  # <-- PASTE KEY HERE ONLY IF YOU MUST (not recommended)

if os.environ.get("GROQ_API_KEY"):
    GROQ_API_KEY = os.environ["GROQ_API_KEY"]
elif GROQ_API_KEY_INLINE:
    GROQ_API_KEY = GROQ_API_KEY_INLINE
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY_INLINE
else:
    GROQ_API_KEY = None

if GROQ_API_KEY:
    print("GROQ key detected (will be used).")
else:
    print("Warning: GROQ_API_KEY not set. Set env var or put it in GROQ_API_KEY_INLINE in this file.")

# ---------- IMPORTS ----------
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Text splitter import compatibility
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None
        print("ERROR: Could not import RecursiveCharacterTextSplitter. Install langchain_text_splitters.")

# Embeddings
HuggingFaceEmbeddings = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Using HuggingFaceEmbeddings from langchain_huggingface")
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("Using HuggingFaceEmbeddings from langchain_community")
    except Exception:
        HuggingFaceEmbeddings = None
        print("ERROR: HuggingFaceEmbeddings import failed.")

# FAISS vectorstore
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS = None
    print("ERROR: FAISS import failed (langchain_community).")

# GROQ LLM
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
    print("ERROR: langchain_groq import failed.")

# LangChain prompt/runnable imports (best-effort)
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    try:
        from langchain import PromptTemplate
    except Exception:
        PromptTemplate = None
    RunnableParallel = RunnableLambda = RunnablePassthrough = StrOutputParser = None
    print("Warning: PromptTemplate or runnable APIs not available; falling back to simpler calls.")

# ---------- Helpers to normalize transcript objects ----------

def transcript_obj_to_text(tobj: Any) -> str:
    """Convert various transcript-return types into joined plain text."""
    if isinstance(tobj, (list, tuple)) and len(tobj) > 0:
        try:
            return " ".join(chunk.get("text", "") for chunk in tobj if isinstance(chunk, dict))
        except Exception:
            return " ".join(str(x) for x in tobj)

    if hasattr(tobj, "fetch") and callable(getattr(tobj, "fetch")):
        try:
            fetched = tobj.fetch()
            return transcript_obj_to_text(fetched)
        except Exception:
            pass

    for attr in ("_transcript", "transcript", "transcripts", "fetched_transcript", "text"):
        if hasattr(tobj, attr):
            val = getattr(tobj, attr)
            if isinstance(val, str):
                return val
            if isinstance(val, (list, tuple)):
                return transcript_obj_to_text(val)
            try:
                return str(val)
            except Exception:
                pass

    try:
        iter(tobj)
        parts = []
        for item in tobj:
            if isinstance(item, dict) and "text" in item:
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        if parts:
            return " ".join(parts)
    except TypeError:
        pass

    try:
        s = str(tobj)
        if s and not s.startswith("<"):
            return s
    except Exception:
        pass

    return ""


def fetch_youtube_transcript(video_id: str, languages: List[str] = ["en"]) -> str:
    """Robust fetching across youtube-transcript-api versions."""
    try:
        # try get_transcript
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            res = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            text = transcript_obj_to_text(res)
            if text:
                print("Fetched transcript using get_transcript()")
                return text

        # try list (class or instance)
        try:
            tl_obj = YouTubeTranscriptApi.list(video_id)
        except TypeError:
            try:
                inst = YouTubeTranscriptApi()
                tl_obj = inst.list(video_id)
            except Exception:
                tl_obj = None

        if tl_obj is not None:
            if hasattr(tl_obj, "find_transcript"):
                t_obj = tl_obj.find_transcript(languages)
                # try normalize t_obj
                text = transcript_obj_to_text(t_obj)
                if text:
                    print("Fetched transcript via tl_obj.find_transcript(...)")
                    return text
                if hasattr(t_obj, "fetch"):
                    try:
                        fetched = t_obj.fetch()
                        text = transcript_obj_to_text(fetched)
                        if text:
                            print("Fetched transcript after calling fetch() on transcript object")
                            return text
                    except Exception:
                        pass

            # try tl_obj itself
            text = transcript_obj_to_text(tl_obj)
            if text:
                print("Fetched transcript from tl_obj directly")
                return text

        # try class-level fetch
        if hasattr(YouTubeTranscriptApi, "fetch"):
            try:
                res = YouTubeTranscriptApi.fetch(video_id)
                text = transcript_obj_to_text(res)
                if text:
                    print("Fetched transcript using fallback fetch()")
                    return text
            except Exception:
                pass

        print("Transcript fetch returned unexpected or empty type")
        return ""

    except TranscriptsDisabled:
        print("TranscriptsDisabled: no transcripts available.")
        return ""
    except Exception as e:
        print("ERROR fetching transcript:", repr(e))
        return ""


# ---------- Main RAG flow (user input) ----------
# Modified: main now accepts optional args and returns a dict {"answer","summary"}.
def main(user_video: str = None, user_question: str = None):
    """
    If user_video/user_question passed, use them.
    Otherwise, check environment variables VIDEO_ID and QUESTION.
    Otherwise, fall back to interactive input (original behavior).
    Returns: dict {"answer": str, "summary": str}
    """

    # Determine video id
    if user_video and user_video.strip():
        video = user_video.strip()
    else:
        video = os.environ.get("VIDEO_ID", "").strip()
        if not video:
            try:
                # interactive fallback
                video = input("Enter YouTube video id (or press Enter to use default 'Gfr50f6ZBvo'): ").strip()
            except Exception:
                video = ""
        if not video:
            video = "Gfr50f6ZBvo"

    # Determine question
    if user_question and user_question.strip():
        question = user_question.strip()
    else:
        question = os.environ.get("QUESTION", "").strip()
        if not question:
            try:
                question = input("Enter the question you want to ask about the video: ").strip()
            except Exception:
                question = ""
        if not question:
            question = "What is this video about?"

    # initialize default outputs so we always return a dict
    answer_text = ""
    summary_text = ""

    transcript = fetch_youtube_transcript(video, languages=["en"])
    print("Transcript length:", len(transcript))
    if not transcript:
        print("No transcript obtained — exiting.")
        return {"answer": "No transcript obtained.", "summary": ""}

    if RecursiveCharacterTextSplitter is None:
        print("Text splitter not available. Install langchain_text_splitters.")
        return {"answer": "Text splitter not available. Install langchain_text_splitters.", "summary": ""}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    print("Chunks created:", len(chunks))
    if not chunks:
        print("No chunks produced; exiting.")
        return {"answer": "No chunks produced.", "summary": ""}

    # show first chunk preview
    print("\n--- First chunk preview (400 chars) ---\n", chunks[0].page_content[:400].replace("\n", " "))

    if HuggingFaceEmbeddings is None:
        print("HuggingFaceEmbeddings not available. Install langchain_huggingface or langchain_community.")
        return {"answer": "HuggingFaceEmbeddings not available. Install dependencies.", "summary": ""}

    try:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except TypeError:
            embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print("Failed to init embeddings:", e)
        return {"answer": f"Failed to init embeddings: {e}", "summary": ""}

    if FAISS is None:
        print("FAISS not available. Install langchain-community.")
        return {"answer": "FAISS not available. Install langchain-community.", "summary": ""}

    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print("Failed to create FAISS index:", e)
        return {"answer": f"Failed to create FAISS index: {e}", "summary": ""}

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # debug retrieved docs
    q = question
    try:
        try:
            retrieved_docs = retriever.invoke(q)
        except Exception:
            if hasattr(retriever, "get_relevant_documents"):
                retrieved_docs = retriever.get_relevant_documents(q)
            elif hasattr(retriever, "get_documents"):
                retrieved_docs = retriever.get_documents(q)
            else:
                retrieved_docs = []
    except Exception as e:
        print("Retriever error:", e)
        retrieved_docs = []

    print("Retrieved docs count:", len(retrieved_docs))
    for i, d in enumerate(retrieved_docs[:5]):
        print(f"\n--- Retrieved doc {i+1} preview ---\n", getattr(d, "page_content", str(d))[:400].replace("\n", " "))

    # relaxed prompt: prefer context but allow fallback
    if PromptTemplate is not None:
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
First try to answer ONLY from the provided transcript context.
If the context does not contain the answer, you may answer using general knowledge — prefix such answers with "[NOT IN TRANSCRIPT]".

Context:
{context}

Question: {question}
""",
            input_variables=["context", "question"]
        )
    else:
        prompt = None

    if ChatGroq is None:
        print("ChatGroq not available. Install langchain_groq.")
        return {"answer": "ChatGroq not available. Install langchain_groq.", "summary": ""}

    if not GROQ_API_KEY:
        print("GROQ_API_KEY missing. Set environment var or edit GROQ_API_KEY_INLINE.")
        return {"answer": "GROQ_API_KEY missing.", "summary": ""}

    try:
        # Use a supported model
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=GROQ_API_KEY)
    except Exception as e:
        print("Failed to initialize ChatGroq:", e)
        return {"answer": f"Failed to initialize ChatGroq: {e}", "summary": ""}

    context_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": q}) if prompt is not None else f"{context_text}\n\nQuestion: {q}"

    try:
        ans = llm.invoke(final_prompt)
        # capture the content into answer_text so we can return it
        answer_text = getattr(ans, "content", str(ans))
        print("\nANSWER:\n", answer_text)
    except Exception as e:
        print("Error during generation:", e)
        return {"answer": f"Error during generation: {e}", "summary": ""}

    # optional summary via LLM
    try:
        if RunnableParallel is not None and PromptTemplate is not None:
            def format_docs(docs):
                return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })
            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | llm | parser
            summary = main_chain.invoke("Summarize the video.")
            summary_text = summary if isinstance(summary, str) else getattr(summary, "content", str(summary))
            print("\nSUMMARY:\n", summary_text)
        else:
            # fallback summary prompt
            fallback_prompt = f"Context:\n{context_text}\n\nQuestion: Summarize the video."
            summary_ans = llm.invoke(fallback_prompt)
            summary_text = getattr(summary_ans, "content", str(summary_ans))
            print("\nSUMMARY (fallback):\n", summary_text)
    except Exception as e:
        print("Skipping summary due to:", e)
        # keep summary_text possibly empty

    # Final returned result (always a dict)
    return {"answer": answer_text, "summary": summary_text}

# Allow external call via run_agent
def run_agent(video_id: str, question: str):
    """
    Programmatic wrapper for the agent.
    Calls main(video, question) and returns a dict {"answer":..., "summary":...}.
    """
    try:
        return main(user_video=video_id, user_question=question)
    except Exception as e:
        return {"answer": f"Agent error: {repr(e)}", "summary": ""}


if __name__ == "__main__":
    # If run directly, use env vars or interactive input, then print one JSON line
    vid = os.environ.get("VIDEO_ID", "") or None
    qn = os.environ.get("QUESTION", "") or None
    out = run_agent(vid, qn)
    # Print only one JSON line at the end so agent2 can parse it
    print(json.dumps(out))
