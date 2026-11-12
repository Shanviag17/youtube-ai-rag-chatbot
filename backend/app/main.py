
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

from app.config import settings
from app.services.youtube import extract_video_id, fetch_transcript
from app.services.rag import rag


app = FastAPI(title="YouTube AI RAG API")


# Allow requests from local development and deployed Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://youtube-ai-rag-chatbot.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


class IngestBody(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


class ChatBody(BaseModel):
    video_id: str
    question: str = Field(min_length=2, max_length=2000)
    history: list[dict] = []


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "chunks": rag.collection.count()
    }


@app.post("/api/videos/ingest")
def ingest(body: IngestBody):
    try:
        vid = extract_video_id(body.url)
        snippets, language = fetch_transcript(vid)
        chunks = rag.ingest(vid, snippets)

        return {
            "video_id": vid,
            "title": f"YouTube video {vid}",
            "language": language,
            "chunks": len(chunks),
            "transcript_chars": sum(
                len(s["text"]) for s in snippets
            )
        }

    except Exception as e:
        raise HTTPException(
            400,
            f"Transcript ingestion failed. Captions may be disabled, unavailable, or blocked. Details: {e}"
        )


@app.post("/api/chat")
def chat(body: ChatBody):
    try:
        passages = rag.retrieve(
            body.video_id,
            body.question
        )

        if (
            not passages
            or max(p["score"] for p in passages)
            < settings.min_relevance_score
        ):
            return {
                "answer": "I couldn't find enough information about this in the video transcript.",
                "sources": [],
                "grounded": False
            }

        if not settings.groq_api_key:
            raise HTTPException(
                503,
                "Add GROQ_API_KEY to the Render environment variables."
            )

        context = "\n\n".join(
            f"[{i}] ({int(p['start'])}s-{int(p['end'])}s) {p['text']}"
            for i, p in enumerate(passages, 1)
        )

        system = (
            "You answer questions using ONLY the transcript excerpts "
            "provided in the user message. "
            "Do not use outside knowledge or guess. "
            "Cite every factual claim using [1], [2], etc. "
            "Never invent facts, quotes or timestamps. "
            "If the evidence is insufficient, reply exactly: "
            "I couldn't find enough information about this in the video transcript. "
            "If a question has an unsupported premise, say the transcript does not establish it."
        )

        msgs = [
            {
                "role": "system",
                "content": system
            }
        ]

        for m in body.history[-6:]:
            if (
                m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str)
            ):
                msgs.append({
                    "role": m["role"],
                    "content": m["content"][:1200]
                })

        msgs.append({
            "role": "user",
            "content": (
                f"Question: {body.question}\n\n"
                f"Transcript excerpts:\n{context}\n\n"
                "Answer only from this evidence and cite it."
            )
        })

        client = Groq(
            api_key=settings.groq_api_key
        )

        result = client.chat.completions.create(
            model=settings.groq_model,
            messages=msgs,
            temperature=0.1,
            max_tokens=800
        )

        answer = result.choices[0].message.content.strip()

        sources = [
            {
                "citation": i,
                "text": p["text"],
                "start": p["start"],
                "end": p["end"],
                "url": (
                    f"https://www.youtube.com/watch?v="
                    f"{body.video_id}&t={int(p['start'])}s"
                )
            }
            for i, p in enumerate(passages, 1)
        ]

        return {
            "answer": answer,
            "sources": sources,
            "grounded": (
                "couldn't find enough information"
                not in answer.lower()
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            500,
            f"Chat failed: {e}"
        )
