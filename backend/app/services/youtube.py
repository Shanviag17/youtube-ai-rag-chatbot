import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    video_id = ""
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
                video_id = parts[1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or ""):
        raise ValueError("Please enter a valid YouTube video URL.")
    return video_id

def fetch_transcript(video_id: str):
    # youtube-transcript-api v1.x returns FetchedTranscriptSnippet objects.
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
    snippets = []
    for item in fetched:
        text = " ".join(str(getattr(item, "text", "")).split())
        start = float(getattr(item, "start", 0.0))
        duration = float(getattr(item, "duration", 0.0))
        if text:
            snippets.append({"text": text, "start": start, "end": start + duration})
    if not snippets:
        raise ValueError("No accessible transcript was found for this video.")
    return snippets, getattr(fetched, "language", "unknown")
