# utils.py
import os
import json
import re
import requests
from dotenv import load_dotenv
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

load_dotenv()

# --- Env & constants ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

IBM_NLU_APIKEY = os.getenv("IBM_NLU_APIKEY")
IBM_NLU_URL = os.getenv("IBM_NLU_URL")

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

# --- IBM NLU init ---
if IBM_NLU_APIKEY and IBM_NLU_URL:
    authenticator = IAMAuthenticator(IBM_NLU_APIKEY)
    nlu = NaturalLanguageUnderstandingV1(
        version="2021-08-01",
        authenticator=authenticator
    )
    nlu.set_service_url(IBM_NLU_URL)
else:
    nlu = None

# --- Groq helper functions ---
def groq_get_reply(context_list, tone="neutral", model="llama-3.1-8b-instant", max_tokens=150, temperature=0.7):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment (.env)")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    tone_map = {
        "joy": "Be upbeat and positive.",
        "sadness": "Be empathetic and gentle.",
        "anger": "Be calm and neutral.",
        "fear": "Be reassuring.",
        "analytical": "Be logical and precise.",
        "neutral": "Be neutral and informative."
    }
    system_instruction = "You are a helpful chatbot. " + tone_map.get(tone, "")

    messages = [{"role": "system", "content": system_instruction}]
    for i, m in enumerate(context_list):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": m})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20.0)
    print("Groq error details:", resp.status_code, resp.text) 
    resp.raise_for_status()
    j = resp.json()
    choices = j.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "").strip()
    return ""

# --- IBM NLU emotion analysis ---
def analyze_emotion_with_ibm(text):
    if not nlu or not text.strip():
        return {"tone": "neutral", "score": 0.0}

    response = nlu.analyze(
        text=text,
        features=Features(emotion=EmotionOptions())
    ).get_result()

    emotions = response.get("emotion", {}).get("document", {}).get("emotion", {})
    if not emotions:
        return {"tone": "neutral", "score": 0.0}

    top_emotion = max(emotions.items(), key=lambda kv: kv[1])
    tone, score = top_emotion
    return {"tone": tone, "score": score}

# --- Last.fm helpers ---
def lastfm_top_tracks_by_tag(tag, limit=8):
    if not LASTFM_API_KEY:
        return []
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "tag.gettoptracks",
        "tag": tag,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        tracks = data.get("tracks", {}).get("track", [])
    except Exception as e:
        print("LastFM tag error:", e)
        return []

    results = []
    for t in tracks:
        artist = t.get("artist", {}).get("name") if isinstance(t.get("artist"), dict) else None
        results.append({
            "name": t.get("name"),
            "artist": artist,
            "url": t.get("url")
        })
    return results

def lastfm_similar_tracks(track, artist, limit=8):
    if not LASTFM_API_KEY:
        return []
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getsimilar",
        "artist": artist,
        "track": track,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        similars = data.get("similartracks", {}).get("track", [])
    except Exception as e:
        print("LastFM similar error:", e)
        return []

    results = []
    for t in similars:
        artist_name = t.get("artist", {}).get("name") if isinstance(t.get("artist"), dict) else t.get("artist")
        results.append({
            "name": t.get("name"),
            "artist": artist_name,
            "url": t.get("url")
        })
    return results
