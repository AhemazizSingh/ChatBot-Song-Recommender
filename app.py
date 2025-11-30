# app.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from utils import analyze_emotion_with_ibm, groq_get_reply, lastfm_top_tracks_by_tag, lastfm_similar_tracks

load_dotenv()
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

TONE_TO_LASTFM_TAG = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "melancholic",
    "disgust": "dark",
    "neutral": "chill"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tone", methods=["POST"])
def tone_endpoint():
    data = request.get_json(force=True)
    text = ""
    if "context" in data:
        text = " ".join(data["context"][-3:])
    else:
        text = data.get("text", "")

    tone_info = analyze_emotion_with_ibm(text)
    tone_id = tone_info.get("tone", "neutral")
    lastfm_tag = TONE_TO_LASTFM_TAG.get(tone_id, "chill")
    return jsonify({"tone_id": tone_id, "score": tone_info.get("score", 0.0), "lastfm_tag": lastfm_tag})

@app.route("/response", methods=["POST"])
def response_endpoint():
    data = request.get_json(force=True)
    context = data.get("context", [])[-6:]
    tone = data.get("tone", "neutral")
    try:
        reply = groq_get_reply(context, tone=tone)
    except Exception as e:
        reply = f"Error generating reply: {e}"
    return jsonify({"response": reply})

@app.route("/songs", methods=["POST"])
def songs_endpoint():
    data = request.get_json(force=True)
    tag = data.get("tag")
    if not tag:
        return jsonify({"error": "tag required"}), 400
    try:
        tracks = lastfm_top_tracks_by_tag(tag, limit=8)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"tracks": tracks})

@app.route("/simmilarsongs", methods=["POST"])
def sim_songs_endpoint():
    data = request.get_json(force=True)
    track = data.get("track")
    artist = data.get("artist")
    if not track or not artist:
        return jsonify({"error": "track and artist required"}), 400
    try:
        similar = lastfm_similar_tracks(track, artist, limit=8)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"tracks": similar})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
