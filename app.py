from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
import uuid
import os
import tempfile

# Attempt to import the reusable transcription helper. It will be added in a later task.
try:
    from inference import transcribe_audio  # type: ignore
except (ImportError, AttributeError):
    transcribe_audio = None  # fallback until refactor is complete

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribes an uploaded audio file using Whisper.

    Expects a multipart/form-data request with a single file field named `file`.
    Returns JSON: { "text": "..." }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Save to a temporary file inside AUDIO_DIR
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(AUDIO_DIR, filename)
    file.save(filepath)

    if transcribe_audio is None:
        # In development, transcribe_audio may not yet be implemented
        return (
            jsonify({"error": "transcribe_audio() not available yet."}),
            501,
        )

    try:
        text = transcribe_audio(filepath)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file (optional – comment out if you need persistence)
        try:
            os.remove(filepath)
        except OSError:
            pass


@app.route("/tts", methods=["POST"])
def tts():
    """Generates TTS audio for provided text.

    Body JSON: { "text": "hello" }
    Returns JSON: { "audio_url": "http://.../audio/<filename>.mp3" }
    """
    data = request.get_json(force=True)
    text = data.get("text", "") if isinstance(data, dict) else ""
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    # Note: `tts.py` must exist and accept these CLI arguments.
    try:
        subprocess.run(
            [
                "python",
                "tts.py",
                "--text",
                text,
                "--lang",
                "rw",
                "--output",
                filepath,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"TTS generation failed: {e}"}), 500

    return jsonify({"audio_url": f"{request.url_root}audio/{filename}"})


@app.route("/audio/<path:filename>")
def audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


if __name__ == "__main__":
    # Coolify sets PORT env var; default to 5000 locally.
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 