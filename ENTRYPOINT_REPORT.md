# Entry Point Decision Report

## Context
The repository currently contains the following executable Python scripts:

1. **`inference.py`** – Performs transcription on a *single hard-coded* audio file (`unseen_audio_data/rw-unseen-001.mp3`) and saves the result to `transcription_output/`. It is **not** parameterised, so running it directly in a container would make the image useful only for that single file.
2. **`testing.py`** – Iterates over all `.wav` files in `audio/` and writes their transcriptions to `transcriptions.txt`. While more flexible than `inference.py`, it still assumes the presence of the full dataset inside the container and does not expose an API.
3. **`train.py`** – Fine-tunes the Whisper model; this is a development workflow, not a runtime service.

There is **no existing `app.py`** (or similar) that exposes functionality over HTTP, yet the user-supplied Dockerfile snippet ends with:

```Dockerfile
EXPOSE 5000
CMD ["python", "app.py"]
```

indicating the desired container interface is **HTTP on port 5000**.

---

## Coolify Deployment Context
Coolify detects the port exposed by the running process and maps it to a publicly accessible URL. By exposing **port `5000`** inside the image and running the Flask server on the **`PORT`** environment variable (defaulting to 5000), we ensure seamless deployment on Coolify without extra configuration.

> **Outcome:** Coolify will give us a public HTTPS endpoint that N8N can invoke directly at runtime **without incurring extra cloud-function charges**.

---

## Decision
- A new **`app.py`** Flask application will be introduced.
- Core transcription logic from `inference.py` will be refactored into a reusable function `transcribe_audio(file_path: str) -> str` so that both CLI and HTTP layers share the same implementation.
- The container's default command will therefore remain `python app.py`, matching the original snippet and exposing `POST /transcribe` for inference **on the port provided by Coolify (`$PORT`, default 5000)**.

This approach offers:

1. **Flexibility** – Any external client can upload arbitrary audio without rebuilding the image.
2. **Consistency** – Aligns with the exposed port 5000 in the Dockerfile.
3. **Re-usability** – Shared core logic avoids duplication and keeps training/inference concerns separate.
4. **Deployability** – Meets Coolify's expectations for a single exposed port and a long-running HTTP process, giving us a stable public endpoint for N8N.

---

## Next Steps
The following tasks depend on this decision (IDs reference Shrimp task list):

1. **Refactor transcription logic** (`767e1abd-0e72-4ec8-b22d-52b6953c4923`).
2. **Implement Flask server** (`d0442309-2645-401a-bb5f-be5b3cd3b2b8`) **with automatic port selection from the `PORT` env variable expected by Coolify**.
3. **Create Dockerfile** (`cdd61ed7-db07-42de-ad9f-87a3ea117eed`).

The Docker container will therefore execute:

```bash
python app.py
```

which launches the HTTP service on port 5000 as intended. 