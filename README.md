# Cloud RAG

Upload any file, ask any question. Answers come from your file first, then the web.
Supports PDF, Word, Excel, PowerPoint, images, HTML, text, CSV, JSON, and code files.

## Run locally

Requires Python 3.10+ and an Anthropic API key (https://console.anthropic.com).
Web search must be enabled for your organization in the Console. Without it, untick "Search the web" in the app.

### macOS / Linux
```bash
cd cloud-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env and paste your ANTHROPIC_API_KEY
uvicorn main:app --reload --port 8000
```

### Windows (PowerShell)
```powershell
cd cloud-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then edit .env and paste your ANTHROPIC_API_KEY
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000. The first upload downloads a small embedding model (~130 MB), so it is slower once.

## Test from the terminal
```bash
curl -F "files=@report.pdf" http://localhost:8000/api/upload
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" \
  -d '{"question":"Summarize the key points","use_web":true}'
```

## Run with Docker
```bash
docker build -t cloud-rag .
docker run -p 8000:8000 --env-file .env -v ragdata:/data cloud-rag
```

## Deploy to the cloud
Any host that runs a Docker container works (Google Cloud Run, Render, Fly.io, Railway, AWS App Runner).
Set `ANTHROPIC_API_KEY` as a secret and mount a persistent volume at `/data` so uploads survive restarts.
Example, Google Cloud Run:
```bash
gcloud run deploy cloud-rag --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars ANTHROPIC_API_KEY=sk-ant-...
```
Cloud Run's disk is temporary, so for production swap the file store in `main.py` (`load`/`save`) for a managed vector DB such as pgvector, Pinecone, or Qdrant. Add authentication before exposing it publicly.

## How it works
1. Upload: the file is converted to text, split into overlapping chunks, embedded, and stored in `data/`.
2. Ask: the question is embedded, the 5 closest chunks are retrieved, and Claude answers with them as context.
3. External sources: when "Search the web" is on, Claude can use web search for anything the files don't cover, and web links are listed as sources.
