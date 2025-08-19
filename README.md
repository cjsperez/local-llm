# 🧠 Local LLM API with Brand-based RAG Support

A FastAPI-powered local Retrieval-Augmented Generation (RAG) backend that supports multi-brand document ingestion, vector store management, and AI querying using [Ollama](https://ollama.com/).

## 📦 Features

- Health checks
- Brand creation and context configuration
- Document upload/delete per brand
- Vector store management (create/rebuild/delete)
- AI querying with streaming support
- Ngrok tunneling for remote access
- Ollama integration for local LLM inference
- Shell utilities for deployment and monitoring

---

## 🚀 Quickstart

### ✅ Health Check

```bash
curl -X GET http://localhost:4001/health
curl -X GET https://<your-ngrok-domain>/health
```

---

## 🧾 Document Management

### Upload Document

```bash
curl -X POST -F "file=@/path/to/file.json" http://localhost:4001/documents/upload/<brand_key>
```

### Delete Document

```bash
curl -X DELETE http://localhost:4001/documents/<brand_key>/<filename>
```

---

## 🏷️ Brand Management

### Create Brand

```bash
curl -X POST http://localhost:4001/brands -H "Content-Type: application/json" -d '{
  "key": "pnc",
  "display_name": "ParkNCharge",
  "products": ["EV charging", "Fire Isolator"],
  ...
}'
```

### Delete Brand

```bash
curl -X DELETE http://localhost:4001/brands/<brand_key>
```

---

## 📚 Vector Store

### Create

```bash
curl -X POST http://localhost:4001/vector_store/create -H "Content-Type: application/json" -d '{
  "brand_key": "pnc",
  "file_ids": ["uuid-here"],
  "recreate": false
}'
```

### Rebuild

```bash
curl -X POST http://localhost:4001/vector_store/rebuild -H "Content-Type: application/json" -d '{
  "brand_key": "pnc"
}'
```

### Delete

```bash
curl -X DELETE "http://localhost:4001/vector_store/delete?brand_key=pnc"
```

---

## 💬 Query the Assistant

### Basic Query

```bash
curl -X POST http://localhost:4001/query -H "Content-Type: application/json" -d '{
  "question": "What is Fire Isolator?",
  "brand_key": "pnc"
}'
```

### With Conversation Thread

```bash
curl -X POST http://localhost:4001/query -H "Content-Type: application/json" -d '{
  "question": "Why did charging stop?",
  "thread_id": "<uuid>",
  "brand_key": "pnc"
}'
```

---

## 🛠️ Developer Notes

### Start FastAPI server

```bash
nohup uvicorn server:app --host 0.0.0.0 --port 4001 --workers 4 > output.log 2>&1 &
```

### Start Ollama

```bash
nohup ollama serve > ollama.log 2>&1 &
```

### Tail Logs

```bash
sudo tail -f /var/log/fastapi_app/stderr.log
```

### Restart server manually

```bash
ps aux | grep uvicorn
sudo kill <PID>
```

---

## 🧰 Deployment Setup (Linux)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To restart via Supervisor:

```bash
supervisorctl restart rag-api
```

---

## 📄 Sample Document Extractor

```bash
python ./utils/doc_extractor.py --url "https://www.parkncharge.ph" --output ./sample_docs/doc11.json
```

---

## 📎 Credits

Built by Crisjahn Perez
