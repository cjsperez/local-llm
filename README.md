# 🧠 AI Support Agent API – ParkNCharge  
FastAPI backend for brand-based document Q&A, using LangChain + Ollama.

## 📡 Base URLs

- **SERVER:** `https://hippo-enabled-wombat.ngrok-free.app`
- **Public (via Ngrok):** `https://hippo-enabled-wombat.ngrok-free.app`

## 🧪 API Quick Start (JS `fetch`)

### 1. 📁 Upload Document

```js
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('{SERVER}/documents/upload/{brand_key}', {
  method: 'POST',
  body: formData,
});
```

### 2. 🏷️ Create Brand

```js
fetch('{SERVER}/brands', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    key: '{brand_key}',
    display_name: 'ParkNCharge',
    products: ['EV charging', 'E-Scooter Charging', 'Fire Isolator'],
    support_email: 'support@parkncharge.com',
    word_limit: 30,
    off_topic_response: "Sorry, I can't help with that.",
    corrections: {
      'Park and charge': 'ParkNCharge',
      'Park and church': 'ParkNCharge',
    },
    prompt_template: "You are {display_name}'s AI assistant. Use this context:\n{context}\nQuestion: {question}\nHistory: {conversation_history}",
    system_message: "Rules:\n1. Correct: {corrections}\n2. Use 'we'\n3. {word_limit} word limit\n4. Off-topic: '{off_topic_response}'",
  }),
});
```

### 3. 🧠 Create Vector Store

```js
fetch('{SERVER}/vector_store/create', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    brand_key: '{brand_key}',
    file_ids: ['your_file_id_here'],
    recreate: false,
  }),
});
```

### 4. 🔁 Rebuild Vector Store

```js
fetch('{SERVER}/vector_store/rebuild', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ brand_key: '{brand_key}' }),
});
```

### 5. ❌ Delete Brand / File / Vector Store

```js
fetch('{SERVER}/brands/{brand_key}', { method: 'DELETE' });
fetch('{SERVER}/documents/{brand_key}/your-file.json', { method: 'DELETE' });
fetch('{SERVER}/vector_store/delete?brand_key={brand_key}', { method: 'DELETE' });
```

### 6. 💬 Query AI

#### Standard (non-streamed)
```js
fetch('{SERVER}/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: 'How much is the charging fee?',
    brand_key: '{brand_key}',
    thread_id: null,
  }),
})
.then(res => res.json())
.then(console.log);
```

#### Streamed (for real-time output)
```js
const res = await fetch('{SERVER}/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: 'List of EV Charger Packages?',
    brand_key: '{brand_key}',
    thread_id: null,
  }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let fullText = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  fullText += decoder.decode(value);
}

console.log(fullText);
```

## 🛠️ Server Management

### ✅ Health Check
```js
fetch('{SERVER}/health').then(res => res.json());
```

### 🧠 Restart Ollama
```bash
nohup ollama serve > ollama.log 2>&1 &
```

### 🔁 Restart API Server
```bash
ps aux | grep uvicorn
sudo kill <PID>
nohup /workspace/localrag/ai-agent/venv/bin/uvicorn server:app --host 0.0.0.0 --port 4001 --workers 4 > output.log 2>&1 &
```

## 📄 Dev Workflow (Frontend Guide)

1. **Create brand**
2. **Upload documents**
3. **Create vector store using file IDs**
4. **Use `/query/stream`** for real-time answers
5. **Maintain `thread_id`** for conversational context