DEMO

https://github.com/user-attachments/assets/bbaf4d7f-d14c-4b7e-a9ba-f47f41732218

# Customer Support Voice RAG Agent 🎙️

Real-time voice-enabled customer support agent using **Gemini LLM**, **Deepgram STT**, **Cartesia TTS**, and **LlamaIndex RAG**. Features continuous listening, push-to-talk, and WebSocket-based real-time communication.

![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square) ![Flask](https://img.shields.io/badge/flask-socketio-red?style=flat-square) ![Status](https://img.shields.io/badge/status-production--ready-green?style=flat-square)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- **Google API Key** (Gemini - FREE)
- **Deepgram API Key** (Speech-to-Text)
- **Cartesia API Key** (Text-to-Speech)

### Installation


# 1. Clone & setup
git clone https://github.com/Jawadbro/customer-support-voice-rag-agent.git
cd customer-support-voice-rag-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install flask flask-socketio python-dotenv llama-index
pip install google-generativeai requests pydantic numpy
pip install pymupdf pdfplumber

# 3. Create .env file
cat > .env << EOL
GOOGLE_API_KEY=your_google_api_key
DEEPGRAM_API_KEY=your_deepgram_key
CARTESIA_API_KEY=your_cartesia_key
CARTESIA_VOICE_ID=your_voice_id
EOL

# 4. Add documents to data/ folder (optional - car.txt already included)

# 5. Run
python app.py


Open **http://localhost:5000** in your browser.

## 📁 Project Structure

```
├── app.py                         # Main application
├── .env                          # API keys
├── data/
│   └── car.txt                   # Knowledge base documents
└── query-engine-storage-free/    # Vector store (auto-generated)
```

## 🎤 Usage

**Voice Modes:**
- **Continuous Listening**: Click "Start Continuous Listening" - speak naturally
- **Push-to-Talk**: Hold button while speaking
- **Text Input**: Type questions directly

**Adding Documents:**
- Drop PDF, TXT, DOCX, HTML, MD files in `data/` folder
- Delete `query-engine-storage-free/` and restart to rebuild knowledge base

## ⚡ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │───▶│  Flask-SocketIO  │───▶│   Audio Buffer  │
│   (Microphone)  │    │    (WebSocket)   │    │   (Real-time)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Cartesia TTS   │◀───│  Response Engine │◀───│ Deepgram STT   │
│  (Audio Output) │    │  (Gemini + RAG)  │    │ (Transcription) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │ LlamaIndex VDB  │
                       │ (Gemini Embed)  │
                       └─────────────────┘
```

**Tech Stack:**
- **LLM**: Google Gemini 1.5 Flash (Free)
- **Embeddings**: Gemini Embeddings (Free)  
- **STT**: Deepgram Nova-2
- **TTS**: Cartesia Sonic
- **RAG**: LlamaIndex with persistent vector store
- **Real-time**: Flask-SocketIO WebSockets

## 🛠️ Configuration

Key settings in `app.py`:
```python
Settings.chunk_size = 512
Settings.chunk_overlap = 50
similarity_top_k = 5
model_name = "gemini-1.5-flash"
```

## 🚀 Deployment

**Development:**
```bash
python app.py  # http://localhost:5000
```

**Production:**
```bash
pip install gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
```

## 🔧 Troubleshooting

- **Microphone issues**: Ensure HTTPS in production, check browser permissions
- **Knowledge base errors**: Delete `query-engine-storage-free/` folder and restart
- **API errors**: Verify all API keys in `.env` file
- **Audio playback**: Check browser audio permissions

## 📈 Performance
- Response time: ~1-2 seconds
- Supports 10-50 concurrent users
- Real-time WebSocket communication
- Persistent vector storage

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Test voice interactions
4. Submit pull request

## 📄 License

MIT License

---

**Built with Gemini (Free) + Deepgram + Cartesia for cost-effective real-time voice AI**
