from pathlib import Path
from dotenv import load_dotenv
import os
import asyncio
import json
import threading
import time
from typing import Any, List
from collections import deque

# Core Flask and WebSocket imports
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit

# LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.base.llms.types import ChatMessage as ChatMessageType

# API imports
import google.generativeai as genai
import requests
from pydantic import Field
import numpy as np
import hashlib

# PDF processing
try:
    import fitz  # PyMuPDF
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PDF processing libraries not available. Install: pip install pymupdf pdfplumber")

load_dotenv()

class GeminiFreeEmbedding(BaseEmbedding):
    """Free embedding using Google Gemini embedding models"""
    
    def __init__(self, model_name: str = "models/embedding-001"):
        super().__init__()
        self.model_name = model_name
        self._embed_batch_size = 100
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            # Fallback to simple hash-based embedding
            return self._create_simple_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            return self._create_simple_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                embeddings.append(self._create_simple_embedding(text))
        return embeddings
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple hash-based embedding as fallback"""
        # Create a consistent embedding from text hash
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            embedding.append(int(hex_pair, 16) / 255.0)
        
        # Pad to required dimension (768 for Gemini)
        while len(embedding) < 768:
            embedding.extend(embedding[:min(len(embedding), 768 - len(embedding))])
        
        return embedding[:768]

    # Async methods
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._get_text_embeddings, texts)

class LLMMetadata:
    def __init__(self, model_name: str, context_window: int, max_tokens: int, temperature: float, num_output: int):
        self.model_name = model_name
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_output = num_output
        self.is_chat_model = True

class GeminiLLM(LLM):
    model_name: str = Field(default="gemini-1.5-flash")
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model_name)
        object.__setattr__(self, 'context_window', 1048576)
        object.__setattr__(self, 'max_tokens', 8192)
        object.__setattr__(self, 'temperature', 0.1)
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            context_window=getattr(self, 'context_window', 1048576),
            max_tokens=getattr(self, 'max_tokens', 8192),
            temperature=getattr(self, 'temperature', 0.1),
            num_output=getattr(self, 'max_tokens', 8192)
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        try:
            response = self._model.generate_content(prompt)
            return CompletionResponse(text=response.text if response.text else "I couldn't generate a response.")
        except Exception as e:
            return CompletionResponse(text=f"Error: {str(e)}")
    
    def stream_complete(self, prompt: str, **kwargs):
        try:
            response = self._model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield CompletionResponse(text=chunk.text, delta=chunk.text)
        except Exception as e:
            yield CompletionResponse(text=f"Error: {str(e)}", delta=f"Error: {str(e)}")
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        try:
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            return CompletionResponse(text=response.text if response.text else "I couldn't generate a response.")
        except Exception as e:
            return CompletionResponse(text=f"Error: {str(e)}")
    
    async def astream_complete(self, prompt: str, **kwargs):
        try:
            def _generate():
                response = self._model.generate_content(prompt, stream=True)
                chunks = []
                for chunk in response:
                    if chunk.text:
                        chunks.append(CompletionResponse(text=chunk.text, delta=chunk.text))
                return chunks
            
            chunks = await asyncio.to_thread(_generate)
            for chunk in chunks:
                yield chunk
        except Exception as e:
            yield CompletionResponse(text=f"Error: {str(e)}", delta=f"Error: {str(e)}")
    
    def chat(self, messages: List[ChatMessageType], **kwargs) -> ChatResponse:
        try:
            prompt = self._messages_to_prompt(messages)
            response = self._model.generate_content(prompt)
            return ChatResponse(message=ChatMessage(role="assistant", content=response.text if response.text else "I couldn't generate a response."))
        except Exception as e:
            return ChatResponse(message=ChatMessage(role="assistant", content=f"Error: {str(e)}"))
    
    def _messages_to_prompt(self, messages: List[ChatMessageType]) -> str:
        prompt_parts = []
        for msg in messages:
            content = getattr(msg, 'content', str(msg))
            role = getattr(msg, 'role', 'user')
            
            if role == "system":
                prompt_parts.append(f"Instructions: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")
        
        return "\n\n".join(prompt_parts)

    async def achat(self, messages: List[ChatMessageType], **kwargs) -> ChatResponse:
        try:
            prompt = self._messages_to_prompt(messages)
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            return ChatResponse(message=ChatMessage(role="assistant", content=response.text if response.text else "I couldn't generate a response."))
        except Exception as e:
            return ChatResponse(message=ChatMessage(role="assistant", content=f"Error: {str(e)}"))
    
    def stream_chat(self, messages: List[ChatMessageType], **kwargs):
        try:
            prompt = self._messages_to_prompt(messages)
            response = self._model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield ChatResponse(
                        message=ChatMessage(role="assistant", content=chunk.text),
                        delta=ChatMessage(role="assistant", content=chunk.text)
                    )
        except Exception as e:
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=f"Error: {str(e)}"),
                delta=ChatMessage(role="assistant", content=f"Error: {str(e)}")
            )
    
    async def astream_chat(self, messages: List[ChatMessageType], **kwargs):
        try:
            prompt = self._messages_to_prompt(messages)
            def _generate():
                response = self._model.generate_content(prompt, stream=True)
                chunks = []
                for chunk in response:
                    if chunk.text:
                        chunks.append(ChatResponse(
                            message=ChatMessage(role="assistant", content=chunk.text),
                            delta=ChatMessage(role="assistant", content=chunk.text)
                        ))
                return chunks
            
            chunks = await asyncio.to_thread(_generate)
            for chunk in chunks:
                yield chunk
        except Exception as e:
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=f"Error: {str(e)}"),
                delta=ChatMessage(role="assistant", content=f"Error: {str(e)}")
            )

# Initialize components
print("🚀 Initializing Gemini components...")

gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)

try:
    embedding_model = GeminiFreeEmbedding()
    print("✅ Gemini embedding model configured")
except Exception as e:
    print(f"⚠️ Embedding model error: {e}")
    exit(1)

try:
    gemini_llm = GeminiLLM(api_key=gemini_api_key)
    print("✅ Gemini LLM configured")
except Exception as e:
    print(f"⚠️ LLM error: {e}")
    exit(1)

Settings.llm = gemini_llm
Settings.embed_model = embedding_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Knowledge base setup
THIS_DIR = Path(__file__).parent
PERSIST_DIR = THIS_DIR / "query-engine-storage-free"

if not PERSIST_DIR.exists():
    print("📚 Building knowledge base...")
    data_dir = THIS_DIR / "data"
    if not data_dir.exists():
        data_dir.mkdir()
        print(f"📁 Created data directory at {data_dir}. Please add documents.")
        exit(1)
    
    documents = []
    for file_path in data_dir.iterdir():
        if file_path.is_file():
            print(f"Processing: {file_path.name}")
            try:
                if file_path.suffix.lower() == '.pdf' and PDF_AVAILABLE:
                    text_content = ""
                    try:
                        with pdfplumber.open(file_path) as pdf:
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text_content += page_text + "\n"
                        if text_content.strip():
                            print(f"✅ PDF extracted: {len(text_content)} characters")
                        else:
                            raise Exception("No text extracted")
                    except Exception as e:
                        print(f"⚠️ PDF extraction failed: {e}")
                        fallback_docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                        for doc in fallback_docs:
                            text_content += doc.text + "\n"
                    
                    if text_content.strip():
                        doc = Document(text=text_content, metadata={"source": str(file_path)})
                        documents.append(doc)
                else:
                    file_docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                    documents.extend(file_docs)
                    print(f"✅ Text file loaded: {file_path.name}")
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
    
    if not documents:
        print("❌ No documents found. Please add some.")
        exit(1)
    
    print(f"📄 Found {len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {len(doc.text)} characters")
        print(f"Preview: {doc.text[:200]}...")
        print("---")
    
    index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("✅ Knowledge base built!")
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, embed_model=embedding_model)
    print("✅ Knowledge base loaded!")

query_engine = index.as_query_engine(
    llm=gemini_llm,
    similarity_top_k=5,
    response_mode="tree_summarize"
)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# API keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "default-voice-id")

if not DEEPGRAM_API_KEY or not CARTESIA_API_KEY:
    raise ValueError("Deepgram or Cartesia API keys not found")

# Real-time processing state
class RealTimeState:
    def __init__(self):
        self.is_listening = False
        self.audio_buffer = deque(maxlen=100)
        self.silence_count = 0
        self.speech_detected = False
        self.processing = False

user_states = {}

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Voice RAG Agent</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 50px; background: #f0f0f0; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            button { padding: 15px 30px; font-size: 18px; cursor: pointer; margin: 10px; border: none; border-radius: 8px; }
            .listen-btn { background: #4CAF50; color: white; }
            .listen-btn.active { background: #f44336; }
            .status { margin: 20px 0; padding: 15px; border-radius: 5px; background: #e3f2fd; }
            .listening { background: #ffebee !important; color: #c62828; }
            .processing { background: #fff3e0 !important; color: #ef6c00; }
            .response { background: #e8f5e8 !important; color: #2e7d32; }
            input { width: 60%; padding: 10px; font-size: 16px; border: 2px solid #ddd; border-radius: 5px; }
            .send-btn { background: #2196F3; color: white; }
            .controls { margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Real-time Voice RAG Agent</h1>
            <p>Press and hold to talk, or use continuous listening mode</p>
            
            <div class="controls">
                <button id="toggleListening" class="listen-btn">Start Continuous Listening</button>
                <button id="pushToTalk" class="listen-btn">Push to Talk (Hold)</button>
            </div>
            
            <div class="controls">
                <input type="text" id="textInput" placeholder="Or type your question here...">
                <button id="sendButton" class="send-btn">Send</button>
            </div>
            
            <div id="status" class="status">Status: Ready to help you! 🤖</div>
        </div>

        <script>
            let socket;
            let mediaRecorder;
            let audioChunks = [];
            let isListening = false;
            let continuousMode = false;
            let pushToTalkMode = false;

            socket = io();
            
            const statusDiv = document.getElementById('status');
            const toggleBtn = document.getElementById('toggleListening');
            const pushBtn = document.getElementById('pushToTalk');
            
            socket.on('connect', () => {
                console.log('Connected to agent');
                updateStatus('Connected! Ready to help you! 🤖');
            });

            socket.on('status_update', (data) => {
                updateStatus(data.message, data.type);
            });

            socket.on('response_text', (data) => {
                updateStatus('🤖 Agent: ' + data, 'response');
            });

            socket.on('response_audio', (data) => {
                playAudio(data);
            });

            toggleBtn.onclick = async () => {
                if (!continuousMode) {
                    await startContinuousListening();
                } else {
                    stopListening();
                }
            };

            pushBtn.onmousedown = async () => {
                if (!pushToTalkMode) {
                    pushToTalkMode = true;
                    await startListening();
                }
            };

            pushBtn.onmouseup = () => {
                if (pushToTalkMode) {
                    pushToTalkMode = false;
                    stopListening();
                }
            };

            document.getElementById('sendButton').onclick = () => {
                const text = document.getElementById('textInput').value.trim();
                if (text) {
                    socket.emit('text_query', { text: text });
                    document.getElementById('textInput').value = '';
                    updateStatus('Processing: ' + text, 'processing');
                }
            };

            document.getElementById('textInput').onkeypress = (e) => {
                if (e.key === 'Enter') {
                    document.getElementById('sendButton').click();
                }
            };

            async function startContinuousListening() {
                continuousMode = true;
                toggleBtn.textContent = 'Stop Listening';
                toggleBtn.classList.add('active');
                await startListening();
                updateStatus('👂 Continuous listening active - speak anytime!', 'listening');
            }

            async function startListening() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: { 
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        } 
                    });
                    
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm;codecs=opus'
                    });
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            const reader = new FileReader();
                            reader.onload = () => {
                                socket.emit('audio_chunk', {
                                    data: reader.result,
                                    continuous: continuousMode
                                });
                            };
                            reader.readAsArrayBuffer(event.data);
                        }
                    };
                    
                    mediaRecorder.start(250);
                    isListening = true;
                    
                    if (!continuousMode) {
                        updateStatus('🎤 Push-to-talk active - keep holding!', 'listening');
                    }
                    
                } catch (error) {
                    console.error('Error starting audio:', error);
                    updateStatus('❌ Microphone access denied', 'error');
                }
            }

            function stopListening() {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    mediaRecorder.stream.getTracks().forEach(track => track.stop());
                }
                
                isListening = false;
                
                if (continuousMode) {
                    continuousMode = false;
                    toggleBtn.textContent = 'Start Continuous Listening';
                    toggleBtn.classList.remove('active');
                    updateStatus('🔇 Stopped listening', 'ready');
                }
            }

            function updateStatus(message, type = 'ready') {
                statusDiv.textContent = message;
                statusDiv.className = 'status ' + type;
            }

            function playAudio(data) {
                const audioBlob = new Blob([data], { type: 'audio/mpeg' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.onloadeddata = () => {
                    audio.play().catch(e => console.error('Audio play failed:', e));
                };
            }
        </script>
    </body>
    </html>
    """)

@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    user_states[session_id] = RealTimeState()
    print(f'Client connected: {session_id}')
    emit('status_update', {'message': 'Connected! Ready to help!', 'type': 'ready'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    session_id = request.sid
    if session_id not in user_states:
        return
    
    state = user_states[session_id]
    audio_data = data['data']
    is_continuous = data.get('continuous', False)
    
    state.audio_buffer.append(audio_data)
    
    if is_continuous:
        if not state.speech_detected:
            state.speech_detected = True
            emit('status_update', {'message': '👂 Listening for your question...', 'type': 'listening'})
        
        state.silence_count = 0
        
        def check_silence():
            time.sleep(1.5)
            if state.silence_count >= 6:
                if state.speech_detected and not state.processing:
                    process_buffered_audio(session_id)
        
        state.silence_count += 1
        threading.Thread(target=check_silence, daemon=True).start()
    
    else:
        if len(state.audio_buffer) > 2:
            process_buffered_audio(session_id)

def process_buffered_audio(session_id):
    if session_id not in user_states:
        return
    
    state = user_states[session_id]
    if state.processing:
        return
    
    state.processing = True
    emit('status_update', {'message': '🔄 Processing your question...', 'type': 'processing'}, room=session_id)
    
    try:
        combined_audio = b''.join(state.audio_buffer)
        
        if len(combined_audio) < 1000:
            emit('status_update', {'message': '👂 Ready - please speak longer', 'type': 'ready'}, room=session_id)
            state.processing = False
            return
        
        deepgram_url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        params = {
            "model": "nova-2",
            "language": "en-US",
            "punctuate": "true"
        }
        
        response = requests.post(deepgram_url, headers=headers, params=params, data=combined_audio, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'results' in result and 'channels' in result['results']:
                channel = result['results']['channels'][0]
                if 'alternatives' in channel and len(channel['alternatives']) > 0:
                    transcript = channel['alternatives'][0]['transcript'].strip()
                    
                    if transcript:
                        print(f"User said: {transcript}")
                        emit('status_update', {'message': f'🎯 You said: "{transcript}"', 'type': 'processing'}, room=session_id)
                        
                        threading.Thread(target=process_user_query_async, args=(transcript, session_id), daemon=True).start()
                    else:
                        emit('status_update', {'message': '👂 Ready - I didn\'t catch that', 'type': 'ready'}, room=session_id)
                        state.processing = False
        else:
            emit('status_update', {'message': '❌ Speech recognition error', 'type': 'error'}, room=session_id)
            state.processing = False
            
    except Exception as e:
        print(f"Error processing audio: {e}")
        emit('status_update', {'message': '❌ Audio processing error', 'type': 'error'}, room=session_id)
        state.processing = False
    
    state.audio_buffer.clear()
    state.speech_detected = False
    state.silence_count = 0

def process_user_query_async(user_text, session_id):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response_text = str(loop.run_until_complete(query_engine.aquery(user_text)))
            print(f"Agent response: {response_text}")
            
            socketio.emit('response_text', response_text, room=session_id)
            
            tts_url = "https://api.cartesia.ai/tts/bytes"
            headers = {
                "Cartesia-Version": "2024-06-10",
                "X-API-Key": CARTESIA_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model_id": "sonic-english",
                "transcript": response_text,
                "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                "output_format": {"container": "mp3", "encoding": "mp3", "sample_rate": 44100}
            }
            
            tts_response = requests.post(tts_url, headers=headers, json=payload)
            if tts_response.status_code == 200:
                socketio.emit('response_audio', tts_response.content, room=session_id)
                print("✅ TTS audio sent")
                
            socketio.emit('status_update', {'message': '👂 Ready for your next question!', 'type': 'ready'}, room=session_id)
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            socketio.emit('response_text', "I encountered an error processing your question.", room=session_id)
        finally:
            loop.close()
            if session_id in user_states:
                user_states[session_id].processing = False
            
    except Exception as e:
        print(f"Error processing query: {e}")
        socketio.emit('response_text', "Sorry, I'm having trouble right now.", room=session_id)
        if session_id in user_states:
            user_states[session_id].processing = False

@socketio.on('text_query')
def handle_text_query(data):
    session_id = request.sid
    user_text = data.get('text', '').strip()
    if not user_text:
        emit('response_text', "Please enter a question.")
        return
        
    print(f"User asked (text): {user_text}")
    threading.Thread(target=process_user_query_async, args=(user_text, session_id), daemon=True).start()

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    if session_id in user_states:
        del user_states[session_id]
    print(f'Client disconnected: {session_id}')

if __name__ == '__main__':
    print("🎙️ Starting Real-time Voice RAG Agent...")
    print("🚀 Features: Continuous listening, Push-to-talk, Real-time processing")
    print("💰 Using: Gemini (Free) + Gemini Embeddings (Free) + Deepgram + Cartesia")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)