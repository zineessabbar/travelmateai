"""
TravelMate AI - Ultimate Edition (Final Stable Version)
Integrated Fixes:
- AssemblyAI V3 Streaming Client (Universal-1)
- ElevenLabs Flash v2.5 (Low Latency)
- Gemini 2.5 Flash (Optimized)
- Pinecone Persistence
- Full Interruption Capabilities
- Static Analysis Fixes (FFmpeg check, exception handling)
- Auto-reconnect on WebSocket timeout (with exponential backoff)
"""

import os
import sys
import re
import signal
import threading
import json
import time
import subprocess
import shutil
import statistics
from functools import lru_cache
from datetime import datetime
from dotenv import load_dotenv

# Instant exit on Ctrl+C (Windows fix)
def force_exit(signum, frame):
    print("\n\n👋 Goodbye!")
    os._exit(0)

signal.signal(signal.SIGINT, force_exit)
if sys.platform == 'win32':
    signal.signal(signal.SIGBREAK, force_exit)

# --- AI & Cloud Imports ---
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    TurnEvent,
    StreamingParameters,
    StreamingError
)
from elevenlabs.client import ElevenLabs
import google.generativeai as genai
from pinecone import Pinecone

# Fix Windows emoji encoding issues in terminal
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Check for FFmpeg at startup
if not shutil.which("ffplay"):
    print("❌ Critical Error: FFmpeg is not installed or not in PATH.")
    print("   Please install FFmpeg to play audio (Windows: winget install ffmpeg)")
    sys.exit(1)

# -------------------------------------------------------------------------
# AUDIO PLAYER CLASS (FFmpeg Streaming)
# -------------------------------------------------------------------------
class InterruptiblePlayer:
    def __init__(self):
        self.process = None
        self._stop_event = threading.Event()
        self.is_playing = False

    def play_stream(self, audio_generator):
        """Plays audio stream using ffplay (requires FFmpeg installed)."""
        self._stop_event.clear()
        self.is_playing = True
        
        # Start ffplay process: -nodisp (no window), -autoexit (close when done)
        command = ["ffplay", "-autoexit", "-nodisp", "-i", "pipe:0"]
        
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            for chunk in audio_generator:
                if self._stop_event.is_set():
                    break
                if chunk and self.process and self.process.stdin:
                    try:
                        self.process.stdin.write(chunk)
                        self.process.stdin.flush()
                    except BrokenPipeError:
                        # Process was killed or finished
                        break 
                        
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            self.stop()

    def stop(self):
        """Immediately stops playback."""
        self._stop_event.set()
        self.is_playing = False
        if self.process:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass # Process already dead
            except Exception as e:
                print(f"Error stopping player: {e}")
            self.process = None

# -------------------------------------------------------------------------
# MAIN INTELLIGENCE CLASS
# -------------------------------------------------------------------------
class TravelMate_Gemini:
    def __init__(self):
        self.check_api_keys()

        # 1. Initialize Google Gemini (The Brain)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.chat_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 2. Initialize ElevenLabs (The Voice)
        self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        
        # 3. Initialize Pinecone (The Memory)
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone_client.Index(os.getenv("INDEX_NAME", "travelmate-gemini"))
        
        # 4. Initialize Audio Components
        self.player = InterruptiblePlayer()
        self.stream_client = None
        self.stream_thread = None
        self.is_listening = False
        self.is_interrupted = False

        # 5. Deduplication (prevent processing same input twice)
        self.last_processed_text = ""
        self.last_processed_time = 0

        # 6. User Context
        self.user_id = "user_001"
        self.conversation_history = []
        self.user_context = self.load_user_context()

        # 7. Analytics/Metrics
        self.metrics = {
            "start_time": time.time(),
            "interactions": 0,
            "interruptions": 0,
            "response_latencies": []
        }

    def check_api_keys(self):
        """Ensures all secrets are present."""
        required = ["ASSEMBLYAI_API_KEY", "GEMINI_API_KEY", "ELEVENLABS_API_KEY", "PINECONE_API_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            print(f"❌ Critical Error: Missing keys in .env: {missing}")
            sys.exit(1)

    # --- MEMORY (RAG) FUNCTIONS ---
    
    @lru_cache(maxsize=128)
    def create_embedding_cached(self, text, task_type):
        """Cached embeddings to save time/cost."""
        try:
            result = genai.embed_content(
                model='models/text-embedding-004', content=text, task_type=task_type
            )
            return tuple(result['embedding']) 
        except Exception as e:
            print(f"Embedding Error: {e}")
            return None

    def create_embedding(self, text, task_type="retrieval_document"):
        return self.create_embedding_cached(text, task_type)

    def load_user_context(self):
        """Fetch user profile from Pinecone."""
        try:
            res = self.index.fetch(ids=[f"user_{self.user_id}_profile"])
            if res and res.vectors:
                return res.vectors[f"user_{self.user_id}_profile"].metadata
            return {"user_id": self.user_id, "travel_style": [], "budget_range": []}
        except Exception as e:
            print(f"Context Load Error: {e}")
            return {"user_id": self.user_id}

    def save_user_context(self):
        """Save updated profile to Pinecone."""
        try:
            emb = self.create_embedding(json.dumps(self.user_context))
            if emb:
                self.index.upsert(vectors=[{
                    "id": f"user_{self.user_id}_profile", 
                    "values": emb, 
                    "metadata": {**self.user_context, "type": "profile", "updated": datetime.now().isoformat()}
                }])
                print("✓ Profile saved to Memory")
        except Exception as e:
             print(f"Context Save Error: {e}")

    # --- BRAIN (GENERATE RESPONSE) ---

    # Smart defaults for trip parameters
    DEFAULT_TRIP_PARAMS = {
        'duration': '3-5 days',
        'pace': 'moderate',
        'travelers': 'flexible',
        'budget': 'mid-range'
    }

    # System prompt that prioritizes suggestions over questions
    SYSTEM_PROMPT = """You are TravelMate, a proactive travel assistant.

CRITICAL RULES:
1. When a user mentions a destination, IMMEDIATELY give 2-3 specific suggestions
2. Only ask ONE clarifying question AFTER providing value first
3. Keep responses under 3 sentences unless giving a full itinerary
4. Be proactive and helpful, NOT interrogative
5. Use smart defaults: assume 3-5 days, moderate pace, mid-range budget unless told otherwise

RESPONSE PATTERN:
- User mentions destination → Give quick suggestions + max 1 optional question
- User says "yes" or agrees → Provide more details or next steps
- User asks question → Answer directly, don't ask back
- User gives partial info → Use defaults for missing details, proceed with suggestions

GOOD EXAMPLE:
User: "I'm going to London"
You: "London is amazing! For a relaxed vibe, try: cozy cafes in Notting Hill, a Thames riverside walk, and browsing Borough Market. Want me to build a day-by-day itinerary?"

BAD EXAMPLE (too many questions):
User: "I'm going to London"
You: "Great! How many days? What's your budget? Who are you traveling with? What activities do you like?"

Remember: GIVE VALUE FIRST, ask questions second (if at all)."""

    def generate_response(self, user_input):
        """Generates AI response using RAG."""
        start_time = time.time()

        # RAG: Search for relevant destinations (expanded keywords)
        destination_keywords = ["recommend", "where", "go", "trip", "visit", "going", "travel",
                                "london", "paris", "tokyo", "bali", "morocco", "dubai", "lisbon",
                                "istanbul", "santorini", "zanzibar", "marrakech", "casablanca"]
        context_str = f"Current User Profile: {self.user_context}"

        if any(w in user_input.lower() for w in destination_keywords):
            emb = self.create_embedding(user_input, "retrieval_query")
            if emb:
                try:
                    res = self.index.query(vector=emb, top_k=3, include_metadata=True, filter={"type": "destination"})
                    matches = []
                    for m in res.matches:
                        name = m.metadata.get('name', '')
                        country = m.metadata.get('country', '')
                        description = m.metadata.get('description', '')[:100]
                        vibes = m.metadata.get('vibes', [])
                        matches.append(f"{name} ({country}): {description}... Vibes: {vibes}")
                    if matches:
                        context_str += f"\nRelevant Destinations from Database:\n" + "\n".join(matches)
                except Exception as e:
                    print(f"Search Error: {e}")

        # Build prompt with improved system instructions
        prompt = f"""{self.SYSTEM_PROMPT}

Context: {context_str}
Conversation History: {[x['content'] for x in self.conversation_history[-3:]]}
User: {user_input}

Remember: Give specific suggestions immediately. Be helpful, not interrogative."""
        
        try:
            # Disable thinking for lower latency (Student Project Optimization)
            response = self.chat_model.generate_content(prompt)
            text = response.text.strip()
        except Exception as e:
            error_str = str(e)
            # Handle rate limiting (429 error)
            if "429" in error_str or "quota" in error_str.lower():
                # Extract retry delay if available, default to 30s
                retry_delay = 30
                if "retry" in error_str.lower() and "seconds" in error_str.lower():
                    match = re.search(r'(\d+)\.?\d*\s*s', error_str)
                    if match:
                        retry_delay = int(float(match.group(1))) + 1
                print(f"⏳ Rate limit hit. Waiting {retry_delay}s...")
                time.sleep(retry_delay)
                # Retry once after waiting
                try:
                    response = self.chat_model.generate_content(prompt)
                    text = response.text.strip()
                except Exception as retry_e:
                    print(f"Gemini Retry Error: {retry_e}")
                    text = "I'm still rate limited. Please wait a moment and try again."
            else:
                print(f"Gemini Error: {e}")
                text = "I'm having a little trouble thinking right now. Could you ask that again?"
            
        latency = time.time() - start_time
        self.metrics["response_latencies"].append(latency)
        self.metrics["interactions"] += 1
        
        # Clear "Thinking..." line
        print(f"\r  ⚡ Latency: {latency:.2f}s" + " " * 20)
        return text

    # --- EARS (ASSEMBLYAI V3 STREAMING) ---

    def on_turn(self, client: StreamingClient, event: TurnEvent):
        """Handles incoming text from microphone."""
        
        # 1. Interruption Handling (Partial Results)
        if not event.end_of_turn:
            # If AI is talking and user speaks > 5 chars, STOP the AI
            if self.player.is_playing and len(event.transcript) > 5:
                print(f"\n🛑 Interruption detected: '{event.transcript}'")
                self.player.stop()
                self.is_interrupted = True
                self.metrics["interruptions"] += 1
            # Print live transcript
            print(f"  {event.transcript}", end="\r")
            
        # 2. Final Sentence Received
        elif event.end_of_turn and event.transcript.strip():
            if self.is_interrupted:
                self.is_interrupted = False # Reset and ignore the interrupted fragment
                return

            # 3. Deduplication: Skip if same text processed within 3 seconds
            current_time = time.time()
            normalized_text = event.transcript.strip().lower()
            if (normalized_text == self.last_processed_text.lower() and
                current_time - self.last_processed_time < 3.0):
                print(f"  (duplicate ignored)", end="\r")
                return

            # Update deduplication tracking
            self.last_processed_text = event.transcript.strip()
            self.last_processed_time = current_time

            print(f"\n👤 You: {event.transcript}")
            self.process_turn(event.transcript)

    def on_error(self, client: StreamingClient, error: StreamingError):
        # Only print if there's actual error content (avoid empty error spam)
        if error and str(error).strip():
            print(f"\n⚠️ Stream Error: {error}")

    def start_transcription(self):
        """Starts the microphone listener in a background thread."""
        if self.is_listening: return
        self.is_listening = True
        
        self.stream_thread = threading.Thread(target=self._run_stream, daemon=True)
        self.stream_thread.start()

    def _run_stream(self):
        """The actual threaded listener function with auto-reconnect."""
        reconnect_delay = 1  # Start with 1 second delay
        max_reconnect_delay = 30  # Max 30 seconds between retries

        while self.is_listening:
            print("🎤 Listening... (Universal-1)", flush=True)
            try:
                # Initialize V3 Client
                self.stream_client = StreamingClient(
                    StreamingClientOptions(
                        api_key=os.getenv("ASSEMBLYAI_API_KEY"),
                        api_host="streaming.assemblyai.com"
                    )
                )

                self.stream_client.on(StreamingEvents.Turn, self.on_turn)
                self.stream_client.on(StreamingEvents.Error, self.on_error)

                # CONNECT V3 (FIXED: Added StreamingParameters)
                self.stream_client.connect(
                    StreamingParameters(
                        sample_rate=16000,
                        word_boost=["TravelMate", "Gemini", "Pinecone"]
                    )
                )

                # Start streaming microphone audio (blocks until disconnected)
                self.stream_client.stream(
                    aai.extras.MicrophoneStream(sample_rate=16000)
                )

                # If stream() exits and we're still supposed to be listening,
                # it means the connection dropped - reconnect
                if self.is_listening:
                    print(f"\n🔄 Connection lost. Reconnecting in {reconnect_delay}s...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue

            except Exception as e:
                if not self.is_listening:
                    # Intentional stop, exit loop
                    break

                error_str = str(e).lower()
                # Check for timeout/connection errors that warrant reconnection
                if "timeout" in error_str or "closed" in error_str or "keepalive" in error_str or "connection" in error_str:
                    print(f"\n🔄 Connection lost. Reconnecting in {reconnect_delay}s...")
                    time.sleep(reconnect_delay)
                    # Exponential backoff
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                else:
                    print(f"❌ Listener Thread Error: {e}")
                    break
            finally:
                self.stream_client = None

    def stop_transcription(self):
        """Safely stops the listener."""
        self.is_listening = False
        if self.stream_client:
            try:
                # FIX: Use disconnect() not close()
                self.stream_client.disconnect()
            except Exception as e:
                # We are shutting down anyway, ignore connection errors
                pass
            self.stream_client = None

    # --- COORDINATOR ---

    def process_turn(self, user_text):
        """The main logic loop: Stop Mic -> Think -> Speak -> Start Mic"""
        
        # 1. Stop listening (so AI doesn't hear itself)
        self.stop_transcription()
        
        # 2. Check for exit command
        if any(w in user_text.lower() for w in ["exit", "quit", "disconnect", "bye", "shutdown"]):
            print("🤖 TravelMate: Goodbye!")
            self.save_user_context()
            os._exit(0) # Force Quit

        # 3. Generate Answer
        print("  🤔 Thinking...", end="\r", flush=True)
        ai_response = self.generate_response(user_text)
        print(f"🤖 TravelMate: {ai_response}")
        
        # 4. Save History
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # 5. Generate & Play Audio
        if self.player:
            try:
                # Use Flash V2.5 for Speed
                audio_stream = self.elevenlabs_client.text_to_speech.convert(
                    text=ai_response,
                    voice_id="21m00Tcm4TlvDq8ikWAM", # Default voice
                    model_id="eleven_flash_v2_5"     # FAST MODEL
                )
                
                # CRITICAL: Start listening *before* playing to allow interruption
                self.start_transcription() 
                
                self.player.play_stream(audio_stream)
                
            except Exception as e:
                print(f"Audio Generation Error: {e}")
                self.start_transcription() # Ensure we listen even if audio fails

# -------------------------------------------------------------------------
# APP ENTRY POINT
# -------------------------------------------------------------------------
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("╔════════════════════════════════════════════════════╗")
    print("║            🌍 TRAVELMATE AI - DASHBOARD            ║")
    print("║          (Student Project: BD&AI Module)           ║")
    print("╠════════════════════════════════════════════════════╣")
    print("║  🟢 STATUS: ONLINE                                 ║")
    print("║  🧠 BRAIN:  Gemini 2.5 Flash                       ║")
    print("║  🎤 EARS:   AssemblyAI Universal-1 (V3)            ║")
    print("║  🔊 VOICE:  ElevenLabs Flash v2.5                  ║")
    print("╚════════════════════════════════════════════════════╝")
    print("\nInitializing System...")
    
    app = TravelMate_Gemini()
    
    # Initial Greeting (proactive, value-first)
    try:
        greeting = "Hey! I'm TravelMate, ready to plan your next adventure. Just tell me where you're headed, and I'll give you instant suggestions!"
        print(f"\n🤖 TravelMate: {greeting}")
        
        # Generate Audio
        audio = app.elevenlabs_client.text_to_speech.convert(
            text=greeting, 
            voice_id="21m00Tcm4TlvDq8ikWAM", 
            model_id="eleven_flash_v2_5"
        )
        
        # Start Listen + Play
        app.start_transcription()
        app.player.play_stream(audio)

        # Keep Main Thread Alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Shutting down...")
        # Immediately force exit - don't wait for anything
        os._exit(0)

if __name__ == "__main__":
    main()