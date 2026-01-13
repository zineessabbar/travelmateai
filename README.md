# TravelMate AI - Voice Travel Assistant

A voice-powered AI travel assistant that listens to your questions, understands your preferences, and speaks back personalized travel recommendations.



## Features

- **Voice Interface**: Speak naturally and get spoken responses
- **Real-time Speech Recognition**: AssemblyAI Universal-1 streaming
- **AI-Powered Recommendations**: Google Gemini 2.5 Flash
- **Semantic Search**: Pinecone vector database for smart destination matching
- **Natural Voice Output**: ElevenLabs Flash v2.5 text-to-speech
- **Interruption Support**: Cut off the AI mid-sentence to ask follow-ups
- **Auto-Reconnect**: Handles connection drops gracefully
- **Proactive Responses**: Gives suggestions first, asks questions second

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Gemini text-embedding-004 (768 dim) |
| Vector DB | Pinecone Serverless |
| Speech-to-Text | AssemblyAI Universal-1 (V3 Streaming) |
| Text-to-Speech | ElevenLabs Flash v2.5 |
| Audio Playback | FFmpeg (ffplay) |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (Required for Audio)

**Windows:**
```bash
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### 3. Configure API Keys

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
ASSEMBLYAI_API_KEY=your_assemblyai_key
ELEVENLABS_API_KEY=your_elevenlabs_key

INDEX_NAME=travelmate-gemini
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### 4. Get API Keys

| Service | URL | Free Tier |
|---------|-----|-----------|
| Google Gemini | https://aistudio.google.com/app/apikey | Yes |
| Pinecone | https://app.pinecone.io/ | Yes |
| AssemblyAI | https://www.assemblyai.com/ | Limited |
| ElevenLabs | https://elevenlabs.io/ | Limited |

### 5. Setup Database (First Time Only)

```bash
python setup_pinecone_gemini.py
```

This creates the vector database with:
- 10 destinations (Paris, Bali, Tokyo, Marrakech, etc.)
- 6+ attractions with prices and hours
- Transport info for major cities
- Sample user profiles

### 6. Run the App

```bash
python app_full_gemini.py
```

## Usage

1. **Start the app** - You'll hear a greeting
2. **Say a destination** - "I'm going to Paris"
3. **Get instant suggestions** - AI gives 2-3 ideas immediately
4. **Ask for more** - "Give me a full itinerary"
5. **Interrupt anytime** - Just start talking to stop the AI
6. **Exit** - Say "exit", "quit", or press Ctrl+C

## Example Conversation

```
You: "I'm going to Paris"
AI: "Paris is magical! Try a morning at the Louvre, afternoon
     stroll along the Seine, and evening at a cozy Montmartre
     cafe. Want a day-by-day plan?"

You: "Yes"
AI: "Day 1: Start at the Eiffel Tower for sunrise, then..."
```

## Project Structure

```
travelmate-ai-gemini_V1.0.1/
├── app_full_gemini.py       # Main voice assistant app
├── setup_pinecone_gemini.py # Database setup script
├── requirements.txt         # Python dependencies
├── .env                     # Your API keys (gitignored)
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Key Features Explained

### Proactive AI Responses
The AI gives suggestions first, questions second:
- **Good**: "London is great! Try Notting Hill, Thames walks, Borough Market. How many days?"
- **Bad**: "How many days? Budget? Who's traveling? What do you like?"

### Auto-Reconnect
If the microphone connection drops (WebSocket timeout), the app automatically reconnects with exponential backoff.

### Deduplication
Prevents the same phrase from being processed multiple times (common with short words like "yes").

### Rate Limit Handling
If Gemini rate limits are hit, the app waits automatically and retries.

## Troubleshooting

### "FFmpeg is not installed"
Install FFmpeg using the commands in the Quick Start section.

### "API key not valid"
- Check your `.env` file has the correct keys
- Verify the keys are active in each service's dashboard
- Gemini keys expire - regenerate if needed

### "Rate limit exceeded"
- Gemini free tier: ~20 requests/day for 2.5 Flash
- Wait for quota reset or use a different model
- Consider `gemini-1.5-flash` for higher limits

### App doesn't respond to voice
- Check your microphone is working
- Ensure AssemblyAI API key is valid
- Try running in a quiet environment

### Connection keeps dropping
- The auto-reconnect will handle this
- Check your internet connection
- AssemblyAI WebSocket timeout is normal after ~30s of silence

## Dependencies

- Python 3.8+
- google-generativeai >= 0.8.3
- pinecone >= 5.0.0
- assemblyai >= 0.33.0
- elevenlabs >= 1.0.0
- python-dotenv >= 1.0.0
- numpy >= 1.26.0

## License

UIR - Student Project

## Version

v1.1.1 - Final Stable Release
