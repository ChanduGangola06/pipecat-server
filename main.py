import asyncio
import json
import base64
import logging
import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    LLMMessagesFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.vad.silero import SileroVADAnalyzer

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Voice Agent Server", version="1.0.0")

# Add CORS middleware for Next.js 15
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",  # For HTTPS in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceAgentSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.pipeline: Optional[Pipeline] = None
        self.runner: Optional[PipelineRunner] = None
        self.task: Optional[PipelineTask] = None
        self.session_id = id(self)
        
        # Initialize services with error handling
        try:
            self.stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                model="nova-2",
                language="en-US",
            )
            
            self.llm = OpenAILLMService(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-3.5-turbo",
                temperature=0.7,
            )
            
            self.tts = CartesiaTTSService(
                api_key=os.getenv("CARTESIA_API_KEY"),
                voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
                sample_rate=16000,
            )
            
            # Optional: VAD for better voice activity detection
            self.vad = SileroVADAnalyzer()
            
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            raise

    async def setup_pipeline(self):
        """Setup the Pipecat pipeline with proper error handling"""
        try:
            # Create aggregators for handling responses
            user_response = LLMUserResponseAggregator()
            assistant_response = LLMAssistantResponseAggregator()
            
            # Create the processing pipeline
            self.pipeline = Pipeline([
                self.stt,           # Speech-to-Text
                user_response,      # Aggregate user input
                self.llm,          # Process with LLM
                self.tts,          # Text-to-Speech
                assistant_response, # Aggregate assistant response
            ])
            
            # Set up event handlers
            user_response.add_event_handler(
                "on_response_complete", 
                self.on_user_transcript
            )
            assistant_response.add_event_handler(
                "on_response_complete", 
                self.on_assistant_response
            )
            self.tts.add_event_handler("on_tts_frame", self.on_audio_frame)
            
            # Initialize the task
            self.task = PipelineTask(self.pipeline)
            
            # Add system message
            system_message = LLMMessagesFrame([
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI voice assistant. "
                        "Keep your responses conversational, concise, and engaging. "
                        "Since this is a voice conversation, avoid using markdown, "
                        "lists, or complex formatting. Speak naturally as if talking to a friend."
                    )
                }
            ])
            await self.task.queue_frame(system_message)
            
            # Create runner
            self.runner = PipelineRunner()
            
            logger.info(f"Pipeline setup complete for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error setting up pipeline for session {self.session_id}: {e}")
            await self.send_error(f"Pipeline setup failed: {str(e)}")
            raise

    async def send_error(self, message: str):
        """Send error message to client"""
        try:
            await self.websocket.send_text(json.dumps({
                "type": "error",
                "message": message
            }))
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")

    async def send_message(self, message_type: str, **kwargs):
        """Send structured message to client"""
        try:
            message = {"type": message_type, **kwargs}
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message {message_type}: {e}")

    async def on_user_transcript(self, aggregator, frame: TextFrame):
        """Handle user speech transcript"""
        logger.info(f"User transcript: {frame.text}")
        await self.send_message("transcript", text=frame.text)

    async def on_assistant_response(self, aggregator, frame: TextFrame):
        """Handle assistant text response"""
        logger.info(f"Assistant response: {frame.text}")
        await self.send_message("response", text=frame.text)

    async def on_audio_frame(self, frame: AudioRawFrame):
        """Handle TTS audio output"""
        try:
            # Send speaking start notification
            await self.send_message("speaking_start")
            
            # Convert audio to base64 for WebSocket transmission
            audio_data = base64.b64encode(frame.audio).decode('utf-8')
            await self.send_message(
                "audio",
                audio=audio_data,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels
            )
            
            # Send speaking end notification after a short delay
            await asyncio.sleep(0.1)  # Small delay to ensure audio is processed
            await self.send_message("speaking_end")
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            await self.send_error("Audio processing failed")

    async def process_audio_message(self, message: Dict[str, Any]):
        """Process incoming audio data from client"""
        try:
            if "data" not in message:
                await self.send_error("No audio data provided")
                return

            # Decode base64 audio data
            audio_data = base64.b64decode(message["data"])
            
            # Create audio frame with proper parameters
            audio_frame = AudioRawFrame(
                audio=audio_data,
                sample_rate=16000,  # Standard rate for speech
                num_channels=1
            )
            
            # Queue frame for processing
            if self.task:
                await self.task.queue_frame(audio_frame)
            else:
                await self.send_error("Pipeline not initialized")
                
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            await self.send_error(f"Audio processing failed: {str(e)}")

    async def start_pipeline(self):
        """Start the pipeline processing"""
        if not self.runner or not self.task:
            raise RuntimeError("Pipeline not properly initialized")
        
        try:
            logger.info(f"Starting pipeline for session {self.session_id}")
            await self.runner.run(self.task)
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            await self.send_error("Pipeline execution failed")

    async def stop_pipeline(self):
        """Stop the pipeline and clean up resources"""
        try:
            if self.task:
                await self.task.stop()
                logger.info(f"Pipeline stopped for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")


# Global session management
active_sessions: Dict[int, VoiceAgentSession] = {}

@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    session_id = None
    session = None
    
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        # Create new session
        session = VoiceAgentSession(websocket)
        session_id = session.session_id
        active_sessions[session_id] = session
        
        # Setup pipeline
        await session.setup_pipeline()
        
        # Send ready signal
        await session.send_message("ready", message="Voice agent ready")
        
        # Start pipeline in background
        pipeline_task = asyncio.create_task(session.start_pipeline())
        
        # Handle incoming messages
        async for data in websocket.iter_text():
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "audio":
                    await session.process_audio_message(message)
                elif message_type == "stop":
                    logger.info("Received stop signal from client")
                    break
                elif message_type == "ping":
                    await session.send_message("pong")
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await session.send_error("Invalid message format")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await session.send_error("Message processing failed")
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected (session {session_id})")
    except Exception as e:
        logger.error(f"WebSocket error (session {session_id}): {e}")
        if session:
            await session.send_error(f"Connection error: {str(e)}")
    finally:
        # Cleanup
        if session:
            await session.stop_pipeline()
        if session_id and session_id in active_sessions:
            del active_sessions[session_id]
        if 'pipeline_task' in locals():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Session {session_id} cleaned up")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "services": {
            "deepgram": bool(os.getenv("DEEPGRAM_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "cartesia": bool(os.getenv("CARTESIA_API_KEY")),
        }
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "active_sessions": len(active_sessions),
        "session_ids": list(active_sessions.keys()),
    }

if __name__ == "__main__":
    # Validate environment variables
    required_env_vars = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        exit(1)
    
    logger.info("Starting AI Voice Agent Server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
        access_log=True,
    )