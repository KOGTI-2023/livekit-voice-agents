"""
LiveKit Voice Agent - Quick Start
==================================
The simplest possible LiveKit voice agent to get you started.
Requires only OpenAI and Deepgram API keys.
"""

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, deepgram, silero
from datetime import datetime
import os

# Load environment variables
load_dotenv(".env")

class Assistant(Agent):

    def __init__(self):
        super().__init__(
            instructions="""Du bist ein freundlicher und hilfreicher Assistent."""
        )


async def entrypoint(ctx: agents.JobContext):
    """Entry point for the agent."""

    # Configure the voice pipeline with the essentials
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            model="gpt-realtime-mini", # oder "gpt-realtime" für das bessere Modell
            voice="echo", # oder "marin", "cedar" (neue Stimmen)
            temperature=0.8,
        )
    )
    
    # Start the session
    await session.start(
        room=ctx.room,
        agent=Assistant()
    )

if __name__ == "__main__":
    # Run the agent
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
