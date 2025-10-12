# livekit_agent_telephony_dsgvo.py
# livekit_agent_telephony_dsgvo.py
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import azure, openai, silero
import httpx
import os

load_dotenv(".env")

class TelephonyAssistant(Agent):
    """DSGVO-konformer deutscher Voice Assistant für Telefonie."""

    def __init__(self):
        super().__init__(
            instructions="""Du bist ein hilfreicher deutscher Assistent.
            Sprich klar und deutlich für Telefongespräche.
            Halte Antworten präzise und nicht zu lang.
            Sei freundlich und professionell."""
        )

    @function_tool
    async def send_to_webhook(
        self, 
        context: RunContext, 
        message: str
    ) -> str:
        """Sendet eine Nachricht oder Frage an einen externen Webhook.
        
        Args:
            message: Die Nachricht oder Frage die gesendet werden soll
        """
        webhook_url = os.getenv("N8N_WEBHOOK_URL")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    webhook_url,
                    json={"message": message}
                )
                
                if response.status_code == 200:
                    return response.text
                else:
                    return f"Fehler: Status {response.status_code}"
                    
        except httpx.TimeoutException:
            return "Webhook Timeout nach 30 Sekunden"
        except Exception as e:
            return f"Fehler: {str(e)}"

    async def on_enter(self):
        """Wird aufgerufen wenn Agent aktiv wird."""
        participant_metadata = self.session.room.local_participant.metadata
        
        if "outbound" not in participant_metadata:
            await self.session.generate_reply(
                instructions="Begrüße den Anrufer freundlich auf Deutsch."
            )


async def entrypoint(ctx: agents.JobContext):
    """Entry point für den Agent - 100% DSGVO-konform."""
    
    session = AgentSession(
        # ✅ Azure Speech STT
        stt=azure.STT(
            speech_key=os.getenv("AZURE_SPEECH_KEY"),
            speech_region=os.getenv("AZURE_SPEECH_REGION"),
            language="de-DE",
        ),
        
        # ✅ Azure OpenAI LLM - KORRIGIERT mit .with_azure()
        llm=openai.LLM.with_azure(
            model="gpt-4o",  # Oder "gpt-4o-mini" je nach deinem Deployment
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        ),
        
        # ✅ Azure Speech TTS
        tts=azure.TTS(
            speech_key=os.getenv("AZURE_SPEECH_KEY"),
            speech_region=os.getenv("AZURE_SPEECH_REGION"),
            voice="de-DE-KatjaNeural",
        ),
        
        vad=silero.VAD.load(),
        turn_detection="semantic",
    )

    await ctx.connect()
    await session.start(room=ctx.room, agent=TelephonyAssistant())


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint
    ))