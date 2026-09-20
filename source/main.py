import asyncio
import logging
import os
import tempfile

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from groq import Groq


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
STT_TOKEN = os.getenv("STT_TOKEN")

if not API_TOKEN:
    raise RuntimeError("API_TOKEN is not set")

if not STT_TOKEN:
    raise RuntimeError("STT_TOKEN is not set")

GROQ_MODEL = "whisper-large-v3-turbo"


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO
)


# ============================================================
# Telegram
# ============================================================

bot = Bot(token=API_TOKEN)

dp = Dispatcher()
router = Router()

dp.include_router(router)


# ============================================================
# Groq
# ============================================================

groq_client = Groq(
    api_key=STT_TOKEN
)


# ============================================================
# Local Whisper
# ============================================================

logging.info("Loading local Whisper model...")

local_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

logging.info("Local Whisper model loaded")


# ============================================================
# Commands
# ============================================================

@router.message(Command("start"))
async def send_welcome(message: Message):
    await message.reply("Hi, I'm AOS!")


@router.message(Command("help"))
async def send_help(message: Message):
    await message.reply("No.")


# ============================================================
# Groq transcription
# ============================================================

def transcribe_with_groq(
    file_path: str
) -> str:

    with open(file_path, "rb") as audio_file:

        transcription = groq_client.audio.transcriptions.create(
            file=audio_file,
            model=GROQ_MODEL,
            response_format="json",
            temperature=0
        )

    return transcription.text.strip()


# ============================================================
# Local Whisper transcription
# ============================================================

def transcribe_with_local_whisper(
    file_path: str
) -> str:

    segments, info = local_model.transcribe(
        file_path,
        beam_size=5
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text.strip()


# ============================================================
# Speech-to-text with fallback
# ============================================================

async def transcribe_audio(
    file_path: str
) -> str:

    try:

        logging.info(
            "Trying Groq transcription..."
        )

        text = await asyncio.to_thread(
            transcribe_with_groq,
            file_path
        )

        logging.info(
            "Groq transcription successful"
        )

        return text

    except Exception:

        logging.exception(
            "Groq transcription failed, "
            "falling back to local Whisper"
        )

    logging.info(
        "Using local Whisper fallback..."
    )

    text = await asyncio.to_thread(
        transcribe_with_local_whisper,
        file_path
    )

    logging.info(
        "Local Whisper transcription successful"
    )

    return text


# ============================================================
# Extract audio from video
# ============================================================

async def extract_audio(
    video_path: str,
    audio_path: str
):

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        audio_path,

        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE
    )

    _, stderr = await process.communicate()

    if process.returncode != 0:

        logging.error(
            "FFmpeg error:\n%s",
            stderr.decode(
                errors="replace"
            )
        )

        raise RuntimeError(
            "FFmpeg failed to extract audio"
        )


# ============================================================
# Voice
# ============================================================

@router.message(F.voice)
async def process_audio(message: Message):

    await transcribe_message(
        message,
        message.voice.file_id,
        ".ogg",
        is_video=False
    )


# ============================================================
# Video circle
# ============================================================

@router.message(F.video_note)
async def process_video(message: Message):

    await transcribe_message(
        message,
        message.video_note.file_id,
        ".mp4",
        is_video=True
    )


# ============================================================
# Common media processing
# ============================================================

async def transcribe_message(
    message: Message,
    file_id: str,
    extension: str,
    is_video: bool
):

    status_message = await message.reply(
        "🎧 Transcribing..."
    )

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            input_path = os.path.join(
                temp_dir,
                f"input{extension}"
            )

            # ------------------------------------------------
            # Download Telegram file
            # ------------------------------------------------

            telegram_file = await bot.get_file(
                file_id
            )

            await bot.download_file(
                telegram_file.file_path,
                destination=input_path
            )

            # ------------------------------------------------
            # Extract audio from video
            # ------------------------------------------------

            if is_video:

                audio_path = os.path.join(
                    temp_dir,
                    "audio.wav"
                )

                logging.info(
                    "Extracting audio from video..."
                )

                await extract_audio(
                    input_path,
                    audio_path
                )

            else:

                audio_path = input_path

            # ------------------------------------------------
            # Speech-to-text
            # ------------------------------------------------

            text = await transcribe_audio(
                audio_path
            )

        if not text:

            text = "I couldn't recognize any speech."

        await status_message.edit_text(
            f"📝 {text}"
        )

    except Exception:

        logging.exception(
            "Failed to transcribe media"
        )

        await status_message.edit_text(
            "❌ Couldn't transcribe this message."
        )


# ============================================================
# Main
# ============================================================

async def main():

    await bot.delete_webhook(
        drop_pending_updates=True
    )

    await dp.start_polling(
        bot
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )