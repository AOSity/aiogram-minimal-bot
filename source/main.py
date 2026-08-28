import asyncio
import logging
import os
import tempfile

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from faster_whisper import WhisperModel


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise RuntimeError("API_TOKEN is not set")


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
# Whisper
# ============================================================

logging.info("Loading Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

logging.info("Whisper model loaded")


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
# Speech-to-text
# ============================================================

def transcribe_audio(file_path: str) -> str:

    segments, info = model.transcribe(
        file_path,
        beam_size=5
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text.strip()


# ============================================================
# Voice
# ============================================================

@router.message(F.voice)
async def process_audio(message: Message):

    file_id = message.voice.file_id
    extension = ".ogg"

    await transcribe_message(
        message,
        file_id,
        extension,
        extract_audio=False
    )


# ============================================================
# Video circle
# ============================================================

@router.message(F.video_note)
async def process_video(message: Message):

    if message.video:
        file_id = message.video.file_id
    else:
        file_id = message.video_note.file_id

    await transcribe_message(
        message,
        file_id,
        ".mp4",
        extract_audio=True
    )


# ============================================================
# Common media processing
# ============================================================

async def transcribe_message(
    message: Message,
    file_id: str,
    extension: str,
    extract_audio: bool
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

            # Get Telegram file information
            telegram_file = await bot.get_file(
                file_id
            )

            # Download it
            await bot.download_file(
                telegram_file.file_path,
                destination=input_path
            )

            audio_path = input_path

            # ------------------------------------------------
            # Extract audio from video
            # ------------------------------------------------

            if extract_audio:

                audio_path = os.path.join(
                    temp_dir,
                    "audio.wav"
                )

                process = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    audio_path,

                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )

                await process.communicate()

                if process.returncode != 0:
                    raise RuntimeError(
                        "FFmpeg failed to extract audio"
                    )

            # ------------------------------------------------
            # Whisper
            # ------------------------------------------------

            text = await asyncio.to_thread(
                transcribe_audio,
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

    # Remove pending updates from before startup
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