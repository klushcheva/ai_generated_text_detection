import asyncio
import os
import logging
from typing import Dict, Any
import httpx  # Replace requests

from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    ContextTypes,
    CommandHandler
)
from redis import Redis

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
TELEGRAM_TOKEN = "7576648646:AAFxMOBoSotMaZPHdHu1bLfCA0ZnDk4cFyA"
FASTAPI_URL = "http://model:8000"

# Initialize Redis (critical fix)
redis_conn = Redis(
    host=REDIS_HOST,
    port=6379,
    db=0,
    decode_responses=False,  # Essential for binary data
    socket_timeout=5,
    health_check_interval=30
)

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def check_job_status(job_id: str) -> Dict[str, Any]:
    """Async job status check"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{FASTAPI_URL}/check_result/{job_id}",
                timeout=3.0
            )
            return response.json()
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"status": "error"}


async def handle_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /predict command"""
    if not context.args:
        await update.message.reply_text("Usage: /predict <your text>")
        return

    text = " ".join(context.args)
    logger.info(f"Predict request: {text[:50]}...")

    async with httpx.AsyncClient() as client:
        try:
            # Enqueue task
            response = await client.post(
                f"{FASTAPI_URL}/enqueue_task",
                json={"text": text, "user_id": str(update.message.chat_id)},
                timeout=5.0
            )

            if response.status_code != 200:
                raise ValueError(f"API error: {response.text}")

            job_id = response.json()["job_id"]
            await update.message.reply_text("🔍 Processing your text...")

            # Poll for results
            max_attempts = 10
            for attempt in range(max_attempts):
                result = await check_job_status(job_id)

                if result["status"] == "completed":
                    confidence = float(result["result"]["confidence"])
                    label = result["result"]["label"]
                    await update.message.reply_text(
                        f"Result: {label}\n"
                        f"Confidence: {confidence:.2%}"
                    )
                    return

                await asyncio.sleep(2)

            await update.message.reply_text("⏳ Processing is taking longer than expected")

        except Exception as e:
            logger.error(f"Predict error: {e}")
            await update.message.reply_text("⚠️ Service unavailable. Please try later.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message"""
    await update.message.reply_text(
        "Welcome! Use:\n"
        "/predict <text> - Analyze text\n"
    )


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set!")

    # Build application
    application = Application.builder() \
        .token(TELEGRAM_TOKEN) \
        .read_timeout(30) \
        .get_updates_read_timeout(30) \
        .build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", handle_predict))

    # Fallback for unhandled messages
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        lambda u, c: logger.info(f"Unhandled message: {u.message.text}")
    ))

    logger.info("Starting bot...")
    application.run_polling(
        poll_interval=0.5,
        timeout=30,
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()