import os
import asyncio
from dotenv import load_dotenv
import bot

load_dotenv()
BOT_TOKEN = os.environ['BOT_TOKEN']


async def run_bot():
    await bot.main(BOT_TOKEN)

if __name__ == "__main__":
    asyncio.run(run_bot())
