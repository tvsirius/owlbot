import os
import asyncio
import logging
from dotenv import load_dotenv
import bot

load_dotenv()



async def run_bot():

    await bot.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_bot())
