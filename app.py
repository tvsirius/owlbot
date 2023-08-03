import os
import asyncio
import logging
import bot


async def run_bot():

    await bot.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_bot())
