import os
import asyncio
import logging
import bot

import sys

# Define the log file path
log_file_path = "log.txt"


async def run_bot():

    await bot.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        # Redirect standard output to the log file
        sys.stdout = log_file
        asyncio.run(run_bot())
