import asyncio
import logging
import datetime
from dotenv import load_dotenv
import os

load_dotenv()
BOT_TOKEN = os.environ['BOT_TOKEN']

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import Message

#-----------------------------------
router = Router()
dp = Dispatcher()
dp.include_router(router)
bot = Bot(BOT_TOKEN, parse_mode="HTML")
#-----------------------------------



user_last_activity={}



@router.message(Command(commands=["start"]))
async def command_start_handler(message: Message) -> None:
    """
    This handler receive messages with `/start` command
    """
    await message.answer(f"Hello, <b>{message.from_user.full_name}!</b>")


@router.message()
async def message_handler(message: types.Message) -> None:
    """
    Handler for message
    """
    try:
        # Send copy of the received message
        print(message.text)
        await message.send_copy(chat_id=message.chat.id)
        user_last_activity[message.from_user.id] = {}
        user_last_activity[message.from_user.id]['idle_time'] = 60
        user_last_activity[message.from_user.id]['time'] = datetime.datetime.now()
    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")


async def check_user_inactivity():
    while True:
        await asyncio.sleep(30)  # Check user inactivity every 10 seconds
        current_time = datetime.datetime.now()
        for user_id in user_last_activity:
            if user_last_activity[user_id]['idle_time'] > 0:
                time_difference = current_time - user_last_activity[user_id]['time']
                if time_difference >= datetime.timedelta(seconds=user_last_activity[user_id]['idle_time']):
                    user_last_activity[user_id]['time']=current_time
                    await handle_user_inactivity(user_id, current_time, time_difference)


async def handle_user_inactivity(user_id: int, current_time, time_difference):
    print(f'Inactivity for user id={user_id}, time_diff={time_difference}')
    await bot.send_message(user_id, text="Спим?")


async def main() -> None:
    asyncio.create_task(check_user_inactivity())
    await dp.start_polling(bot)


