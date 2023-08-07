import asyncio
import logging
import datetime
from dotenv import load_dotenv
import os

from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher.filters import Command

from aiogram.types import Message

from chains import OwlChat, StudentMemory
from chache import LRUCache

# ------------------------------------
load_dotenv()
BOT_TOKEN = os.environ['BOT_TOKEN']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# -----------------------------------
bot = Bot(BOT_TOKEN)
dp = Dispatcher(bot)
# -----------------------------------


STUDENTS = LRUCache(50)

owlchat = OwlChat(OPENAI_API_KEY)

welcome_message = '''Вітаю тебе у чаті з найкращім віртуальним тьютором - Кібер Совою!'''

do_check_for_inactivity=False

@dp.message_handler(Command(commands=["start"]))
async def command_start_handler(message: Message) -> None:
    """
    This handler receive messages with `/start` command
    """
    await message.answer(f"Hello, <b>{message.from_user.full_name}!</b><br>{welcome_message}")


@dp.message_handler(content_types=types.ContentType.TEXT)
async def message_handler(message: types.Message) -> None:
    """
    Handler for message
    """
    global do_check_for_inactivity
    do_check_for_inactivity = False
    try:
        if message.chat.type == "private":
            print(f'message {message.chat.type}')
            user_id = message.chat.id
            student = STUDENTS.get(user_id)
            if student is None:
                student = StudentMemory(OPENAI_API_KEY, user_id, username=message.from_user.full_name)
                STUDENTS.put(user_id, student)

            print(f'message from {student.user_id, student.name}: {message.text}')
            student.input = message.text
            if len(student.input) > 0:
                response = await owlchat.chat(student)
                if len(response) > 0:
                    await message.answer(text=response)
                student.last_time = datetime.datetime.now()
            do_check_for_inactivity=True
        else:
            pass
    except TypeError:
        print('typeerror')
        await message.answer("Вибачь, сталась помилка!")
        do_check_for_inactivity = True


async def check_user_inactivity():
    '''
    await asyncio.sleep(30)  # Check user inactivity every 10 seconds
    current_time = datetime.datetime.now()
    for user_id in :
        if user_last_activity[user_id]['idle_time'] > 0:
            time_difference = current_time - user_last_activity[user_id]['time']
            if time_difference >= datetime.timedelta(seconds=user_last_activity[user_id]['idle_time']):
                user_last_activity[user_id]['time']=current_time
                await handle_user_inactivity(user_id, current_time, time_difference)'''
    while True:
        await asyncio.sleep(20)
        if do_check_for_inactivity:
            for key,student_cache in STUDENTS.cache.items():
                check_student=student_cache['value']
                if check_student.idle_check_time and check_student.last_time:
                    time_diff = datetime.datetime.now() - check_student.last_time
                    if time_diff >=  datetime.timedelta(seconds=check_student.idle_check_time):
                        try:
                            user_id = key
                            student = STUDENTS.get(user_id)

                            response = await owlchat.chat(student, is_student_inactive=True)
                            await bot.send_message(chat_id=student.user_id, text=response)
                            student.last_time=datetime.datetime.now()
                        except:
                            print("Error with inactivity call")



async def handle_user_inactivity(user_id: int, current_time, time_difference):
    print(f'Inactivity for user id={user_id}, time_diff={time_difference}')
    await bot.send_message(user_id, text="Спим?")


async def main() -> None:
    asyncio.create_task(check_user_inactivity())
    await dp.start_polling(bot)
