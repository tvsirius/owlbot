import asyncio
import logging
import datetime
import requests
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
LOG_CHANNEL=os.environ['LOG_CHANNEL']
# -----------------------------------
bot = Bot(BOT_TOKEN)
dp = Dispatcher(bot)
# -----------------------------------


STUDENTS = LRUCache(50)

owlchat = OwlChat(OPENAI_API_KEY)

welcome_message = '''Вітаю тебе у чаті з найкращім віртуальним тьютором - Кібер Совою!'''

DO_INACTIVITY_CHECK = True

IDLE_COUNT = 4

# LOG HELPER function
def get_full_user_info(user_id):
    user_info = ""

    # Get user profile photos
    photos_url = f'https://api.telegram.org/bot{BOT_TOKEN}/getUserProfilePhotos?user_id={user_id}'
    response = requests.get(photos_url)
    photos_data = response.json()

    if 'result' in photos_data and photos_data['result']['total_count'] > 0:
        photo_info = photos_data['result']['photos'][0][0]
        photo_id = photo_info['file_id']
        photo_url = f'https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={photo_id}'
        photo_response = requests.get(photo_url)
        photo_data = photo_response.json()

        if 'result' in photo_data:
            file_path = photo_data['result']['file_path']
            photo_link = f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}'
            user_info += f"Profile Photo: {photo_link}\n"

    # Get user chat info
    chat_url = f'https://api.telegram.org/bot{BOT_TOKEN}/getChat?chat_id={user_id}'
    chat_response = requests.get(chat_url)
    chat_data = chat_response.json()

    if 'result' in chat_data:
        chat_info = chat_data['result']
        first_name = chat_info.get('first_name', '')
        last_name = chat_info.get('last_name', '')
        username = chat_info.get('username', '')

        user_info += f"First Name: {first_name}\n"
        user_info += f"Last Name: {last_name}\n"
        user_info += f"Username: {username}\n"
        user_info += f"User ID: {user_id}\n"

    return user_info

def get_short_user_info(user_id):
    # Get user chat info
    chat_url = f'https://api.telegram.org/bot{BOT_TOKEN}/getChat?chat_id={user_id}'
    chat_response = requests.get(chat_url)
    chat_data = chat_response.json()

    if 'result' in chat_data:
        chat_info = chat_data['result']
        first_name = chat_info.get('first_name', '')
        last_name = chat_info.get('last_name', '')
        username = chat_info.get('username', '')

        return f'{first_name}{last_name}({username},id={user_id}'
    else:
        return f'{user_id}'


@dp.message_handler(Command(commands=["start"]))
async def command_start_handler(message: Message) -> None:
    """
    This handler receive messages with `/start` command
    """
    await message.answer(f"Hello, {message.from_user.full_name}!\n{welcome_message}")
    await bot.send_message(chat_id=LOG_CHANNEL, text='/Starting conversation with\n\n' + get_short_user_info(message.from_user.id))



@dp.message_handler(content_types=types.ContentType.TEXT)
async def message_handler(message: types.Message) -> None:
    """
    Handler for message
    """
    global do_check_for_inactivity
    do_check_for_inactivity = False
    try:
        if message.chat.type == "private":
            if message.text.lower().strip()=="clear":
                print(f'message CLEAR!! {message.chat.type}')
                user_id = message.chat.id
                student = STUDENTS.get(user_id)
                student = StudentMemory(OPENAI_API_KEY, user_id, username=message.from_user.full_name)
                STUDENTS.put(user_id, student)
                await message.answer(text="Історію спілкування і всі данні перезаписано. Починайте нове спілкування.")
            else:
                print(f'message {message.chat.type}')
                user_id = message.chat.id
                student = STUDENTS.get(user_id)
                if student is None:
                    student = StudentMemory(OPENAI_API_KEY, user_id, username=message.from_user.full_name)
                    STUDENTS.put(user_id, student)
                    await bot.send_message(chat_id=LOG_CHANNEL, text='Starting conversation with\n\n'+get_full_user_info(user_id))
                student.inprocess = True
                print(f'message from {student.user_id, student.name}: {message.text}')
                student.input = message.text
                if len(student.input) > 0:
                    response, thought, prompt_ch = await owlchat.chat(student)
                    if len(response) > 0:
                        await message.answer(text=response)
                        await bot.send_message(chat_id=LOG_CHANNEL,
                                               text=f'{get_short_user_info(user_id)}:{student.input}\n'
                                                    f'Owl-Bot thought: ({prompt_ch}):{thought}\n'
                                                    f'Owl-Bot response: {response}')
                    student.last_time = datetime.datetime.now()
                student.idle_times = 0
                student.inprocess = False

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
    if DO_INACTIVITY_CHECK:
        while True:
            await asyncio.sleep(10)
            for key,student_cache in STUDENTS.cache.items():
                check_student=student_cache['value']
                if not check_student.inprocess:
                    if check_student.idle_check_time and check_student.last_time:
                        time_diff = datetime.datetime.now() - check_student.last_time
                        if time_diff >=  datetime.timedelta(seconds=check_student.idle_check_time):
                            if check_student.idle_times<IDLE_COUNT:
                                try:
                                    user_id = key
                                    student = STUDENTS.get(user_id)
                                    student.inprocess=True
                                    student.idle_times+=1
                                    response,thought, prompt_ch = await owlchat.chat(student, is_student_inactive=True)
                                    await bot.send_message(chat_id=student.user_id, text=response)
                                    await bot.send_message(chat_id=LOG_CHANNEL,
                                                           text=f'INACTIVITY OF {get_short_user_info(user_id)}\n'
                                                                f'Owl-Bot thought: ({prompt_ch}):{thought}\n'
                                                                f'Owl-Bot response: {response}')
                                    student.inprocess=False
                                    student.last_time=datetime.datetime.now()
                                except:
                                    print("Error with inactivity call")

async def main() -> None:
    asyncio.create_task(check_user_inactivity())
    await bot.send_message(chat_id=LOG_CHANNEL, text='Instance started')
    await dp.start_polling(bot)



'''
import requests

def get_channel_id(channel_username):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates'
    response = requests.get(url)
    data = response.json()
    print(data)

    if 'result' in data:
        for update in data['result']:
            if 'channel_post' in update:
                post = update['channel_post']
                if 'chat' in post and 'username' in post['chat'] and post['chat']['username'] == channel_username:
                    return post['chat']['id']
    return None


    channel_username = 'Owl-bot_log_chanel'
    channel_id = get_channel_id(channel_username)
    print(channel_id)
'''