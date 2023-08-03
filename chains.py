from langchain.prompts import load_prompt

from load_prompt_fix_encoding import load_prompt_utf

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from prompts.learning_program import fractions
from json import loads, dumps

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

#
CONTROL_PROMPT = load_prompt_utf("prompts/control.yaml")
PLAN_PROMPT = load_prompt_utf("prompts/plan.yaml")
TUTOR_PROMPT = load_prompt_utf("prompts/tutor.yaml")
RELAX_PROMPT = load_prompt_utf("prompts/relax.yaml")

TUTOR_TEMPERATURE = 0.5
RELAX_TEMPERATURE = 0.7

# learning_program = fractions

memory_defaults = {
    "memory_key": "history",
    "input_key": "input",
    "ai_prefix": "owl",
    "human_prefix": "student",
}

control_thought_schema=ResponseSchema(name="thought",
                      description="Твоя думка яка передбачає потреби учня з урахуванням поточного діалогу та його досягнень, а також перераховує інші фрагменти даних, які допоможуть покращити навчання. \
                      Ця думка може містити вказівки щодо того, як виконувати поточний крок взаємодії, який ти обираєш.")
control_next_step_schema=ResponseSchema(name="next_step",
                      description='Твій вибір наступного кроку. Повинен бути одинм із цих значеннь: ["PLAN", "TUTOR", "RELAX"].')

plan_response_schema=ResponseSchema(name="response",
                      description="Твоя відповідь учню. Якщо наступний урок ще не обрано, то ти відповідаєш на питання учня, і при цьому заохочуєш його продовжити навчання. Ти спілкуєшся з учнем українською мовою.")

plan_student_current_lesson_schema=ResponseSchema(name="student_current_lesson",
                        description="Якщо не обрано наступний урок, то ти повинен повернути пусту строку "". Якщо обрано - то student_current_lesson повинна містити у собі строку із заголовком та змістом урока.")

tutor_response_schema=ResponseSchema(name="response",
                        description="Твоя відповідь учню, від імені Кібер-Сови, розумного віртуальний тьютора, по поточному уроку, на українській мові.")
tutor_student_advance_schema=ResponseSchema(name="student_advance",
                        description='Якщо є висновок, що даний урок пройдено, то student_advance повинно дорівнювати "True", якщо ні, або ти не знаєш - то "False". Answer True if yes, False if not or unknown.')
tutor_student_summary_update_schema=ResponseSchema(name="student_summary_update",
                        description="Якщо є висновок, що даний урок пройдено, то в цю змінну ти заносиш коментар, щодо успіху учня по цьому уроку, для майбутнього розуміння його прогресу.")

relax_response_schema=ResponseSchema(name="response",
                        description="Твоя відповідь учню, або його неактивності, від імені Кібер-Сови, розумного, дружнього та зацікавленого віртуального тьютора, на українській мові.")



class StudentMemory:
    # memories
    #   DialogMemory - ConversationSummaryBufferMemory
    #   ControlThoughtsMemory - SummaryMemory
    #   StudentProgressSummary - SummaryMemory
    # state:
    #   CurrentEducationStep : str
    # time:
    #   InactivityThreshold : seconds
    #   LessonStart : datetime
    def __init__(self, user_id=None, username=''):
        # super().__init__()
        self.history = ConversationSummaryBufferMemory(**memory_defaults)
        self.though_history = ConversationSummaryBufferMemory(**memory_defaults, max_token_limit=500)
        self.student_progress = ConversationSummaryMemory()
        self.student_current_lesson = ""
        self.is_student_inactive = ""
        self.inactivity_duration = 0
        self.input = ""
        self.learning_program = fractions
        self.user_id = user_id
        self.name = username


class OwlChat:

    def __init__(self, OPENAI_API_KEY):

        self.llm0 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
        self.llm_TUTOR = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=TUTOR_TEMPERATURE,
                                    openai_api_key=OPENAI_API_KEY)
        self.llm_RELAX = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=RELAX_TEMPERATURE,
                                    openai_api_key=OPENAI_API_KEY)

        self.control_chain = LLMChain(llm=self.llm0,
                                      prompt=CONTROL_PROMPT,
                                      verbose=True
                                      )
        self.control_parser=StructuredOutputParser.from_response_schemas([control_thought_schema,control_next_step_schema])

        self.plan_chain = LLMChain(llm=self.llm0,
                                   prompt=PLAN_PROMPT,
                                   verbose=True
                                   )
        self.plan_parser=StructuredOutputParser.from_response_schemas([plan_response_schema, plan_student_current_lesson_schema])

        self.tutor_chain = LLMChain(llm=self.llm_TUTOR,
                                    prompt=TUTOR_PROMPT,
                                    verbose=True
                                    )

        self.tutor_parser=StructuredOutputParser.from_response_schemas([tutor_response_schema, tutor_student_advance_schema, tutor_student_summary_update_schema])

        self.relax_chain = LLMChain(llm=self.llm_TUTOR,
                                    prompt=RELAX_PROMPT,
                                    verbose=True
                                    )

        self.relax_parser = StructuredOutputParser.from_response_schemas([relax_response_schema])

    async def chat(self, student: StudentMemory):
        # ---------------------------------------------------------------------------------------------
        async def do_plan(thought):
            response = await self.plan_chain.apredict(
                student_progress=student.student_progress.load_memory_variables({})['history'],
                thought=thought,
                history=student.history.load_memory_variables({})['history'],
                input=student.input,
                learning_program=student.learning_program,
                format_instructions=self.plan_parser.get_format_instructions()
            )
            try:
                response_dict = self.plan_parser.parse(response)
            except:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            return response_dict["response"]

        async def do_tutor(thought):
            response = await self.tutor_chain.apredict(
                student_progress=student.student_progress.load_memory_variables({})['history'],
                thought=thought,
                history=student.history.load_memory_variables({})['history'],
                student_current_lesson=student.student_current_lesson,
                input=student.input,
                format_instructions=self.tutor_parser.get_format_instructions()
            )
            try:
                response_dict = self.tutor_parser.parse(response)
            except:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            if response_dict["student_advance"]:
                lesson_completed = student.student_current_lesson
                student.student_current_lesson = ""
                print(f"Урок пройдено: {lesson_completed}", response_dict["student_summary_update"])
                student.student_progress.save_context({"input": f"Урок пройдено: {lesson_completed}"},
                                                      {"output": response_dict["student_summary_update"]})
            return response_dict["response"]

        async def do_relax(thought):
            response = await self.relax_chain.apredict(
                student_progress=student.student_progress.load_memory_variables({})['history'],
                thought=thought,
                history=student.history.load_memory_variables({})['history'],
                input=student.input,
                student_current_lesson=student.student_current_lesson,
                is_student_inactive=student.is_student_inactive,
                format_instructions=self.relax_parser.get_format_instructions()
            )
            try:
                response_dict = self.relax_parser.parse(response)
            except:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            return response_dict["response"]

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        print(f"chat with {student.user_id, student.name}, input={student.input}")
        thought_response = await self.control_chain.apredict(
            student_progress=student.student_progress.load_memory_variables({})['history'],
            student_current_lesson=student.student_current_lesson,
            is_student_inactive=student.is_student_inactive,
            history=student.history.load_memory_variables({})['history'],
            though_history=student.though_history.load_memory_variables({})['history'],
            input=student.input,
            format_instructions=self.control_parser.get_format_instructions()
        )
        try:
            thought_response_dict = self.control_parser.parse(thought_response)
        except:
            print(f'ERROR parsing output: {thought_response}')
            return
        print(f'Thoghts_response_dict={thought_response_dict}')
        thought, next_step = thought_response_dict['thought'], thought_response_dict['next_step']

        student.though_history.save_context({"input": student.input}, {"output": thought})
        if next_step == "PLAN":
            response = await do_plan(thought)
        elif next_step == "TUTOR":
            response = await do_tutor(thought)
        else:
            response = await do_relax(thought)

        return response
