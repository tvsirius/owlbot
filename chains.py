from langchain.prompts import load_prompt, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from prompts.learning_program import fractions
from json import loads,dumps

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

CONTROL_PROMPT = load_prompt("prompts/control.yaml")
PLAN_PROMPT = load_prompt("prompts/plan.yaml")
TUTOR_PROMPT = load_prompt("prompts/tutor.yaml")
RELAX_PROMPT = load_prompt("prompts/relax.yaml")

TUTOR_TEMPERATURE = 0.5
RELAX_TEMPERATURE = 0.7

# learning_program = fractions

memory_defaults = {
    "memory_key": "history",
    "input_key": "input",
    "ai_prefix": "owl",
    "human_prefix": "student",
}

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
        self.though_history = ConversationSummaryBufferMemory(**memory_defaults,max_token_limit=500)
        self.student_progress = ConversationSummaryMemory()
        self.student_current_lesson = ""
        self.is_student_inactive = ""
        self.inactivity_duration = 0
        self.input = ""
        self.learning_program = fractions
        self.user_id=user_id
        self.name=username



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
        self.plan_chain = LLMChain(llm=self.llm0,
                              prompt=PLAN_PROMPT,
                              verbose=True
                              )
        self.tutor_chain = LLMChain(llm=self.llm_TUTOR,
                               prompt=TUTOR_PROMPT,
                               verbose=True
                               )
        self.relax_chain = LLMChain(llm=self.llm_TUTOR,
                               prompt=RELAX_PROMPT,
                               verbose=True
                               )

    async def chat(self, student: StudentMemory):
        #---------------------------------------------------------------------------------------------
        async def do_plan(thought):
            response = await self.plan_chain.apredict(
                student_progress=student.student_progress.load_memory_variables({})['history'],
                thought=thought,
                history=student.history.load_memory_variables({})['history'],
                input=student.input,
                learning_program=student.learning_program,
            )
            try:
                response_dict=loads(response)
            except:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input":student.input}, {"output": response_dict["response"]})
            return response_dict["response"]

        async def do_tutor(thought):
            response = await self.tutor_chain.apredict(
                student_progress=student.student_progress.load_memory_variables({})['history'],
                thought=thought,
                history=student.history.load_memory_variables({})['history'],
                student_current_lesson=student.student_current_lesson,
                input=student.input,
            )
            try:
                response_dict=loads(response)
            except:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input":student.input}, {"output": response_dict["response"]})
            if response_dict["student_advance"]:
                lesson_completed=student.student_current_lesson
                student.student_current_lesson=""
                print(f"Урок пройдено: {lesson_completed}",response_dict["student_summary_update"])
                student.student_progress.save_context({"input":f"Урок пройдено: {lesson_completed}"},{"output":response_dict["student_summary_update"]})
            return response_dict["response"]


        async def do_relax(thought):
            response = await self.relax_chain.apredict(
                student_progress=student.student_progress.load_memory_variables({})['history'],
                thought=thought,
                history=student.history.load_memory_variables({})['history'],
                input=student.input,
                student_current_lesson=student.student_current_lesson,
                is_student_inactive=student.is_student_inactive,
            )
            try:
                response_dict=loads(response)
            except:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input":student.input}, {"output": response_dict["response"]})
            return response_dict["response"]


        #------------------------------------------------------------
        #------------------------------------------------------------
        print(f"chat with {student.user_id, student.name}, input={student.input}")
        thought_response=await self.control_chain.apredict(
            student_progress=student.student_progress.load_memory_variables({})['history'],
            student_current_lesson=student.student_current_lesson,
            is_student_inactive=student.is_student_inactive,
            history=student.history.load_memory_variables({})['history'],
            though_history=student.though_history.load_memory_variables({})['history'],
            input=student.input
        )
        try:
            thought_response_dict=loads(thought_response)
        except:
            print(f'ERROR parsing output: {thought_response}')
            return
        print(f'Thoghts_response_dict={thought_response_dict}')
        thought, next_step =thought_response_dict['thought'],thought_response_dict['next_step']

        student.though_history.save_context({"input":student.input}, {"output": thought})
        if next_step=="PLAN":
            response=await do_plan(thought)
        elif next_step=="TUTOR":
            response=await do_tutor(thought)
        else:
            response=await do_relax(thought)

        return response

