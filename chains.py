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
CONTROL_PLAN_PROMPT = load_prompt_utf("prompts/control-to-plan.yaml")
CONTROL_TUTOR_PROMPT = load_prompt_utf("prompts/control-to-tutor.yaml")
PLAN_PROMPT = load_prompt_utf("prompts/plan.yaml")
TUTOR_PROMPT = load_prompt_utf("prompts/tutor.yaml")
RELAX_PROMPT = load_prompt_utf("prompts/relax.yaml")

TUTOR_TEMPERATURE = 0.7
RELAX_TEMPERATURE = 0.8

# learning_program = fractions


control_thought_schema = ResponseSchema(name="thought",
                                        description="Yor Thought that makes a prediction about the scholar's needs given current dialogue")
control_plan_next_step_schema = ResponseSchema(name="next_step",
                                               description='Your chose of best prompt. Must be one of: ["PLAN", "RELAX"].')
control_tutor_next_step_schema = ResponseSchema(name="next_step",
                                                description='Your chose of best prompt. Must be one of: ["TUTOR", "RELAX"].')

plan_response_schema = ResponseSchema(name="response",
                                      description="Response in Ukrainian language.")
plan_student_current_lesson_schema = ResponseSchema(name="student_current_lesson",
                                                    description='Must be left blank (equals "") if the next lesson topic is not confirmed, or you are not sure. If the topic is confirmed, must be set to lesson number and title and content - listing of topics of the lesson ')

tutor_response_schema = ResponseSchema(name="response",
                                       description="Response to the scholar input in Ukrainian language.")
tutor_student_advance_schema = ResponseSchema(name="student_advance",
                                              description='If lesson is completed must be set to "True", if not, or you are not sure must be set "False"')
tutor_student_summary_update_schema = ResponseSchema(name="student_summary_update",
                                                     description="Only if there is a conclusion, that scholar have successfully completed this lesson, must contain your comment on scholar achievement on this lesson")

relax_response_schema = ResponseSchema(name="response",
                                       description="Funny response to the scholar input in Ukrainian language.")

memory_defaults = {
    "memory_key": "history",
    "input_key": "input",
    "ai_prefix": "Cyber-Owl",
     # "human_prefix": "Scholar",
}


def manual_parse(s:str):
    return loads("{"+s.split("{")[1].split("}")[0].strip()+"}")

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
    def __init__(self, openai_api_key, user_id=None, username=''):
        # super().__init__()]
        self.name = username
        self.user_id = user_id
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4, openai_api_key=openai_api_key)
        self.history = ConversationSummaryBufferMemory(llm=self.llm, **memory_defaults, max_token_limit=700, human_prefix=self.name)
        # self.though_history = ConversationSummaryBufferMemory(llm=self.llm, **memory_defaults, max_token_limit=500)
        self.student_progress = ConversationSummaryMemory(llm=self.llm)
        self.student_current_lesson = ""
        self.is_student_inactive = ""
        self.inactivity_duration = 0
        self.input = ""
        self.learning_program = fractions


class OwlChat:

    def __init__(self, openai_api_key):

        self.llm0 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        self.llm_TUTOR = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=TUTOR_TEMPERATURE,
                                    openai_api_key=openai_api_key)
        self.llm_RELAX = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=RELAX_TEMPERATURE,
                                    openai_api_key=openai_api_key)

        self.control_plan_chain = LLMChain(llm=self.llm0,
                                           prompt=CONTROL_PLAN_PROMPT,
                                           verbose=True
                                           )
        self.control_plan_parser = StructuredOutputParser.from_response_schemas(
            [control_thought_schema, control_plan_next_step_schema])

        self.control_tutor_chain = LLMChain(llm=self.llm0,
                                            prompt=CONTROL_TUTOR_PROMPT,
                                            verbose=True
                                            )
        self.control_tutor_parser = StructuredOutputParser.from_response_schemas(
            [control_thought_schema, control_tutor_next_step_schema])

        self.plan_chain = LLMChain(llm=self.llm0,
                                   prompt=PLAN_PROMPT,
                                   verbose=True
                                   )
        self.plan_parser = StructuredOutputParser.from_response_schemas(
            [plan_response_schema, plan_student_current_lesson_schema])

        self.tutor_chain = LLMChain(llm=self.llm_TUTOR,
                                    prompt=TUTOR_PROMPT,
                                    verbose=True
                                    )

        self.tutor_parser = StructuredOutputParser.from_response_schemas(
            [tutor_response_schema, tutor_student_advance_schema, tutor_student_summary_update_schema])

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
                name=student.name,
                learning_program=student.learning_program,
                format_instructions=self.plan_parser.get_format_instructions()
            )
            try:
                response_dict = self.plan_parser.parse(response)
            except:
                try:
                    response_dict=manual_parse(response)
                except:
                    print(f'ERROR parsing output: {response}')
                    return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            student.student_current_lesson = response_dict["student_current_lesson"]
            return response_dict["response"]

        async def do_tutor(thought):
            response = await self.tutor_chain.apredict(
                # student_progress=student.student_progress.load_memory_variables({})['history'],
                thought=thought,
                history=student.history.load_memory_variables({})['history'],
                student_current_lesson=student.student_current_lesson,
                input=student.input,
                name=student.name,
                format_instructions=self.tutor_parser.get_format_instructions()
            )
            try:
                response_dict = self.tutor_parser.parse(response)
            except:
                try:
                    response_dict = manual_parse(response)
                except:
                    print(f'ERROR parsing output: {response}')
                    return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            if response_dict["student_advance"] == "True":
                lesson_completed = student.student_current_lesson
                student.student_current_lesson = ""
                print(f"Урок пройдено: {lesson_completed}", response_dict["student_summary_update"])
                student.student_progress.save_context({"input": f"Урок пройдено: {lesson_completed}"},
                                                      {"output": response_dict["student_summary_update"]})
            return response_dict["response"]

        async def do_relax(thought):
            response = await self.relax_chain.apredict(
                # student_progress=student.student_progress.load_memory_variables({})['history'],
                thought=thought,
                history=student.history.load_memory_variables({})['history'],
                input=student.input,
                # student_current_lesson=student.student_current_lesson,
                is_student_inactive=student.is_student_inactive,
                name=student.name,
                format_instructions=self.relax_parser.get_format_instructions()
            )
            try:
                response_dict = self.relax_parser.parse(response)
            except:
                try:
                    response_dict = manual_parse(response)
                except:
                    print(f'ERROR parsing output: {response}')
                    return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            return response_dict["response"]

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        print(f"chat with {student.user_id, student.name}, input={student.input}")
        if student.student_current_lesson == '':
            print('Entering plan control')
            thought_response = await self.control_plan_chain.apredict(
                student_progress=student.student_progress.load_memory_variables({})['history'],
                # student_current_lesson=student.student_current_lesson,
                is_student_inactive=student.is_student_inactive,
                history=student.history.load_memory_variables({})['history'],
                # though_history=student.though_history.load_memory_variables({})['history'],
                input=student.input,
                # learning_program=student.learning_program,
                name=student.name,
                format_instructions=self.control_plan_parser.get_format_instructions()
            )
            print('Complete plan control')
            try:
                thought_response_dict = self.control_plan_parser.parse(thought_response)
            except:
                try:
                    thought_response_dict = manual_parse(thought_response)
                except:
                    print(f'ERROR parsing output: {thought_response}')
                    return
            print(f'Thoghts_response_dict={thought_response_dict}')
            thought, next_step = thought_response_dict['thought'], thought_response_dict['next_step']
        else:
            thought_response = await self.control_tutor_chain.apredict(
                student_progress=student.student_progress.load_memory_variables({})['history'],
                student_current_lesson=student.student_current_lesson,
                is_student_inactive=student.is_student_inactive,
                history=student.history.load_memory_variables({})['history'],
                # though_history=student.though_history.load_memory_variables({})['history'],
                input=student.input,
                name=student.name,
                format_instructions=self.control_tutor_parser.get_format_instructions()
            )
            try:
                thought_response_dict = self.control_tutor_parser.parse(thought_response)
            except:
                print(f'ERROR parsing output: {thought_response}')
                return
            print(f'Thoghts_response_dict={thought_response_dict}')
            thought, next_step = thought_response_dict['thought'], thought_response_dict['next_step']

        # student.though_history.save_context({"input": student.input}, {"output": thought})
        if next_step == "RELAX":
            response = await do_relax(thought)
        elif next_step == "TUTOR" and not (student.student_current_lesson == ""):
            response = await do_tutor(thought)
        elif next_step == "PLAN" and student.student_current_lesson == "":
            response = await do_plan(thought)
        else:
            print("PROGRAM ERROR IN REASONING ENGINE")
            response = await do_plan(thought)

        return response
