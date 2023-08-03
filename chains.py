from langchain.prompts import load_prompt, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory,ConversationSummaryBufferMemory, ConversationSummaryMemory
from prompts.learning_program import fractions

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


CONTROL_PROMPT=load_prompt("prompts/control.yaml")
PLAN_PROMPT=load_prompt("prompts/plan.yaml")
TUTOR_PROMPT=load_prompt("prompts/tutor.yaml")
RELAX_PROMPT=load_prompt("prompts/relax.yaml")

TUTOR_TEMPERATURE=0.5
RELAX_TEMPERATURE=0.7

learning_program=fractions

class OwlBotChat:
    # chains
    #   ControlChain
    #   PlanChain
    #   TutorChain
    #   RelaxChain
    def __init__(self, OPENAI_API_KEY):
        #   ControlChain
        self.llm0=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
        self.llm_TUTOR=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=TUTOR_TEMPERATURE, openai_api_key=OPENAI_API_KEY)
        self.llm_RELAX=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=RELAX_TEMPERATURE, openai_api_key=OPENAI_API_KEY)

        control_chain=LLMChain(llm=self.llm0,
                               prompt=CONTROL_PROMPT,
                               verbose=True
                            )
        plan_chain=LLMChain(llm=self.llm0,
                               prompt=PLAN_PROMPT,
                               verbose=True
                            )
        tutor_chain=LLMChain(llm=self.llm_TUTOR,
                               prompt=TUTOR_PROMPT,
                               verbose=True
                            )
        relax_chain=LLMChain(llm=self.llm_TUTOR,
                               prompt=RELAX_PROMPT,
                               verbose=True
                            )



class SudentMemory:
    # memories
    #   DialogMemory - ConversationSummaryBufferMemory
    #   ControlThoughtsMemory - SummaryMemory
    #   StudentProgressSummary - SummaryMemory
    # state:
    #   CurrentEducationStep : str
    # time:
    #   InactivityThreshold : seconds
    #   LessonStart : datetime
    #    #    ["student_progres", "student_current_lesson",  "is_student_inactive",  "history", "though_history", "input"]

    def __init__(self):
        # super().__init__()
        self.history=ConversationSummaryBufferMemory()
        self.though_history=ConversationSummaryBufferMemory(max_token_limit=500)
        self.student_progres=ConversationSummaryMemory()
        self.student_current_lesson=""
        self.is_student_inactive=""
        self.inactivity_duration=0
        self.input=""



