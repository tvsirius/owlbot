from langchain.prompts import load_prompt
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


CONTROL_PROMPT=load_prompt("prompts/control.yaml")


class OwlBotChat:
    # chains
    #   ControlChain
    #   EducationPlanChain
    #   OwlEducationChain
    #   RelaxChain
    def __init__(self):
        #   ControlChain

        pass


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
    def __init__(self):
        # super().__init__()
        pass


