from load_prompt_fix_encoding import load_prompt_utf
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from json import loads
import datetime
from langchain.prompts.prompt import PromptTemplate

#
CONTROL_PROMPT = load_prompt_utf("prompts/control.yaml")
PLAN_PROMPT = load_prompt_utf("prompts/plan.yaml")
TUTOR_PROMPT = load_prompt_utf("prompts/tutor.yaml")
RELAX_PROMPT = load_prompt_utf("prompts/relax.yaml")
UPDATE_PROMPT = load_prompt_utf("prompts/update.yaml")

CONTROL_TEMPERATURE = 0.0
PLAN_TEMPERATURE = 0.2
TUTOR_TEMPERATURE = 1
RELAX_TEMPERATURE = 1

SUMMARY_MEMORY_TOKEN_LIMIT = 1200

IDLE_INACTIVITY_DEFAULT = 50

memory_defaults = {
    "memory_key": "history",
    "input_key": "input",
    "ai_prefix": "Cyber-Owl",
}

from data.learning_program import learning_prorgam

LEARNING_PROGRAM_KEYS = "[" + ", ".join(['"' + key + '"' for key in learning_prorgam.keys()]) + "]"
LEARNING_PROGRAM_INFO = {}
for key, program in learning_prorgam.items():
    LEARNING_PROGRAM_INFO[key] = f'Программа навчання темі {program["name"]}:\n' + "\n".join(
        [lesson.split("\n")[0] for lesson in program["lessons"]])

LEARNING_PROGRAM_ALL_INFO = "\n\n".join(
    [learning_prorgam[key]["name"] + ' ("' + key + '")' + ':' + value for key, value in LEARNING_PROGRAM_INFO.items()])

print(LEARNING_PROGRAM_KEYS)
print(LEARNING_PROGRAM_ALL_INFO)


def json_parse(s: str):
    check_words = ['"student_advance"', '"program"']
    ## Helper function to fix ChatGPT sometimes missing coma (,) in JSON output
    try:
        for check_word in check_words:
            split_s = s.split(check_word)
            if len(split_s) > 1:
                split_s_2 = split_s[0].split('"')
                if not split_s_2[-1].startswith(','):
                    split_s_2[-1] = ',' + split_s_2[-1]
                s = check_word.join(['"'.join(split_s_2)] + split_s[1:])
        return loads("{" + s.split("{")[1].split("}")[0].strip() + "}")
    except:
        pass


def SUMMARY_PROMPT_UKRAINAN():
    _DEFAULT_SUMMARIZER_TEMPLATE = '''Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary. 
    This is a convrsation between Cyber-Owl teacher and a sholar. You should produce a new summary about all scholar progress in this lesson.
    Summary must track all scholar understandign and advancement. 
    Unrelevant information does not need saving.

    Current summary:
    {summary}
    
    New lines of conversation:
    {new_lines}
    
    New summary:
    
    Output summary in Ukrainian language.
    
    New summary:'''
    return PromptTemplate(input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE)


class StudentMemory:

    def __init__(self, openai_api_key, user_id=None, username=''):
        self.name = username
        self.user_id = user_id
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=CONTROL_TEMPERATURE,
                              openai_api_key=openai_api_key)
        self.history = ConversationSummaryBufferMemory(llm=self.llm, **memory_defaults,
                                                       max_token_limit=SUMMARY_MEMORY_TOKEN_LIMIT,
                                                       prompt=SUMMARY_PROMPT_UKRAINAN(),
                                                       human_prefix=self.name
                                                       )
        self.update_chain = LLMChain(
            llm=self.llm,
            prompt=UPDATE_PROMPT,
            verbose=True
        )
        self.progress = {"general": ""}
        for key, program in learning_prorgam.items():
            self.progress[key] = {
                "progress": "",
                "lesson": 0
            }
        self.current_program = None
        self.student_inactive_info = ""
        self.idle_check_time = 0
        self.idle_wait_next_phrase = False
        self.last_time = None
        self.inprocess = False
        self.idle_times = 0
        self.input = ""

    async def update_progress(self, program=None, advance=False):
        if program is None:
            program = self.current_program
        print('Running UPDATE prompt')
        if advance:
            program_done = f'Учень тільки що успішно завершив {self.progress[program]["lesson"]} урок із {len(learning_prorgam[program]["lessons"]) + 1} уроків программи {learning_prorgam[program]["name"]}'
        else:
            program_done = f'Учень пройшов {self.progress[program]["lesson"]} уроків із {len(learning_prorgam[program]["lessons"]) + 1} уроків программи {learning_prorgam[program]["name"]}, і ' \
                           f'вирішив поки займатись по інший програмі, або зробити перерву'
        if self.current_program:
            current_progress = self.progress[self.current_program]
            lesson_topic =learning_prorgam[self.current_program]["lessons"][current_progress[
                "lesson"]]
        else:
            lesson_topic = "Програму навчання ще не обрано"
        response = await self.update_chain.apredict(
            program=learning_prorgam[program]["name"],
            history=self.history.load_memory_variables({})['history'],
            total_progress=self.progress["general"],
            program_progress=self.progress[program]["progress"],
            program_done=program_done,
            lesson_topic=lesson_topic,
        )
        response_dict = json_parse(response)
        if not response_dict:
            print(f'ERROR parsing output: {response}')
            return
        print(response_dict)

        self.progress["general"] = response_dict["general_progress"]
        self.progress[program]["progress"] = response_dict["program_progress"]
        self.history = ConversationSummaryBufferMemory(llm=self.llm, **memory_defaults,
                                                       max_token_limit=SUMMARY_MEMORY_TOKEN_LIMIT,
                                                       prompt=SUMMARY_PROMPT_UKRAINAN(),
                                                       human_prefix=self.name
                                                       )
        self.history.save_context({"input": ""}, {"output": "system: " + program_done})
        print("Student progress updated")

    def process_inactivity_info(self, is_student_inactive):
        if is_student_inactive:
            self.input = "(учень мовчить)"
            if self.idle_wait_next_phrase:
                self.student_inactive_info = f"Пройшло вже більше {self.idle_check_time} секунд паузи у навчанні, можна продовжувати навчання"
                self.idle_wait_next_phrase = False
                self.idle_check_time = IDLE_INACTIVITY_DEFAULT
            else:
                self.student_inactive_info = f"Пройшло вже більше {self.idle_check_time} секунд як учнень не відповідає. Треба привернути його увагу"
        else:
            self.student_inactive_info = ""


class OwlChat:

    def __init__(self, openai_api_key):

        self.llm_CONTROL = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=CONTROL_TEMPERATURE,
                                      openai_api_key=openai_api_key)
        self.llm_PLAN = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=PLAN_TEMPERATURE,
                                   openai_api_key=openai_api_key)
        self.llm_TUTOR = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=TUTOR_TEMPERATURE,
                                    openai_api_key=openai_api_key)
        self.llm_RELAX = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=RELAX_TEMPERATURE,
                                    openai_api_key=openai_api_key)

        self.control_chain = LLMChain(llm=self.llm_CONTROL,
                                      prompt=CONTROL_PROMPT,
                                      verbose=True
                                      )

        self.plan_chain = LLMChain(llm=self.llm_CONTROL,
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

    async def chat(self, student: StudentMemory, is_student_inactive=False):
        # ---------------------------------------------------------------------------------------------
        async def do_plan(thought):
            response = await self.plan_chain.apredict(
                current_program=learning_prorgam[student.current_program][
                    "name"] if student.current_program else "Програму навчання ще не обрано",
                learning_program_info=LEARNING_PROGRAM_ALL_INFO,
                learning_program_keys=LEARNING_PROGRAM_KEYS,
                total_progress=student.progress["general"],
                name=student.name,
                history=student.history.load_memory_variables({})['history'],
                student_inactive_info=student.student_inactive_info,
                input=student.input,
                thought=thought,
            )
            response_dict = json_parse(response)
            if not response_dict:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            if response_dict["change_program"] and response_dict["program"] in learning_prorgam.keys():
                await student.update_progress()
            if (response_dict["program"] in learning_prorgam.keys()) and not (
                    response_dict["program"] == student.current_program):
                student.current_program = response_dict["program"]
                current_progress = student.progress[student.current_program] if student.current_program else None
                lesson_plan = learning_prorgam[student.current_program]["lessons"][
                    current_progress["lesson"]] if student.current_program else "Програму навчання ще не обрано"
                student.history = ConversationSummaryBufferMemory(llm=student.llm, **memory_defaults,
                                                                  max_token_limit=SUMMARY_MEMORY_TOKEN_LIMIT,
                                                                  prompt=SUMMARY_PROMPT_UKRAINAN(),
                                                                  human_prefix=student.name
                                                                  )
                student.history.save_context({"input": ""}, {"output": "Учень повчав навчання"})
            if response_dict["program"] in learning_prorgam.keys() and student.current_program is None:
                print(f'Program {response_dict["program"]} selected')
                student.current_program = response_dict["program"]
                current_progress = student.progress[student.current_program] if student.current_program else None
                lesson_plan = learning_prorgam[student.current_program]["lessons"][
                    current_progress["lesson"]] if student.current_program else "Програму навчання ще не обрано"
                student.history = ConversationSummaryBufferMemory(llm=student.llm, **memory_defaults,
                                                                  max_token_limit=SUMMARY_MEMORY_TOKEN_LIMIT,
                                                                  prompt=SUMMARY_PROMPT_UKRAINAN(),
                                                                  human_prefix=student.name)
                student.history.save_context({"input": ""}, {"output": "Учень повчав навчання"})
            else:
                print("Program not selected")
            if response_dict["do_learn"]:
                # student.idle_wait_next_phrase = True
                student.idle_check_time = IDLE_INACTIVITY_DEFAULT
            else:
                student.idle_check_time = 0
            return response_dict["response"]

        async def do_tutor(thought):
            current_progress = student.progress[student.current_program] if student.current_program else None
            response = await self.tutor_chain.apredict(
                # current_program=learning_prorgam[student.current_program][
                #     "name"] if student.current_program else "Програму навчання ще не обрано",
                lesson_topic=learning_prorgam[student.current_program]["lessons"][current_progress[
                    "lesson"]] if student.current_program else "Програму навчання ще не обрано",
                # progress_on_program=current_progress[
                #     "progress"] if student.current_program else "Програму навчання ще не обрано",
                # total_progress=student.progress["general"],
                name=student.name,
                history=student.history.load_memory_variables({})['history'],
                student_inactive_info=student.student_inactive_info,
                input=student.input,
                thought=thought,
            )
            response_dict = json_parse(response)
            if not response_dict:
                try:
                    if response is str and len(response) > 0:
                        if response.startswith('Cyber-Owl:') or response.startswith('Teacher:'):
                            response = " ".join(response.split(" ")[1:])
                        response_dict = {"response": response, "student_advance": False}
                    else:
                        print(f'ERROR parsing output: {response}')
                        return
                except:
                    print(f'ERROR parsing output: {response}')
                    return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})

            if (response_dict["student_advance"] == "True") or (
                    isinstance(response_dict["student_advance"], bool) and response_dict["student_advance"]):
                print('LESSON COMPLETED!')
                await student.update_progress(student.current_program, advance=True)
                student.progress[student.current_program]["lesson"] += 1
                pass

            student.idle_check_time = IDLE_INACTIVITY_DEFAULT

            return response_dict["response"]

        async def do_relax(thought):
            response = await self.relax_chain.apredict(
                # current_program=learning_prorgam[student.current_program][
                #     "name"] if student.current_program else "Програму навчання ще не обрано",
                lesson_topic=learning_prorgam[student.current_program]["lessons"][current_progress[
                    "lesson"]] if student.current_program else "Програму навчання ще не обрано",
                # progress_on_program=student.progress[student.current_program][
                #     "progress"] if student.current_program else "Програму навчання ще не обрано",
                # total_progress=student.progress["general"],
                name=student.name,
                history=student.history.load_memory_variables({})['history'],
                student_inactive_info=student.student_inactive_info,
                input=student.input,
                thought=thought,
            )
            response_dict = json_parse(response)
            if not response_dict:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            return response_dict["response"]

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        current_progress = student.progress[student.current_program] if student.current_program else None
        student.process_inactivity_info(is_student_inactive)
        print(f"chat with {student.user_id, student.name}, input={student.input}")
        control_response = await self.control_chain.apredict(
            current_program=learning_prorgam[student.current_program][
                "name"] if student.current_program else "Програму навчання ще не обрано",
            # lesson_topic=learning_prorgam[student.current_program]["lessons"][current_progress[
            #     "lesson"]] if student.current_program else "Програму навчання ще не обрано",
            total_progress=student.progress["general"],
            name=student.name,
            history=student.history.load_memory_variables({})['history'],
            student_inactive_info=student.student_inactive_info,
            input=student.input,
        )

        control_response_dict = json_parse(control_response)
        if not control_response_dict:
            print(f'ERROR parsing output: {control_response}')
            return
        print(f'Control_response_dict={control_response_dict}')
        next_prompt = control_response_dict['prompt']
        thought = control_response_dict['thought']

        # student.though_history.save_context({"input": student.input}, {"output": thought})
        if next_prompt == "RELAX":
            response = await do_relax(thought)
        elif next_prompt == "TUTOR" and student.current_program is not None:
            response = await do_tutor(thought)
        elif next_prompt == "PLAN":
            response = await do_plan(thought)
        else:
            print("ERROR IN CONTROL PROMPT REASONING")
            response = await do_plan(thought)

        return response, thought, next_prompt
