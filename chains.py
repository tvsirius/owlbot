from load_prompt_fix_encoding import load_prompt_utf
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from json import loads

#
CONTROL_PROMPT = load_prompt_utf("prompts/control.yaml")
PLAN_PROMPT = load_prompt_utf("prompts/plan.yaml")
TUTOR_PROMPT = load_prompt_utf("prompts/tutor.yaml")
RELAX_PROMPT = load_prompt_utf("prompts/relax.yaml")
UPDATE_PROMPT = load_prompt_utf("prompts/update.yaml")

CONTROL_TEMPERATURE = 0.0
TUTOR_TEMPERATURE = 0.6
RELAX_TEMPERATURE = 0.8

IDLE_INACTIVITY_DEFAUL=30

memory_defaults = {
    "memory_key": "history",
    "input_key": "input",
    "ai_prefix": "Cyber-Owl",
    # "human_prefix": "Scholar",
}

from data.learning_program import learning_prorgam

LEARNING_PROGRAM_KEYS = "[" + ", ".join(['"' + key + '"' for key in learning_prorgam.keys()]) + "]"
LEARNING_PROGRAM_INFO = {}
for key, program in learning_prorgam.items():
    LEARNING_PROGRAM_INFO[key] = f'Программа навчання темі {program["name"]}:\n' + "\n".join(
        [lesson.split("\n")[0] for lesson in program["lessons"]])

LEARNING_PROGRAM_ALL_INFO = "\n\n".join(['"' + key + '":' + value for key, value in LEARNING_PROGRAM_INFO.items()])

print(LEARNING_PROGRAM_KEYS, LEARNING_PROGRAM_ALL_INFO)


def json_parse(s: str):
    try:
        return loads("{" + s.split("{")[1].split("}")[0].strip() + "}")
    except:
        pass


class StudentMemory:

    def __init__(self, openai_api_key, user_id=None, username=''):
        # super().__init__()]
        self.name = username
        self.user_id = user_id
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=CONTROL_TEMPERATURE,
                              openai_api_key=openai_api_key)
        self.history = ConversationSummaryBufferMemory(llm=self.llm, **memory_defaults, max_token_limit=700,
                                                       human_prefix=self.name)
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
        self.input = ""

    def update_progress(self, program):
        print('Running UPDATE prompt')
        response = await self.update_chain.apredict(
            program_advance=learning_prorgam[program]["name"],
            history=self.history.load_memory_variables({})['history'],
            total_progress=self.progress["general"],
            program_progress=self.progress[program]["progress"],
            program_done=f'Учень успішно завершив {self.progress[program]["lesson"]} уроків із {len(learning_prorgam[program]["lessons"]) + 1} программи',
        )
        response_dict = json_parse(response)
        if not response_dict:
            print(f'ERROR parsing output: {response}')
            return
        print(response_dict)
        self.progress["general"] = response_dict["general_progress"]
        self.progress[program] = response_dict["program_progress"]
        print("Student progress updated")

    def process_inactivity_info(self, is_student_inactive):
        if is_student_inactive:
            if self.idle_wait_next_phrase:
                self.student_inactive_info = f"Пройшло {self.idle_check_time} секунд паузи між подачею інформації, можна продовжувати і розповісти учню наступний блок"
                self.idle_wait_next_phrase=False
                self.idle_check_time=IDLE_INACTIVITY_DEFAULT
            else:
                self.student_inactive_info = f"Пройшло {self.idle_check_time} секунд після того, як учню було задане питання, але він не відповідає. Треба привернути його увагу"
        else:
            self.student_inactive_info = ""


class OwlChat:

    def __init__(self, openai_api_key):

        self.llm_CONTROL = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=CONTROL_TEMPERATURE,
                                      openai_api_key=openai_api_key)
        self.llm_TUTOR = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=TUTOR_TEMPERATURE,
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
        async def do_plan(next_input):
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
            )
            response_dict = json_parse(response)
            if not response_dict:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            if response_dict["current_program"] in learning_prorgam.keys():
                print(f'Program {response_dict["current_program"]} selected')
                student.current_program = response_dict["current_program"]
            else:
                print("Program not selected")

            return response_dict["response"]

        async def do_tutor(next_input):
            current_progress = student.progress[student.current_program] if student.current_program else None
            response = await self.tutor_chain.apredict(
                current_program=learning_prorgam[student.current_program][
                    "name"] if student.current_program else "Програму навчання ще не обрано",
                lesson_topic=learning_prorgam[student.current_program][
                    current_progress["lesson"]] if student.current_program else "Програму навчання ще не обрано",
                progress_on_program=current_progress[
                    "progress"] if student.current_program else "Програму навчання ще не обрано",
                total_progress=student.progress["general"],
                name=student.name,
                history=student.history.load_memory_variables({})['history'],
                student_inactive_info=student.student_inactive_info,
                input=student.input,
            )
            response_dict = json_parse(response)
            if not response_dict:
                print(f'ERROR parsing output: {response}')
                return
            print(response_dict)
            student.history.save_context({"input": student.input}, {"output": response_dict["response"]})
            student.idle_check_time = response_dict["idle_check_time"]
            student.idle_wait_next_phrase = response_dict["idle_wait_next_phrase"]
            if response_dict["student_advance"] == "True":
                print('LESSON COMPLETED!')
                student.progress[student.current_program]["lesson"] += 1
                student.update_progress(student.current_program)
                pass
            return response_dict["response"]

        async def do_relax(next_input):
            response = await self.relax_chain.apredict(
                lesson_topic=learning_prorgam[student.current_program][student.progress[student.current_program]],
                progress_on_program=student.progress[student.current_program]["progress"],
                total_progress=student.progress["general"],
                name=student.name,
                history=student.history.load_memory_variables({})['history'],
                student_inactive_info=student.student_inactive_info,
                input=student.input,

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
        student.process_inactivity_info(is_student_inactive)
        print(f"chat with {student.user_id, student.name}, input={student.input}")
        control_response = await self.control_chain.apredict(
            current_program=learning_prorgam[student.current_program][
                "name"] if student.current_program else "Програму навчання ще не обрано",
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
        next_input, next_step = control_response_dict['next_input'], control_response_dict['next_step']

        # student.though_history.save_context({"input": student.input}, {"output": thought})
        if next_step == "RELAX":
            response = await do_relax(next_input)
        elif next_step == "TUTOR" and student.current_program is not None:
            response = await do_tutor(next_input)
        elif next_step == "PLAN":
            response = await do_plan(next_input)
        else:
            print("ERROR IN CONTROL PROMPT REASONING")
            response = await do_plan(next_input)

        return response
