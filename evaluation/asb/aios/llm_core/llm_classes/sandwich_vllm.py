import os
from .base_llm import BaseLLM
import time

# could be dynamically imported similar to other models

from pyopenagi.utils.chat_template import Response
from ...utils.utils import get_from_env

from transformers import AutoTokenizer

from huggingface_hub import login
import re
import vllm

class Sandwich_vLLM(BaseLLM):

    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 4096,
                 log_mode: str = "console"):

        super().__init__(llm_name,
                         max_gpu_memory,
                         eval_device,
                         max_new_tokens,
                         log_mode)

    def load_llm_and_tokenizer(self) -> None:
        """ fetch the model from huggingface and run it """
        # self.available_gpus = list(self.max_gpu_memory.keys())
        # self.gpu_nums = len(self.available_gpus)
        try:
            import vllm
        except ImportError:
            raise ImportError(
                "Could not import vllm python package. "
                "Please install it with `pip install vllm`."
            )

        """ only casual lms for now """
        self.model = vllm.LLM(
            model = self.model_name,
            # download_dir = get_from_env("HF_HOME"),
            tensor_parallel_size = int(os.getenv("ASB_TENSOR_PARALLEL_SIZE", "2")),
            max_model_len=int(os.getenv("ASB_MAX_MODEL_LEN", "16192")),
            gpu_memory_utilization=float(os.getenv("ASB_GPU_MEMORY_UTILIZATION", "0.8"))
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.sampling_params = vllm.SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=self.max_new_tokens,
        )

    def extract_answer(self, text):
        match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            think_pos = text.lower().find('</think>')
            if think_pos != -1:
                return text[think_pos + len('</think>'):].strip()
            else:
                return text.strip()

    def process(self,
                agent_process,
                temperature=0.1) -> None:
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level = "executing"
        )

        messages = agent_process.query.messages
        tools = agent_process.query.tools
        message_return_type = agent_process.query.message_return_type

        if tools:
            print("Calling Using Tools")
            format_enforcement = (
                '\n\nYou must strictly follow the output format. '
                'Output ONLY a raw JSON list containing the plan. '
                'Format: [{"message": "description...", "tool_use": ["tool_name", ...]}]. '
                'Do NOT use Markdown block (```json). Do NOT add explanations. '
                'The output must start with [ and end with ].'
            )
            for msg in messages:
                if 'content' in msg and isinstance(msg['content'], str):
                    msg['content'] = msg['content'].replace(format_enforcement, "").strip()
            messages[-1]["content"] += f'\nRemember, {messages[0]["content"]}'
            messages = self.tool_calling_input_format(messages, tools)
            # print(messages)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                # tools = tools,
                tokenize = False,
                add_generation_prompt=True
            )
            # prompt = self.parse_messages(messages)
            sampling_params = vllm.SamplingParams(
                temperature=0.1,
                top_p=0.95,
                max_tokens=4096,
            )
            try:
                response = self.model.generate(
                    prompt, sampling_params
                )
                result = self.extract_answer(response[0].outputs[0].text)
            except Exception as e:
                result = "Error"
            # print(response)


            print(f"***** Result: {result} *****")

            tool_calls = self.parse_tool_calls(
                result
            )
            if tool_calls:
                agent_process.set_response(
                    Response(
                        response_message = None,
                        tool_calls = tool_calls
                    )
                )
            else:
                agent_process.set_response(
                    Response(
                        response_message = result
                    )
                )

        else:
            print("Calling Thinking plans")
            format_enforcement = (
                '\n\nYou must strictly follow the output format. '
                'Output ONLY a raw JSON list containing the plan. '
                'Format: [{"message": "description...", "tool_use": ["tool_name", ...]}]. '
                'Do NOT use Markdown block (```json). Do NOT add explanations. '
                'The output must start with [ and end with ].'
            )
            messages[-1]["content"] += format_enforcement
            messages[-1]["content"] += f'\nRemember, {messages[0]["content"]}'
            # print(messages)
            # 3. 接下来再正常处理 prompt 生成
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # print(messages)

            # last_msg_content = messages[-1]['content']
            # messages[-1]['content'] = last_msg_content + format_enforcement
            # prompt = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize = False,
            #     add_generation_prompt=True

            # )
            # print(prompt)
            # prompt = prompt + "<|im_start|>assistant\n"
            # prompt = self.parse_messages(messages)
            sampling_params = vllm.SamplingParams(
                temperature=0.1,
                top_p=0.95,
                max_tokens=4096,
            )
            try:
                response = self.model.generate(
                    prompt, sampling_params
                )
                result = self.extract_answer(response[0].outputs[0].text)
            except Exception as e:
                result = "Error"

            # result =  self.extract_answer(response[0].outputs[0].text)
            # print(response)
            # print(f"VLLM MODEL RESP: {response[0].outputs[0].text}")
            if message_return_type == "json":
                result = self.parse_json_format(result)

            agent_process.set_response(
                Response(
                    response_message=result
                )
            )

        agent_process.set_status("done")

        agent_process.set_end_time(time.time())
