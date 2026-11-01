import os
import json
import importlib
from tqdm import tqdm
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
from src.output_parsing import evaluate_output_prompted


def predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=False):
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)

    input = item['User Instruction']

    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return output

def main(params):
    print({k: ("<set>" if v else None) if k == "api_key" else v for k, v in params.items()})
    output_dir_params = params["output_dir"]
    if not params["only_get_score"] and not params.get("dry_run"):
        from src.models import MODELS, SandwichModel, Instruction_cotModel, InstructionModel, LocateModel, SentinelModel, Struq, PromptArmorModel
        model_class = MODELS[params['model_type']](params)
        if params.get("pe", "") == "sandwich":
            model_class = SandwichModel(model_class)
        elif params.get("pe", "") == "instruction":
            model_class = InstructionModel(model_class)
        elif params.get("pe", "") == "instruction_cot":
            model_class = Instruction_cotModel(model_class)
        elif params.get("guardrail", "") == "locate":
            model_class = LocateModel(ft_path=params["sentinel_path"], base_model=model_class)
        elif params.get("guardrail", "") == "sentinel":
            model_class = SentinelModel(ft_path=params["sentinel_path"], base_model=model_class)
        elif params.get("guardrail", "") == "promptarmor":
            model_class = PromptArmorModel(base_model=model_class)

    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    if params.get("pe", ""):
        head_fix = params["pe"]
    elif params.get("guardrail", ""):
        head_fix = params["guardrail"]
    else:
        head_fix = params['model_type']
    output_model_name = params['model_name'].split('/')[-1]
    print(output_model_name)
    output_dir = f"{output_dir_params}/prompted_{head_fix}_{output_model_name}_{params['prompt_type']}"
    if not params.get("dry_run"):
        os.makedirs(output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    # data_dir = './data'

    if params.get("dry_run"):
        for attack in ('dh', 'ds'):
            path = Path(data_dir) / f"test_cases_{attack}_{params['setting']}.json"
            rows = json.loads(path.read_text())
            if params.get('max_cases', 0):
                rows = rows[:params['max_cases']]
            required = ('User Tool', 'Attacker Tools', 'User Instruction', 'Tool Parameters', 'Tool Response', 'Thought', 'Attacker Instruction')
            if not rows or any(any(k not in row for k in required) for row in rows):
                raise ValueError(f"Invalid input: {path}")
            print(f"{attack}: {len(rows)} cases; output: {output_dir}")
        return

    tool_dict = get_tool_dict()

    error_count = 0
    output_files = {}
    for attack in ['dh', 'ds']:
        file_name = f"test_cases_{attack}_{params['setting']}.json"
        test_case_file = os.path.join(data_dir, file_name)
        output_file = os.path.join(output_dir, f"test_cases_{attack}_{params['setting']}.jsonl")
        output_files[attack] = output_file

        with open(test_case_file, 'r') as f:
            data = json.load(f)
        if params.get('max_cases', 0):
            data = data[:params['max_cases']]

        if params['use_cache']:
            existing_data_step1 = {}
            existing_data_step2 = {}
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f.readlines():
                        try:
                            tmp = json.loads(line)
                            if type(tmp['output']) == str and len(tmp['output']):
                                existing_data_step1[(tmp['User Instruction'], tmp['Tool Response'])] = tmp['output']
                            if type(tmp['output Step 2']) == str and len(tmp['output Step 2']):
                                existing_data_step2[(tmp['User Instruction'], tmp['Tool Response'], tmp['output'])] = tmp['output Step 2']
                        except:
                            continue
                print(f"Got {len(existing_data_step1)} existing outputs for the first step and {len(existing_data_step2)} existing outputs for the second step")

        if not params['only_get_score']:
            with open(output_file, 'w') as f:
                for item in tqdm(data):
                    try:
                        if params['use_cache'] and (item['User Instruction'], item['Tool Response']) in existing_data_step1:
                            # print("Use existing output for step 1")
                            output = existing_data_step1[(item['User Instruction'], item['Tool Response'])]
                        else:
                            output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict)
                        item['output'] = output

                        parsed_output = evaluate_output_prompted(item['output'], item['Attacker Tools'][0], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
                        item['eval'] = parsed_output['eval']
                        if item['eval'] == 'invalid':
                            item['invalid_reason'] = parsed_output['invalid_reason']

                        if not params['only_first_step'] and item['eval'] == 'succ' and attack == 'ds':
                            item["Attacker Tool Parameters"] = parsed_output["Attacker Tool Parameters"]
                            item["Attacker Tool Thought"] = parsed_output["Attacker Tool Thought"]
                            item["Attacker Tool Response"] = get_simulated_attacker_tool_response(item['Attacker Tools'][0], item["Attacker Tool Parameters"], tool_dict)

                            if params['use_cache'] and (item['User Instruction'], item['Tool Response'], item['output']) in existing_data_step2:
                                output = existing_data_step2[(item['User Instruction'], item['Tool Response'], item['output'])]
                                # print("Use existing output for step 2")
                            else:
                                output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=True)

                            item['output Step 2'] = output
                            parsed_output = evaluate_output_prompted(output, item['Attacker Tools'][1], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
                            item['eval Step 2'] = parsed_output['eval']
                            if item['eval Step 2'] == 'invalid':
                                item['invalid_reason  Step 2'] = parsed_output['invalid_reason']

                        f.write(json.dumps(item)+'\n')
                    except Exception as e:
                        # This block will execute if an exception occurs
                        error_count += 1
                        print(f"An error occurred: {e} in {output_file}")
                        # assert 1==0

    if error_count:
        raise RuntimeError(f"{error_count} cases failed; do not score incomplete results")
    scores = get_score(output_files)
    print(json.dumps(scores, indent=True))

if __name__ == "__main__":
    params = parse_arguments(agent_type="prompted")

    main(params)
