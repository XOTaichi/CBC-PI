import argparse

def define_arguments_prompted(parser):
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Validate data without loading a model or making API calls")
    parser.add_argument("--max-cases", dest="max_cases", type=int, default=0, help="Maximum cases per attack type; 0 runs all")
    parser.add_argument("--model_type", help="Type of the model to evaluate")
    parser.add_argument("--model_name", help="Name of the model to evaluate", default='', required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--only_first_step", default=False, action='store_true', help="Only predict the first step")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Use existing dataset")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only get score without inference")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--pe", type=str)
    parser.add_argument("--guardrail", type=str)
    parser.add_argument("--sentinel_path", type=str, default="../models/datasentinel_detector")
    parser.add_argument("--struq_family", type=str, default="qwen2")

def define_arguments_finetuned(parser):
    parser.add_argument("--model_name", help="Name of the model to evaluate", required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--only_first_step", action='store_true', help="Only predict the first step")

def parse_arguments(agent_type=None):
    parser = argparse.ArgumentParser()
    if agent_type == "prompted":
        define_arguments_prompted(parser)
    elif agent_type == "finetuned":
        define_arguments_finetuned(parser)
    args = parser.parse_args()
    if getattr(args, "max_cases", 0) < 0:
        parser.error("--max-cases must be non-negative")
    params = args.__dict__
    return params
