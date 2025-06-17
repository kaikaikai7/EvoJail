import csv
import json
from language_models import get_deepseek_response
from prompt_templates import PROMPT_FOR_OBFUSCATING_HARMFUL_OBJECTIVES


def read_malicious_instructions(file_path, max_rows=None):
    """Read malicious instructions from a CSV file."""
    malicious_instructions = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            # Stop reading if the maximum number of rows is reached
            if max_rows is not None and i >= max_rows:
                break
            malicious_instructions.append(row['Goal'])

    return malicious_instructions


def read_task_instructions(file_path, num_tasks):
    """Read task instructions from a JSON file and select the first `num_tasks` entries."""
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        task_data = json.load(file)

    # Select the first `num_tasks` task instructions
    selected_tasks = task_data[:num_tasks]

    return selected_tasks


def generate_jailbreak_prompts(malicious_instruction, task_instructions):
    """Generate initial jailbreak prompts by combining a malicious instruction with a list of task instructions."""
    initial_jailbreak_prompts = []
    initial_input_tokens_used = 0
    initial_output_tokens_used = 0

    # Iterate over each task instruction
    for index, task in enumerate(task_instructions, start=1):
        task_instruction = task['instruction']

        # Display current progress
        print("\n" + "=" * 50)
        print(f"Merging {index}/{len(task_instructions)}...")
        print(f"Malicious Instruction: {malicious_instruction}")
        print(f"Task Instruction: {task_instruction}")

        # Construct the prompt template for merging
        prompt_template = PROMPT_FOR_OBFUSCATING_HARMFUL_OBJECTIVES.format(
            task_instruction=task_instruction,
            malicious_instruction=malicious_instruction
        )

        # Call the model to generate a merged instruction
        result = get_deepseek_response(prompt_template)

        if result is not None:
            model, merged_instruction, input_tokens, output_tokens = result

            # Check if the model refused to respond
            if "I'm sorry" in merged_instruction or "I cannot" in merged_instruction or "I can't" in merged_instruction:
                print("Warning: Model refused to respond. Using fallback template to merge!")
                print("Model Response:", merged_instruction)
                merged_instruction = f"{task_instruction} Meanwhile, {malicious_instruction}"
                print("Merged Instruction:", merged_instruction)
            else:
                print("Merged Instruction:", merged_instruction)

            print("Input Tokens:", input_tokens)
            print("Output Tokens:", output_tokens)
            initial_input_tokens_used += input_tokens
            initial_output_tokens_used += output_tokens
        else:
            # If the request fails, use a fallback template
            merged_instruction = f"{task_instruction} Meanwhile, {malicious_instruction}"
            print("Warning: Request failed. Using fallback template to merge!")
            print("Merged Instruction:", merged_instruction)

        # Add the merged instruction to the list
        initial_jailbreak_prompts.append(merged_instruction)

    # Summary of initialization
    print("\n" + "=" * 50)
    print("Jailbreak prompt initialization completed.")
    print("Total input tokens used during initialization:", initial_input_tokens_used)
    print("Total output tokens used during initialization:", initial_output_tokens_used)
    print("=" * 50)

    # Return the merged prompts and token usage
    return initial_jailbreak_prompts, initial_input_tokens_used, initial_output_tokens_used


if __name__ == "__main__":
    malicious_instruction = read_malicious_instructions("./data/data_harmful-behaviors.csv", 1)[0]
    task_instructions = read_task_instructions("./data/task_instructions.json", 5)
    generate_jailbreak_prompts(malicious_instruction, task_instructions)
