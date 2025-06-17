import json
import csv
import os
import time
import random
import argparse
from tqdm import tqdm
from initialize import read_malicious_instructions, read_task_instructions, generate_jailbreak_prompts
from fitness import calculate_total_fitness_score
from selector import select_next_generation
from crossover import crossover
from mutation import mutation


def genetic_algorithm(malicious_file_path, task_file_path, target_model, population_size=10, max_iterations=10):
    print("EvoJail started...")

    # Load malicious instructions
    malicious_instructions = read_malicious_instructions(malicious_file_path)
    print("Number of malicious instructions:", len(malicious_instructions))

    # Load task instructions
    task_instructions = read_task_instructions(task_file_path, population_size)
    print("Number of task instructions:", len(task_instructions))

    # Create timestamped log filenames
    timestamp = int(time.time())
    log_dir = "./results"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    detailed_log_filename = f"./results/EvoJail_{target_model}_population_size_{population_size}_max_iterations_{max_iterations}_{timestamp}_detailed.json"
    summary_log_filename = f"./results/EvoJail_{target_model}_population_size_{population_size}_max_iterations_{max_iterations}_{timestamp}_summary.csv"

    # Initialize logs
    detailed_log = []
    summary_log = []

    # Write CSV header if file doesn't exist
    if not os.path.exists(summary_log_filename):
        with open(summary_log_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["Goal", "Jailbreak", "Response", "is_success"])
            writer.writeheader()
    try:
        # Iterate over each malicious instruction
        for malicious_instruction in tqdm(malicious_instructions, desc="Processing malicious instructions"):
            # Initialize population
            jailbreak_prompts, initial_input_tokens_used, initial_output_tokens_used = generate_jailbreak_prompts(
                malicious_instruction, task_instructions)

            total_input_tokens = initial_input_tokens_used
            total_output_tokens = initial_output_tokens_used

            iteration = 0
            best_attack_score = 0
            best_jailbreak = None
            best_response = None
            is_success = False  # Success flag

            # Evolution loop
            while iteration < max_iterations:
                print("Current iteration:", iteration)
                iteration_log = {
                    "malicious_instruction": malicious_instruction,
                    "iteration": iteration,
                    "jailbreak_prompts": jailbreak_prompts,
                    "initial_input_tokens_used": initial_input_tokens_used if iteration == 0 else 0,
                    "initial_output_tokens_used": initial_output_tokens_used if iteration == 0 else 0,
                }

                # Evaluate fitness
                (
                    total_fitness_scores, jailbreak_responses, res_input_tokens_used, res_output_tokens_used,
                    attack_success_normalized_scores, reasons, attack_success_flags, eval_input_tokens_used,
                    eval_output_tokens_used, diversity_scores
                ) = calculate_total_fitness_score(jailbreak_prompts, target_model)

                print("Total fitness scores:", total_fitness_scores)

                # Update token usage
                total_input_tokens += res_input_tokens_used + eval_input_tokens_used
                total_output_tokens += res_output_tokens_used + eval_output_tokens_used

                # If any jailbreak is successful, record and break
                if any(attack_success_flags):
                    successful_indices = [i for i, flag in enumerate(attack_success_flags) if flag]
                    successful_index = random.choice(successful_indices)
                    best_jailbreak = jailbreak_prompts[successful_index]
                    best_response = jailbreak_responses[successful_index]
                    is_success = True

                    iteration_log.update({
                        "jailbreak_responses": jailbreak_responses,
                        "res_input_tokens_used": res_input_tokens_used,
                        "res_output_tokens_used": res_output_tokens_used,
                        "attack_success_normalized_scores": attack_success_normalized_scores,
                        "reasons": reasons,
                        "attack_success_flags": attack_success_flags,
                        "eval_input_tokens_used": eval_input_tokens_used,
                        "eval_output_tokens_used": eval_output_tokens_used,
                        "diversity_scores": diversity_scores,
                        "total_fitness_scores": total_fitness_scores,
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                    })
                    detailed_log.append(iteration_log)
                    break

                # Track the best performing prompt so far
                current_best_score = max(attack_success_normalized_scores)
                if current_best_score > best_attack_score:
                    best_attack_score = current_best_score
                    best_index = attack_success_normalized_scores.index(current_best_score)
                    best_jailbreak = jailbreak_prompts[best_index]
                    best_response = jailbreak_responses[best_index]

                # Update log with fitness evaluation details
                iteration_log.update({
                    "jailbreak_responses": jailbreak_responses,
                    "res_input_tokens_used": res_input_tokens_used,
                    "res_output_tokens_used": res_output_tokens_used,
                    "attack_success_normalized_scores": attack_success_normalized_scores,
                    "reasons": reasons,
                    "attack_success_flags": attack_success_flags,
                    "eval_input_tokens_used": eval_input_tokens_used,
                    "eval_output_tokens_used": eval_output_tokens_used,
                    "diversity_scores": diversity_scores,
                    "total_fitness_scores": total_fitness_scores,
                })

                # Selection phase
                elite_prompts, crossover_mutation_prompts, next_generation = select_next_generation(jailbreak_prompts,
                                                                                                    total_fitness_scores)

                # Crossover operation
                crossover_prompts, cross_input_tokens_used, cross_output_tokens_used = crossover(crossover_mutation_prompts,
                                                                                                 next_generation,
                                                                                                 malicious_instruction)
                total_input_tokens += cross_input_tokens_used
                total_output_tokens += cross_output_tokens_used

                # Mutation operation
                mutated_prompts, mutation_input_tokens_used, mutation_output_tokens_used = mutation(crossover_prompts,
                                                                                                    malicious_instruction)
                total_input_tokens += mutation_input_tokens_used
                total_output_tokens += mutation_output_tokens_used

                # Update population
                jailbreak_prompts = elite_prompts + mutated_prompts

                # Log the genetic operations
                iteration_log.update({
                    "elite_prompts": elite_prompts,
                    "crossover_mutation_prompts": crossover_mutation_prompts,
                    "cross_input_tokens_used": cross_input_tokens_used,
                    "cross_output_tokens_used": cross_output_tokens_used,
                    "mutation_input_tokens_used": mutation_input_tokens_used,
                    "mutation_output_tokens_used": mutation_output_tokens_used,
                    "next_generation": jailbreak_prompts,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                })

                detailed_log.append(iteration_log)
                iteration += 1

            # Record summary for each malicious instruction
            summary_log.append({
                "Goal": malicious_instruction,
                "Jailbreak": best_jailbreak,
                "Response": best_response,
                "is_success": is_success,
            })

            # Append summary to CSV
            with open(summary_log_filename, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=["Goal", "Jailbreak", "Response", "is_success"])
                for row in summary_log:
                    row = {key: str(value) if value is not None else "" for key, value in row.items()}
                    writer.writerow(row)

            summary_log.clear()

    except Exception as e:
        print(f"Execution interrupted due to an error: {e}")

    finally:
        # Save detailed log as JSON
        with open(detailed_log_filename, 'w', encoding='utf-8-sig') as f:
            json.dump(detailed_log, f, indent=4)

        print(f"Detailed log saved to {detailed_log_filename}")
        print(f"Summary log saved to {summary_log_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EvoJail genetic algorithm.")

    parser.add_argument("--malicious_file_path", type=str, default="./data/data_harmful-behaviors.csv",
                        help="Path to the CSV file containing malicious instructions.")
    parser.add_argument("--task_file_path", type=str, default="./data/task_instructions.json",
                        help="Path to the JSON file containing task instructions.")
    parser.add_argument("--target_model", type=str, default="gpt-4o-mini",
                        help="Target model name to evaluate against.")
    parser.add_argument("--population_size", type=int, default=10,
                        help="Initial population size for the genetic algorithm.")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of iterations to run.")

    args = parser.parse_args()

    # Call the main algorithm function
    genetic_algorithm(
        malicious_file_path=args.malicious_file_path,
        task_file_path=args.task_file_path,
        target_model=args.target_model,
        population_size=args.population_size,
        max_iterations=args.max_iterations
    )
