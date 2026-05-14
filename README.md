# EvoJail: Evolutionary Diverse Jailbreak Prompt Generation for Large Language Models

## 1. Directory Structure

```text
EvoJail/
├── main.py                     # Main workflow (genetic algorithm entry point)
├── initialize.py               # Data loading & initial jailbreak generation
├── fitness.py                  # Response generation, attack evaluation, diversity, and fitness calculation
├── selector.py                 # Selection operator
├── crossover.py                # Crossover operator
├── mutation.py                 # Mutation operator
├── language_models.py          # API wrapper (DeepSeek / OpenAI)
├── prompt_templates.py         # Prompt templates
├── config.py                   # API key and model path configuration
├── data/
│   ├── data_harmful-behaviors.csv
│   └── task_instructions.json
└── results/                    # Automatically generated after execution
```

## 2. Environment Setup (Quick Reproduction)

### 2.1 Python Version

Recommended: **Python 3.10+**

### 2.2 Install Dependencies

```bash
pip install openai numpy scikit-learn tqdm
```

### 2.3 Configure API Keys

Edit `config.py`:

* `DEEPSEEK_API_KEY`: Used for initialization, crossover, mutation, and related calls.
* `CHATGPT_API_KEY`: Used for target model responses and safety evaluation.

Example (replace with your actual keys):

```python
DEEPSEEK_API_KEY = 'your_deepseek_key'
CHATGPT_API_KEY = 'your_openai_key'
```

## 3. Data Format Description

### 3.1 Malicious Target File (CSV)

Default path: `./data/data_harmful-behaviors.csv`

The file must contain at least the following column:

* `Goal`: Each row represents one attack target.

### 3.2 Task Instruction File (JSON)

Default path: `./data/task_instructions.json`

The file should be a list, where each item contains at least:

* `instruction`: A normal task description used to wrap the malicious target.

## 4. Quick Start

Run with default data:

```bash
python main.py
```

Common usage example:

```bash
python main.py \
  --malicious_file_path ./data/data_harmful-behaviors.csv \
  --task_file_path ./data/task_instructions.json \
  --target_model gpt-4o-mini \
  --population_size 10 \
  --max_iterations 10
```

### Parameter Description

* `--malicious_file_path`: Path to the malicious target CSV file.
* `--task_file_path`: Path to the task instruction JSON file.
* `--target_model`: Name of the target model used for attack/evaluation.
* `--population_size`: Initial population size.
* `--max_iterations`: Maximum number of evolution iterations.

## 5. Output Results

After execution, timestamped result files will be generated under `results/`:

### 1. `*_detailed.json`

Contains:

* Candidate prompts for each iteration
* Model responses
* Scores and evaluations
* Crossover and mutation results
* Accumulated token consumption

### 2. `*_summary.csv`

Contains:

* The best jailbreak prompt for each malicious target
* Corresponding model response
* Whether the attack succeeded (`is_success`)

