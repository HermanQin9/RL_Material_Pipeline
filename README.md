# Machine Learning Pipeline with PPO RL 

### Author: Herman Qin

## Project Overview

This project implements a Reinforcement Learning (RL)-based automated Machine Learning (AutoML) pipeline. Utilizing the Proximal Policy Optimization (PPO) algorithm, the pipeline automatically selects and tunes preprocessing methods, feature engineering techniques, and machine learning models to optimize predictive performance formation energy.

## Repository Structure

```
.
├── config.py            # Global configuration (paths, API keys, hyperparameters)
├── nodes.py             # Node base class and all node implementations
├── pipeline.py          # Main pipeline (run_pipeline)
├── methods/             # Data and model methods
│   ├── data_methods.py  # Data processing functions
│   └── model_methods.py # Model training and prediction functions
├── env/                 # Environment-related code
│   ├── __init__.py
│   ├── pipeline_env.py  # PipelineEnv class definition
│   └── utils.py         # Helper functions (_get_obs, _compute_action_mask, reward calculation)
├── ppo/                 # PPO algorithm components
│   ├── __init__.py      #
│   ├── policy.py        # PPOPolicy implementation
│   ├── buffer.py        # RolloutBuffer (stores transitions)
│   ├── utils.py         # Utility functions (GAE, loss calculation)
│   └── trainer.py       # Training loop and optimization logic
├── scripts/             # Command-line scripts
│   ├── train_ppo.py     # PPO training script
│   ├── eval_ppo.py      # Policy evaluation script
│   ├── example_usage.py # Example usage demonstration
│   ├── debug_pipeline.py # Pipeline debugging utilities
│   ├── train_ppo_4k.py  # 4K dataset PPO training
│   ├── train_ppo_safe.py # Safe PPO training with error handling
│   ├── generate_4k_data.py # 4K dataset generation
│   ├── fix_4k_data.py   # Data fixing utilities
│   ├── main.py          # Main execution script
│   ├── run.py           # Alternative run script
│   ├── analysis/        # Analysis and visualization scripts
│   │   ├── analyze_ppo_results.py # PPO results analysis
│   │   └── reward_analysis.py     # Reward function analysis
│   └── debug/           # Debugging utilities
│       └── check_training_mode.py # Training mode checker
├── tests/               # Unit tests and validation scripts
│   ├── test_all_files.py
│   ├── test_all_models.py
│   ├── test_and_train_ppo.py
│   ├── test_components.py
│   ├── test_pipeline.py
│   ├── test_ppo.py
│   ├── test_data_methods.py
│   ├── test_4k_data.py  # 4K dataset testing
│   ├── test_ppo_simple.py # Simple PPO testing
│   ├── validate_ppo_training.py # PPO training validation
│   ├── extended_ppo_validation.py # Extended validation
│   └── simplified_ppo_validation.py # Simplified validation
├── utils/               # Utility functions
│   └── pipeline_utils.py # Pipeline utilities
├── notebooks/           # Jupyter notebooks
│   ├── PPO_Testing_and_Debugging.ipynb
│   └── _setup.ipynb
├── docs/                # Documentation files
│   ├── COMPLIANCE_ANALYSIS.md
│   ├── IMPORT_FIX_REPORT.md
│   ├── PROJECT_ORGANIZATION.md
│   ├── STATUS_REPORT.md
│   ├── STATUS_UPDATE.md
│   ├── TESTING_REPORT.md
│   ├── VALIDATION_SUMMARY.md
│   ├── PPO_VALIDATION_REPORT.md  # PPO validation results
│   └── DATASET_INFO.md           # Dataset information
├── data/                # Data storage
│   ├── raw/             # Original datasets
│   └── processed/       # Processed datasets
├── models/              # Trained model checkpoints cache
├── logs/                # Training and evaluation logs
└── dash_app/            # Visualization dashboard application
    └── data/            # Dashboard-specific data
```

## Key Components

### Nodes and Methods

The pipeline consists of the following nodes and methods:

| Node | Name            | Available Methods               |
| ---- | --------------- | ------------------------------- |
| N0   | Data\_Fetch     | api                             |
| N1   | Impute          | mean, median, knn, none         |
| N2   | Feature\_Matrix | default                         |
| N3   | Feat\_Select    | none, variance, univariate, pca |
| N4   | Scale           | std, robust, minmax, none       |
| N5   | Learner         | rf, gbr, lgbm, xgb, cat         |
| N6   | END             | Pipeline termination node       |

### PPO Reinforcement Learning

The PPO algorithm automatically selects:

* **Nodes**: The sequence of steps in the pipeline.
* **Methods**: Specific methods at each node.
* **Hyperparameters**: Optimal parameter settings for each method.

## Installation

Create and activate the environment using conda:

```bash
conda env create -f environment.yml
conda activate base
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Training PPO

**Standard training (200 samples):**
```bash
python scripts/train_ppo.py
```

**4K dataset training:**
```bash
$env:PIPELINE_TEST="0"; python scripts/train_ppo_4k.py
```

**Safe training with error handling:**
```bash
python scripts/train_ppo_safe.py
```

### Data Generation and Testing

**Generate 4K dataset:**
```bash
python scripts/generate_4k_data.py
```

**Test 4K dataset:**
```bash
python tests/test_4k_data.py
```

**Validate PPO training:**
```bash
python tests/validate_ppo_training.py
```

### Analysis and Visualization

**Analyze PPO results:**
```bash
python scripts/analysis/analyze_ppo_results.py
```

**Reward function analysis:**
```bash
python scripts/analysis/reward_analysis.py
```

### Example Usage

```bash
python scripts/example_usage.py
```

### Debugging

**Check training mode:**
```bash
python scripts/debug/check_training_mode.py
```

## Testing

### Unit Tests
Run unit tests from the project root:

```bash
python tests/test_pipeline.py
python tests/test_ppo.py
python tests/test_components.py
```

### 4K Dataset Testing
```bash
python tests/test_4k_data.py
```

### PPO Validation
```bash
python tests/validate_ppo_training.py
python tests/extended_ppo_validation.py
```

### Jupyter Notebooks
Run the included notebooks in the `notebooks/` directory:

* `PPO_Testing_and_Debugging.ipynb`
* `_setup.ipynb`

### Visualization (Hasn't been built)

Launch the dashboard app:

```bash
cd dash_app
python app.py
```

## Testing

Unit tests are available in `pipeline_env_unit_tests.ipynb` to verify the functionality of pipeline components and the Gym environment.

## Logs and Models

Checkpoints and logs are stored in the `models` and `logs` directories, respectively, for tracking experiment results and trained models.

---

For further details, refer to `Project_Summary1.docx`.
