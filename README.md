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
│   └── debug_pipeline.py # Pipeline debugging utilities
├── tests/               # Unit tests
│   ├── test_all_files.py
│   ├── test_all_models.py
│   ├── test_and_train_ppo.py
│   ├── test_components.py
│   ├── test_pipeline.py
│   ├── test_ppo.py
│   └── test_data_methods.py
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
│   └── VALIDATION_SUMMARY.md
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

```bash
python scripts/train_ppo.py
```

### Example Usage

```bash
python scripts/example_usage.py
```

### Evaluation

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
