# Lab 2 — Project Structure & File Ownership

> Dataset: CIFAR-10  
> Runtime: local computer with NVIDIA GPU  
> Workflow: reusable Python scripts + notebooks for exploration and analysis  
> Status: proposed structure; this document defines the plan, not implemented code.

## 1. Repository tree

    Lab2_DL_Group4/
    ├── README.md
    ├── main.py
    ├── requirements.txt
    ├── .gitignore
    ├── configs/
    │   ├── README.md
    │   ├── E1_vgg16.yaml
    │   ├── E2_densenet121.yaml
    │   ├── E3_learning_rate.yaml
    │   ├── E4_batch_size.yaml
    │   └── E5_finetuning.yaml
    ├── data/
    │   └── README.md
    ├── src/
    │   ├── README.md
    │   ├── data.py
    │   ├── models.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── utils.py
    │   └── predict.py
    ├── notebooks/
    │   ├── README.md
    │   ├── 01_dataset_exploration.ipynb
    │   └── 02_results_analysis.ipynb
    ├── tests/
    │   └── README.md
    ├── outputs/
    │   ├── README.md
    │   ├── experiment_summary.csv
    │   └── figures/
    ├── reports/
    │   ├── Lab2_Report.md
    │   └── sections/
    │       ├── 01_data.md
    │       ├── 02_model.md
    │       ├── 03_training.md
    │       ├── 04_evaluation.md
    │       ├── 05_experiments.md
    │       └── 06_integration.md
    └── docs/
        ├── 00_Lab2_Planning.md
        ├── ASSIGNMENT_LAB2.md
        ├── 01_Project_Structure.md
        ├── TEAM_CONTRACT.md
        ├── FILE_OWNERS.md
        ├── GIT_WORKFLOW.md
        └── REVIEW_STATUS.md

This is the intended structure. Files are not considered implemented merely because they appear in this tree.

## 2. File ownership

| Path | Responsibility | Owner |
|---|---|---|
| configs/ | Experiment configurations | TV5 |
| data/README.md | Data acquisition notes; downloaded data stays local | TV1 |
| src/data.py | CIFAR-10 transforms, split, DataLoaders | TV1 |
| notebooks/01_dataset_exploration.ipynb | Inspect sample images, labels, shapes and preprocessing | TV1 |
| src/models.py | Pre-trained models, 10-class classifier, freeze policy | TV2 |
| src/train.py | Training/validation loop, checkpoint selection | TV3 |
| src/evaluate.py | Evaluate selected checkpoint | TV4 |
| notebooks/02_results_analysis.ipynb | Analyze measured metrics and plots | TV4 |
| outputs/experiment_summary.csv | Actual configurations and measured metrics | TV5 |
| outputs/figures/ | Figures selected for report | TV4 |
| src/utils.py | Shared seed/config/metric helpers | TV6 |
| src/predict.py | Optional single-image inference demonstration | TV6 |
| main.py | Entry point coordinating modules | TV6 |
| README.md, requirements.txt, .gitignore | Setup, commands and repository hygiene | TV6 |
| reports/sections/ | Individual report sections | TV1–TV6 |
| docs/ | Planning, contracts, ownership and review status | TV6 coordinates |
| tests/ | Module tests | Each module owner; TV6 coordinates |

## 3. Module interface contract

These are logical responsibilities, not final Python function signatures. Agree on exact signatures and return types before implementation.

### TV1 — Data
- Provide a documented entry point for train, validation and test DataLoaders.
- Report class names and class count (10).
- Document transforms, normalization, split seed and loader settings.
- Keep test data out of hyperparameter/model selection.

### TV2 — Model
- Provide a model factory for architecture, pretrained weights, class count and freeze/fine-tune options.
- Output logits with shape [batch_size, 10].
- Report total and trainable parameters.
- Do not add Softmax before CrossEntropyLoss.

### TV3 — Training
- Consume model, loaders, optimizer/loss settings and run metadata.
- Record train/validation loss and accuracy by epoch.
- Save checkpoint selected using validation performance.
- Return history and checkpoint metadata/path.

### TV4 — Evaluation
- Evaluate the selected checkpoint on test data after model selection.
- Return measured metrics and create plots from recorded results.
- Do not use test metrics to select hyperparameters.

### TV5 — Experiments
- Give each run a unique ID/name and explicit configuration.
- Link measured metrics to the configuration that produced them.
- Prefer controlled comparisons that change one primary factor at a time.

### TV6 — Integration
- Coordinate shared interfaces and verify the environment.
- Keep main.py as an orchestrator rather than duplicating module logic.
- Ensure README commands match implemented interfaces.

## 4. Proposed experiment plan

These IDs are planning suggestions, not a number of experiments mandated by the assignment.

| ID | Purpose | Main variable |
|---|---|---|
| E1 | VGG16 baseline | Baseline architecture |
| E2 | DenseNet121 comparison | Architecture |
| E3 | Learning-rate comparison | Learning rate |
| E4 | Batch-size comparison | Batch size |
| E5 | Fine-tuning comparison | Freeze/unfreeze strategy |

Choose exact learning rates, batch sizes, epoch count and fine-tuning depth after checking the GPU and VRAM. Do not add metric values until experiments have actually run.

## 5. Output and Git policy

- CIFAR-10 downloads and cache stay local; do not commit them.
- Do not commit virtual environments, credentials, or .env files.
- Keep TensorBoard logs local by default; save selected screenshots/figures for the report.
- Do not push large checkpoints without team agreement; use an agreed artifact location.
- Commit result tables and report figures only after verifying that all values are measured.
- Keep the main report concise and link member-owned report sections.

## 6. Branch and PR plan

Planned work branches:
- feature/data — TV1
- feature/model — TV2
- feature/training — TV3
- feature/evaluation — TV4
- feature/experiments — TV5
- feature/integration — TV6

Create these from the agreed main only after the assignment and structure documents have been reviewed. This documentation change does not create any of those six branches.

Each PR should state changed files, verification steps and actual output; update the relevant report section; mention interface changes and known limitations; and receive review before merge.

## 7. Definition of ready

- [ ] All six members accept their assignment.
- [ ] Assignment and structure documents agree.
- [ ] Dependent members agree on module interfaces.
- [ ] GPU, Python and PyTorch installation approach have been checked.
- [ ] Experiment settings are configurable rather than hard-coded.
- [ ] Everyone understands validation is for model selection and test is for final evaluation.
