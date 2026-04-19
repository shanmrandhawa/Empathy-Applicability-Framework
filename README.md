# Empathy Applicability Framework (EAF)

Code and data for the paper:
**"Empathy Applicability Modeling for General Health Queries"**
(ACL Findings, 2026)

## Overview

EAF is a theory-driven framework that anticipates patient empathetic needs by assessing the applicability of emotional reactions and interpretations in patient queries, enabling anticipatory empathy modeling in asynchronous healthcare communication.

## Requirements

- Python 3.8.10
- PyTorch 2.2.2
- Transformers 4.39.3
- Datasets 2.18.0
- Accelerate 0.28.0
- scikit-learn 1.3.2
- pandas 2.0.3
- numpy 1.24.4
- wandb 0.16.5

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── LICENSE
├── scripts/
│   ├── annotation/
│   │   ├── emotional_reaction_annotation_GPT.py  # GPT annotation script for Emotional Reactions
│   │   ├── interpretation_annotation_GPT.py      # GPT annotation script for Interpretations
│   │   ├── emotional_reaction_annotation_o1_without_framework.py   # o1 zero-shot annotation without EAF - EA
│   │   └── interpretation_annotation_o1_without_framework.py       # o1 zero-shot annotation without EAF - IA
│   └── training/
│       ├── ea_classifier.py                      # RoBERTa classifier for Emotional Reactions
│       └── ia_classifier.py                      # RoBERTa classifier for Interpretations
│       ├── traditional_baselines.py              # TF-IDF + Logistic Regression / LinearSVC
│       └── heuristics_o1_zeroshot_baselines.py   # Random / Always-Applicable / Always-Not / o1 evaluation
├── data/
│   ├── annotated_set/                            # EAF benchmark dataset
│   │   ├── Humans_annotations_1300.csv           # 1,300 queries annotated by two human annotators
│   │   └── GPT_annotations_1300.csv              # GPT-4o annotations (majority-voted, 5 passes)
│   ├── classifier_experiments/
│   │   ├── EA_train.csv                          # Human-consensus training split (EA)
│   │   ├── EA_eval.csv                           # Human-consensus eval split (EA)
│   │   ├── EA_test.csv                           # Human-consensus test split (EA)
│   │   ├── IA_train.csv                          # Human-consensus training split (IA)
│   │   ├── IA_eval.csv                           # Human-consensus eval split (IA)
│   │   ├── IA_test.csv                           # Human-consensus test split (IA)
│   │   ├── o1_zeroshot_EA_test.csv               # o1 zero-shot without EAF labeling of EA test set
│   │   ├── o1_zeroshot_IA_test.csv               # o1 zero-shot without EAF labeling of IA test set
│   │   ├── output_human_EA_test.csv              # Transformer (Human Supervised) classifier predictions on EA test set
│   │   ├── output_human_IA_test.csv              # Transformer (Human Supervised) classifier predictions on IA test set
│   │   ├── output_autonomous_EA_test.csv         # Transformer (Autonomous-GPT Supervised) classifier predictions on EA test set
│   │   ├── output_autonomous_IA_test.csv         # Transformer (Autonomous-GPT Supervised) classifier predictions on IA test set
│   │   └── unseen_EA_IA_train_autonomous.csv     # 8,000 GPT-only annotated queries (Autonomous Set)
│   └── analysis/
│       └── misalignment_analysis.csv             # Qualitative error analysis (Section 5.2)
```

## Usage

To train the classifiers, load the corresponding train, eval, and
test split CSVs from `data/classifier_experiments/` into the
training folder and execute.

To run the baselines, set `dim = 'EA'` or `dim = 'IA'` at the top
of each script in `training/` and ensure the relevant CSVs
from `data/classifier_experiments/` are loaded. 
For McNemar significance tests, `output_human_EA.csv` and `output_human_IA.csv` (Transformer Human Supervised predictions) must also be loaded.


## Acknowledgments

Patient queries are sourced from the HealthCareMagic and iCliniq
datasets released by [Li et al. (2023)](https://arxiv.org/abs/2303.14070).

## Citation

```bibtex
@article{randhawa2026empathy,
  title={Empathy Applicability Modeling for General Health Queries},
  author={Randhawa, Shan and Raza, Agha Ali and Toyama, Kentaro and Hui, Julie and Naseem, Mustafa},
  journal={arXiv preprint arXiv:2601.09696},
  year={2026}
}
```
