# Interpretable Rules for Online Failure Prediction: A Case Study on the Metro do Porto dataset

- First, generate data from MetroPT1.csv and found [here](https://zenodo.org/records/6854240) and MetroPT2.csv found [here](https://zenodo.org/records/7766691) by calling `python generate_chunks.py`. Remember to rename the downloaded files.
- Next, train models with `python train_models.py PT1_TCN` and `python train_models.py PT2_TCN`.
- Extract rules for our approach (`bash run_orules.sh`) and AMRules (`bash run_amrules.sh`).
- Results for failure prediction are found in `pt2_failure_detection.ipynb` and `pt1_failure_detection.ipynb`.
- Code and experiments for the online rule-learning approach are found in `online_rules.py`