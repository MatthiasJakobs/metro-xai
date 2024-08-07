# Interpretable Rules for Online Failure Prediction: A Case Study on the Metro do Porto dataset

- First, generate data from MetroPT2.csv found [here](https://zenodo.org/records/7766691) by calling `python generate_chunks.py`
- Next, train model with `python train_models.py TCN`
- Results for failure prediction are found in `pt2_failure_detection.ipynb`.
- Code and experiments for the online rule-learning approach are found in `online_rules.py`