# Solutions for Encrypted Network Traffic Classification

Code and results for a comparison of three solutions for detecting VPN traffic from statistical flow features.

**Authors:** Harshita Shah, Aditi Singh, Vinikesh Hiranandani and Shreyansh Mohanty
Department of Computer Engineering and Technology, Dr. Vishwanath Karad MIT World Peace University, Pune, India

## Solutions compared

1. **Random Forest:** 200 trees, maximum depth 20
2. **Standard 1D-CNN:** two convolutional layers, 58,625 parameters
3. **Improved 1D-CNN:** global max pooling and L2 regularization, 12,993 parameters, plus variants trained with FGSM adversarial training

SHAP and LIME explain the decisions. FGSM, PGD, a transfer attack and a random-noise control measure robustness.

## Dataset

ISCX VPN-nonVPN, 15-second time-based flows: 18,758 flows, binary label (VPN or Non-VPN), 15 of 23 features selected by mutual information, stratified 80/10/10 split. The 15-second file is in [`data/`](data/); see [`data/README.md`](data/README.md) for the source, citation and the other timeouts.

## Results

### Test performance of the saved models

| Metric | Random Forest | Standard 1D-CNN | Improved 1D-CNN |
|---|---:|---:|---:|
| Accuracy (%) | 91.79 | 71.48 | 72.55 |
| Macro precision | 0.9185 | 0.7143 | 0.7348 |
| Macro recall | 0.9171 | 0.7145 | 0.7294 |
| Macro F1 | 0.9176 | 0.7144 | 0.7246 |
| ROC-AUC | 0.9765 | 0.8025 | 0.8159 |
| Training time (s, Google Colab) | 14.84 | 41.23 | 53.60 |

The Random Forest beat the Improved 1D-CNN significantly (McNemar p < 0.001). The two CNNs did not differ significantly (p = 0.28).

### Five training seeds on the same split (mean ± SD)

| Model | Clean accuracy (%) | FGSM, ε=0.10 (%) | PGD, ε=0.10 (%) |
|---|---:|---:|---:|
| Random Forest | 92.08 ± 0.18 | not applicable | not applicable |
| Standard 1D-CNN | 72.30 ± 1.14 | 21.35 ± 8.80 | 6.93 ± 2.22 |
| Improved 1D-CNN | 72.32 ± 1.01 | 36.70 ± 4.83 | 6.00 ± 2.27 |
| Adversarially trained, ε=0.05 | 67.80 ± 1.36 | 63.98 ± 3.45 | 29.86 ± 5.34 |
| Adversarially trained, ε=0.10 | 68.33 ± 1.67 | 77.68 ± 2.34 | 22.25 ± 3.00 |

Trees have no input gradient, so the Random Forest was tested separately. Attack examples crafted on the Improved 1D-CNN at ε=0.01 reduced it to 39.23%, and random noise of ε=0.01 reduced it to 58.50%. See `results/rerun/robustness_saved.csv` and `results/rerun/random_noise.csv`.

### Main findings

- The Random Forest was the most accurate model, but also very sensitive to small changes in standardized features.
- The Improved 1D-CNN used 78% fewer parameters than the Standard 1D-CNN, with no repeatable accuracy gain.
- Adversarial training helped against FGSM, but PGD still reduced accuracy well below 50%, so robustness is only partial.
- SHAP rankings for the Random Forest and the Improved 1D-CNN were negatively correlated (ρ = -0.59).
- The Standard 1D-CNN never uses `std_flowiat`: max pooling discards the only position that depends on this last input feature.

## Repository layout

```
notebooks/
  WORKINGMODEL_6.ipynb       Project notebook on the real dataset
  Rerun_Experiments.ipynb    Colab notebook for the added experiments
code/scripts/
  common.py                  Data loading and the fixed split
  verify_saved_models.py     Reproduces the saved models' test results
  attacks_shap_lime.py       FGSM, PGD, transfer attack, statistics, SHAP, LIME
  random_noise_control.py    Random-noise control and feature scale checks
  train_five_seeds.py        Five-seed training, including adversarial training
  summarize_seeds.py         Summarizes the five-seed results
  rf_stability_check.py      Random Forest over five splits, with and without duplicates
  dataset_check.py           Checks dataset size, placeholders and column values
  make_figures.py            Regenerates results/figures
results/
  original_run/              Models, tables and plots saved by the original notebook
  rerun/                     Added experiment tables, attributions, logs and models
  figures/                   Figures made from those results
data/                        ISCX VPN-nonVPN 15-second flow file and its source
```

## How to run

**Colab (not yet test-run there):** open `notebooks/Rerun_Experiments.ipynb`, upload the file in `data/` zipped as `ISCX_VPN_nonVPN_real_CSVs.zip` and the contents of `results/original_run`, with a retrained `random_forest.pkl` added, zipped as `Complete_Research_Results.zip`, then choose Runtime > Run all.

**Locally:** first create `random_forest.pkl` as described under Saved Random Forest below, then:

```bash
pip install -r requirements.txt
python code/scripts/verify_saved_models.py
python code/scripts/attacks_shap_lime.py
python code/scripts/random_noise_control.py
python code/scripts/train_five_seeds.py
python code/scripts/summarize_seeds.py
python code/scripts/make_figures.py
```

These scripts overwrite the files in `results/rerun` and `results/figures`. Retrained models vary between machines, so `train_five_seeds.py` may not match the stored numbers exactly. Only open `.pkl` files from sources you trust.

## Saved Random Forest

The saved Random Forest (`random_forest.pkl`, about 30 MB) is not stored in this repository, to keep it small. `code/scripts/verify_saved_models.py`, `attacks_shap_lime.py` and `random_noise_control.py` load it from `results/original_run/`, so retrain it first using the settings in `notebooks/WORKINGMODEL_6.ipynb` and save it there with `joblib`. Their stored outputs in `results/rerun` used the original model. Retraining with the settings in `notebooks/WORKINGMODEL_6.ipynb` gave 91.84% test accuracy on a newer scikit-learn version, against 91.79% for the saved model.

## Limitations

- One dataset, one flow timeout and a binary label, so results do not cover application-level classification.
- Flows were split at random and not by capture session, so results on new networks may be lower.
- The dataset uses -1 as an apparent placeholder (59,089 cells), and these values were scaled as ordinary numbers.
- Some column names look inconsistent with their values: `total_fiat` is at most `min_fiat` in every flow. Interpret attributions by feature name with caution.
- ε is measured on standardized features, and the attacks do not check whether a real flow could produce the perturbed values, so they approach a worst case.
- SHAP used the first 300 test flows and LIME the first 100.

## Acknowledgment

Thanks to Prof. Sagar Apune, seminar guide, and Dr. Balaji Patil, Program Director.
