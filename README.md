# Volvo_Discovery_Challenge_ECML_PKDD_2024
Implementation of the 2nd leaderboard submisison to the 2024 ECML PKDD Discovery Challenge

# Predictive Maintenance for Volvo Trucks Using LSTM and Pseudo-Labeling

This repository contains the code for the methodology presented in the paper titled "Achieving Predictive Precision: Leveraging LSTM and Pseudo Labeling for Volvo’s Discovery Challenge at ECML-PKDD 2024". The paper details our approach to using Long Short-Term Memory (LSTM) networks and pseudo-labeling for predicting maintenance needs in Volvo trucks, which secured us the second place at the Volvo Discovery Challenge.

## Overview

The methodology involves preprocessing the dataset to closely mirror the structure of the test set, using a base LSTM model to iteratively label the test data, and refining the model's predictions through a series of boosting techniques and pseudo-labeling. This approach enhanced the predictive capabilities of our models, achieving a macro-average F1-score of 0.879.

## Repository Structure

This repository is organized into three main directories:

- **EDA**: Contains Jupyter notebooks and scripts used for exploratory data analysis (EDA). This includes inspection, visualization, and preliminary analysis of the dataset provided for the challenge.
- **Model**: Includes Python scripts for building and training the LSTM model. This section also contains the pseudo-labeling implementation and the boosting techniques used.
- **Test and Evaluation**: Features scripts used for testing, evaluation, and post-processing adjustments. This also includes consistency checks and scripts for generating final predictions and evaluating them against the provided metrics.

# Citation
If you use the methodologies or the codebase in this repository in your research, please cite it as follows:


@inproceedings{volvo2024predictive,
title={Achieving Predictive Precision: Leveraging LSTM and Pseudo Labeling for Volvo’s Discovery Challenge at ECML-PKDD 2024},
author={Carlo Metta et alt.},
booktitle={ECML-PKDD 2024},
year={2024}
}

# License
This project is licensed under the MIT License - see the LICENSE.md file for details.

# Contact
For any queries related to this project, please contact:

Carlo Metta, ISTI-CNR, Italy - carlo.metta@isti.cnr.it
