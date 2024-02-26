# Reproducing a State-of-the-Art Paper on Machine Learning Applied to Brain Computer Interfaces

This repository hosts the code and documentation for the paper "Reproducing a State-of-the-Art Paper on Machine Learning Applied to Brain Computer Interfaces" by Eduard Ramon Aliaga Torrens and Alberto Cabellos Aparicio. Our work advances the research on Brain-Computer Interfaces (BCIs) by proposing a Combined Convolutional Neural Network (CombinedCNN) model, aimed at enhancing the accuracy of EEG signal analysis for detecting user intent in a typing task.

You will find the combined cnn model among the other ones here: [Combined-cnn and others](https://github.com/EduardAliaga/Combined-cnn-model-for-EEG-signal-classification/blob/main/src/bci_disc_models/models/neural_net/network_arch.py)
# Setup

Setup project with make and activate virtualenv with source venv/bin/activate

# Usage

To reproduce our experiments, please follow these steps:

Preprocess data: ```python scripts/prepare_data.py```
Pretrain models: ```python scripts/train.py```
Evaluate models in simulated typing task: ```python scripts/evaluate.py```
Parse saved results from evaluation: ```python scripts/parse_results.py```
Collect statistics from parsed results: ```python scripts/analyze_results.py```
Make plots: ```python scripts/plots.py```
To run tests: ```pytest --disable-warnings -s```


## Resources

- **Data:** This project utilizes the `thu-rsvp-dataset` for benchmarking. [Dataset Details](https://www.frontiersin.org/articles/10.3389/fnins.2020.568000/full)

- **Papers:** For detailed insights into our methodology and findings, refer to our paper. [Access the Paper](https://github.com/EduardAliaga/Combined-cnn-model-for-EEG-signal-classification/blob/main/Reproducing%20a%20State-of-the-Art%20paper%20on%20Machine%20Learning%20applied%20to%20Brain%20Computer%20Interfaces.pdf)


