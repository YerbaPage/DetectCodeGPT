<div align="center">

# DetectCodeGPT

[![Conference](https://img.shields.io/badge/Conference-ICSE%202025-brightgreen)](https://icse2025.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9.7-blue.svg)](https://www.python.org/downloads/release/python-397/)

</div>

Welcome to the repository for the research paper: **"Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers."** Our paper has been accepted to the 47th International Conference on Software Engineering (**ICSE 2025**).

## Table of Contents

- [DetectCodeGPT](#detectcodegpt)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Data Preparation](#data-preparation)
  - [Usage](#usage)
    - [Conducting the Empirical Study](#conducting-the-empirical-study)
    - [Using DetectCodeGPT](#using-detectcodegpt)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

## Getting Started

### Prerequisites

Experiments are conducted using **Python 3.9.7** on an **Ubuntu 22.04.1** server.

To install all required packages, navigate to the root directory of this project and run:

```bash
pip install -r requirements.txt
```

### Data Preparation

To prepare the datasets used in our study:

1. Navigate to the `code-generation` directory.
2. Obtain datasets from either:
   - [CodeSearchNet](https://github.com/github/CodeSearchNet)
   - [Preprocessed version of The Stack (The Vault)](https://github.com/FSoft-AI4Code/TheVault)
3. Update the data paths and model specifications in `generate.py` to reflect your local setup.
4. Execute the data generation script:

   ```bash
   python generate.py
   ```

## Usage

### Conducting the Empirical Study

> **Note**: You can skip the empirical study if you are only interested in detecting machine-generated code with DetectCodeGPT.

After data preparation, you can proceed to the empirical analysis:

1. Navigate to the `code-analysis` directory.
2. Analyze code length:

   ```bash
   python analyze_length.py
   ```

3. Verify Zipf's and Heaps' laws, and compute token frequencies:

   ```bash
   python analyze_law_and_frequency.py
   ```

4. Analyze the proportion of different token categories:

   ```bash
   python analyze_proportion.py
   ```

5. Study the naturalness of code snippets:

   ```bash
   python analyze_naturalness.py
   ```

### Using DetectCodeGPT

To evaluate our DetectCodeGPT model:

1. Navigate to the `code-detection` directory.
2. Configure `main.py` with the appropriate model and dataset paths.
3. Run the model evaluation script:

   ```bash
   python main.py
   ```

> **Note**: If you are using your custom model to generate code, please update `'base_model_name': "codellama/CodeLlama-7b-hf"` in `main.py` to your model name during the detection stage.

## Acknowledgements

The code is modified based on the original repositories of [DetectGPT](https://github.com/eric-mitchell/detect-gpt/tree/main/) and [DetectLLM](https://github.com/mbzuai-nlp/DetectLLM). We thank the authors for their contributions.

## Citation

If you use DetectCodeGPT in your research, please cite our paper:

```bibtex
@inproceedings{shi2025detectcodegpt,
  title={Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers},
  author={Shi, Yuling and Zhang, Hongyu and Wan, Chengcheng and Gu, Xiaodong},
  booktitle={Proceedings of the 47th International Conference on Software Engineering (ICSE 2025)},
  year={2025},
  organization={IEEE}
}
```