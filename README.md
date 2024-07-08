# DetectCodeGPT

Welcome to the repository for the research paper: "Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers." Our paper has been accepted to the 47th International Conference on Software Engineering (ICSE 2025). 

## Getting Started

### Prerequisites

Experiments are conducted using Python 3.9.7 on a Ubuntu 22.01.1 server.

To install all required packages, navigate to the root directory of this project and run:

```
pip install -r requirements.txt
```

### Data Preparation

To prepare the datasets used in our study:

1. Navigate to the `code-generation` directory.
2. Obtain datasets from either:
   - [CodeSearchNet](https://github.com/github/CodeSearchNet)
   - [Preprocessed version of The Stack (The Vault)](https://github.com/FSoft-AI4Code/TheVault)

3. Update the data paths and model specifications in `generate.py` to reflect your local setup.
4. Execute the data generation script with:
   ```
   python generate.py
   ```

## Conducting the Empirical Study

After data preparation, you can proceed to the empirical analysis:

1. Navigate to the `code-analysis` directory.
2. Analyze code length by running:
   ```
   python analyze_length.py
   ```
3. Verify Zipf's and Heaps' laws, and compute token frequencies with:
   ```
   python analyze_law_and_frequency.py
   ```
4. Analyze the proportion of different token categories by executing:
   ```
   python analyze_proportion.py
   ```
5. Study the naturalness of code snippets via:
   ```
   python analyze_naturalness.py
   ```

## Using DetectCodeGPT

To evaluate our DetectCodeGPT model:

1. Navigate to the `code-detection` directory.
2. Configure `main.py` with the appropriate model and dataset paths.
3. Run the model evaluation script with:
   ```
   python main.py
   ```


## Acknowledgements

The codes are modified based on the original repository of [DetectGPT](https://github.com/eric-mitchell/detect-gpt/tree/main/) and the original repository of [DetectLLM](https://github.com/mbzuai-nlp/DetectLLM). We thank the authors for their contributions.

## Citation

If you use DetectCodeGPT in your research, please cite our paper:

```bibtex
@inproceedings{shi2024detectcodegpt,
title={Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers},
author={Shi, Yuling and Zhang, Hongyu and Wan, Chengcheng and Gu, Xiaodong},
booktitle={Proceedings of the The 47th International Conference on Software Engineering (ICSE 2025)},
year={2025},
organization={IEEE}
}
```