# DetectCodeGPT

Welcome to the repository for the research paper: "Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers." Here, we present DetectCodeGPT, a novel approach to distinguish between machine- and human-generated code snippets. This README will guide you through setting up and using the DetectCodeGPT framework.

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

## Support

If you encounter any issues or have questions, please feel free to contact us!

We hope that our work will aid in advancing the field of machine learning in code generation and detection. Thank you for your interest in DetectCodeGPT!
