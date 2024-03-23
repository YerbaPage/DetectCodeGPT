import math
from transformers import AutoTokenizer
import numpy as np
from scipy.stats import norm
import scipy.stats
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from baselines.all_baselines import run_all_baselines
from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model
from baselines.utils.preprocessing import preprocess_and_save
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from baselines.sample_generate.generate import generate_data


def generate_data(dataset, key, max_num=200, min_len=0, max_len=128, max_comment_num=10, max_def_num=5, cut_def=False, max_todo_num=3):

    path = f'../code-generation/output/{dataset}/{key}/outputs.txt'

    logger.info(f'Loading data from {path}')
    import json
    all_originals = []
    all_samples = []  # machine generated

    max_def_num_count = 0
    min_len_count = 0
    max_comment_num_count = 0
    function_comment_num_count = 0
    max_todo_num_count = 0

    with open(path, 'r') as f:
        for line in tqdm(f, ncols=70):
            line = line.strip()
            if line == '':
                continue
            line = json.loads(line)

            # cut out the 'def' part after the first generation
            if cut_def:
                line['output'] = line['output'].split('def')[0]
                line['solution'] = line['solution'].split('def')[0]

            # cut if line number < 2
            if line['solution'].count('\n') < 2 or line['output'].count('\n') < 2:
                continue

            # cut to 128 tokens
            all_originals.append(' '.join(line['solution'].split(' ')[:max_len]))
            all_samples.append(' '.join(line['output'].split(' ')[:max_len]))

    logger.info(f'{max_def_num_count} examples have more than {max_def_num} "def"')
    logger.info(f'{min_len_count} examples have less than {min_len} words')
    logger.info(f'{max_comment_num_count} examples have more than {max_comment_num} comments')
    logger.info(f'{max_todo_num_count} examples have more than {max_todo_num} TODOs')
    logger.info(f'{function_comment_num_count} examples have more than 1 function comment')
    logger.info(f'Loaded {len(all_originals)} examples after filtering, and will return {min(max_num, len(all_originals))} examples')

    data = {
        "original": all_originals[:max_num],
        "sampled": all_samples[:max_num]
    }

    return data


def tokenize_data(data, tokenizer):
    token_counts = []
    line_counts = []

    for code in data:
        tokens = tokenizer.tokenize(code)
        token_counts.append(len(tokens))

        lines = code.split('\n')
        line_counts.append(len(lines))

    return token_counts, line_counts


def plot_distribution(original_counts, sampled_counts, title, save_name=None):
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(original_counts, bins=25, density=True, alpha=0.5, label='Human', color='orange', edgecolor='orange')
    plt.hist(sampled_counts, bins=25, density=True, alpha=0.5, label='Machine', color='green', edgecolor='green')

    # Fitting Gaussian distribution for original counts and plotting
    mu_o, std_o = norm.fit(original_counts)
    xmin_o, xmax_o = plt.xlim()
    x_o = np.linspace(xmin_o, xmax_o, 100)
    p_o = norm.pdf(x_o, mu_o, std_o)
    plt.plot(x_o, p_o, 'k', linewidth=3, color='orange')

    # Fitting Gaussian distribution for sampled counts and plotting
    mu_s, std_s = norm.fit(sampled_counts)
    xmin_s, xmax_s = plt.xlim()
    x_s = np.linspace(xmin_s, xmax_s, 100)
    p_s = norm.pdf(x_s, mu_s, std_s)
    plt.plot(x_s, p_s, 'k', linewidth=3, color='green')

    # Labels and title
    # plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    # increase the font size
    plt.rcParams.update({'font.size': 16})

    # save to pdf
    if save_name:
        plt.savefig(f'figures/{save_name}.pdf', bbox_inches='tight')

    plt.show()


def main(dataset, dataset_key, temperature, tokenizer_name):

    data = generate_data(dataset, dataset_key, max_num=100000, min_len=0, max_len=512, max_comment_num=10, max_def_num=5, cut_def=True, max_todo_num=3)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

    # Tokenize data
    original_token_counts, original_line_counts = tokenize_data(data['original'], tokenizer)
    sampled_token_counts, sampled_line_counts = tokenize_data(data['sampled'], tokenizer)

    # remove those with too many tokens ( > 500)
    original_token_counts = [x for x in original_token_counts if x <= 500]
    sampled_token_counts = [x for x in sampled_token_counts if x <= 500]

    # Token distribution
    plot_distribution(original_token_counts, sampled_token_counts, "Distribution of Number of Tokens", save_name=f'token_distribution_tp{temperature}')

    # Line distribution
    plot_distribution(original_line_counts, sampled_line_counts, "Distribution of Number of Lines", save_name=f'line_distribution_tp{temperature}')


dataset = "CodeSearchNet"
tokenizer_name = "bigcode/starcoderbase-3b"
temperature = 0.2
dataset_key = "starcoderbase-3b-5000-tp0.2-nostop"

main(dataset, dataset_key, temperature, tokenizer_name)

temperature = 1.0
dataset_key = "starcoderbase-3b-5000-tp1.0-nostop"
main(dataset, dataset_key, temperature, tokenizer_name)


def tokenize_data(data, tokenizer):
    token_counts = []
    line_counts = []

    for code in data:
        tokens = tokenizer.tokenize(code)
        token_counts.append(len(tokens))

        lines = code.split('\n')
        line_counts.append(len(lines))

    return token_counts, line_counts


def plot_distribution(original_counts, sampled_counts, title, save_name=None):
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(original_counts, bins=50, density=True, alpha=0.5, label='Human', color='orange', edgecolor='orange')
    plt.hist(sampled_counts, bins=50, density=True, alpha=0.5, label='Machine', color='green', edgecolor='green')

    # Fitting Gaussian distribution for original counts and plotting
    mu_o, std_o = norm.fit(original_counts)
    xmin_o, xmax_o = plt.xlim()
    x_o = np.linspace(xmin_o, xmax_o, 100)
    p_o = norm.pdf(x_o, mu_o, std_o)
    plt.plot(x_o, p_o, 'k', linewidth=3, color='orange')

    # Fitting Gaussian distribution for sampled counts and plotting
    mu_s, std_s = norm.fit(sampled_counts)
    xmin_s, xmax_s = plt.xlim()
    x_s = np.linspace(xmin_s, xmax_s, 100)
    p_s = norm.pdf(x_s, mu_s, std_s)
    plt.plot(x_s, p_s, 'k', linewidth=3, color='green')

    # Labels and title
    # plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    # increase the font size
    plt.rcParams.update({'font.size': 16})
    if 'line' in title.lower():
        plt.xlim(-5, 70)

    # save to pdf
    if save_name:
        plt.savefig(f'figures/{save_name}.pdf', bbox_inches='tight')

    plt.show()


def main(dataset, dataset_key, temperature, tokenizer_name):

    data = generate_data(dataset, dataset_key, max_num=100000, min_len=0, max_len=512, max_comment_num=10, max_def_num=5, cut_def=True, max_todo_num=3)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

    # Tokenize data
    original_token_counts, original_line_counts = tokenize_data(data['original'], tokenizer)
    sampled_token_counts, sampled_line_counts = tokenize_data(data['sampled'], tokenizer)

    # remove those with too many tokens ( > 500)
    original_token_counts = [x for x in original_token_counts if x <= 500]
    sampled_token_counts = [x for x in sampled_token_counts if x <= 500]

    # Token distribution
    plot_distribution(original_token_counts, sampled_token_counts, "Distribution of Number of Tokens", save_name=f'token_distribution_tp{temperature}')

    # Line distribution
    plot_distribution(original_line_counts, sampled_line_counts, "Distribution of Number of Lines", save_name=f'line_distribution_tp{temperature}')


dataset = "CodeSearchNet"
tokenizer_name = "codellama/CodeLlama-7b-hf"
temperature = 0.2
dataset_key = "CodeLlama-7b-hf-5000-tp0.2-nostop"

main(dataset, dataset_key, temperature, tokenizer_name)

temperature = 1.0
dataset_key = "CodeLlama-7b-hf-5000-tp1.0-nostop"
main(dataset, dataset_key, temperature, tokenizer_name)
