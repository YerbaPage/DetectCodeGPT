import pandas as pd
import numpy as np
from token_tagging import *
import multiprocessing
import torch
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from baselines.all_baselines import run_all_baselines
from transformers import AutoTokenizer
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="writing")
parser.add_argument('--dataset_key', type=str, default="document")
parser.add_argument('--pct_words_masked', type=float, default=0.3)
parser.add_argument('--span_length', type=int, default=2)
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--n_perturbation_list', type=str, default="10")
parser.add_argument('--n_perturbation_rounds', type=int, default=1)
parser.add_argument('--base_model_name', type=str, default="")
parser.add_argument('--scoring_model_name', type=str, default="")
parser.add_argument('--mask_filling_model_name', type=str, default="Salesforce/CodeT5-large")
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--chunk_size', type=int, default=20)
parser.add_argument('--n_similarity_samples', type=int, default=20)
parser.add_argument('--int8', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--base_half', action='store_true')
parser.add_argument('--do_top_k', action='store_true')
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--do_top_p', action='store_true')
parser.add_argument('--top_p', type=float, default=0.96)
parser.add_argument('--output_name', type=str, default="test_ipynb")
parser.add_argument('--openai_model', type=str, default=None)
parser.add_argument('--openai_key', type=str)
parser.add_argument('--DEVICE', type=str, default='cuda')
parser.add_argument('--buffer_size', type=int, default=1)
parser.add_argument('--mask_top_p', type=float, default=1.0)
parser.add_argument('--mask_temperature', type=float, default=1.0)
parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
parser.add_argument('--pre_perturb_span_length', type=int, default=5)
parser.add_argument('--random_fills', action='store_true')
parser.add_argument('--random_fills_tokens', action='store_true')
parser.add_argument('--cache_dir', type=str, default="~/.cache/huggingface/hub")
parser.add_argument('--prompt_len', type=int, default=30)
parser.add_argument('--generation_len', type=int, default=200)
parser.add_argument('--min_words', type=int, default=55)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--baselines', type=str, default="LRR,DetectGPT,NPR")
parser.add_argument('--perturb_type', type=str, default="random")
parser.add_argument('--pct_identifiers_masked', type=float, default=0.5)
parser.add_argument('--min_len', type=int, default=0)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--max_comment_num', type=int, default=10)
parser.add_argument('--max_def_num', type=int, default=5)
parser.add_argument('--cut_def', action='store_true')
parser.add_argument('--max_todo_num', type=int, default=3)

args_dict = {

    'dataset': "CodeSearchNet",
    'dataset_key': "CodeLlama-7b-hf-5000-tp0.2",
    'pct_words_masked': 0.5,
    'pct_identifiers_masked': 0.75,
    'span_length': 2,
    'n_samples': 5000,
    'n_perturbation_list': "50",
    'n_perturbation_rounds': 1,
    'base_model_name': "bigcode/santacoder",
    'scoring_model_name': "",
    'mask_filling_model_name': "Salesforce/codet5p-770m",
    'batch_size': 25,
    'chunk_size': 10,
    'n_similarity_samples': 20,
    'int8': False,
    'half': False,
    'base_half': False,
    'do_top_k': False,
    'top_k': 40,
    'do_top_p': False,
    'top_p': 0.96,
    'output_name': "test_ipynb",
    'openai_model': None,
    'openai_key': None,
    'DEVICE': 'cuda',
    'buffer_size': 1,
    'mask_top_p': 1.0,
    'mask_temperature': 1,
    'pre_perturb_pct': 0.0,
    'pre_perturb_span_length': 5,
    'random_fills': False,
    'random_fills_tokens': False,
    'cache_dir': "~/.cache/huggingface/hub",
    'prompt_len': 30,
    'generation_len': 200,
    'min_words': 55,
    'temperature': 1,
    'baselines': "LRR,DetectGPT,NPR",
    'perturb_type': "random-insert-space+newline",  # half of the examples will have newline, half will have new space
    'min_len': 0,
    'max_len': 128,
    'max_comment_num': 10,
    'max_def_num': 5,
    'cut_def': False,
    'max_todo_num': 3
}

input_args = []
for key, value in args_dict.items():
    if value:
        input_args.append(f"--{key}={value}")

args = parser.parse_args(input_args)

mask_filling_model_name = args.mask_filling_model_name
n_samples = args.n_samples
batch_size = args.batch_size
n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
n_perturbation_rounds = args.n_perturbation_rounds
n_similarity_samples = args.n_similarity_samples

logger.info(f'args: {args}')

tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, cache_dir=args.cache_dir)


token_categories = ['keyword', 'identifier', 'literal', 'operator', 'statement_function', 'data_structure', 'module_block', 'delimiters', 'other', 'comment', 'whitespace']
categories2idx = {category: i for i, category in enumerate(token_categories)}
idx2categories = {i: category for i, category in enumerate(token_categories)}


def process_chunk(chunk):
    results = []
    for code_item_original, code_item_sampled in chunk:
        tokens_with_categories_original, unrecognized_categories_original, hf_tokens_original = align_tokens_with_categories(code_item_original, 'python', tokenizer, return_hf_tokens=True)
        tokens_with_categories_sampled, unrecognized_categories_sampled, hf_tokens_sampled = align_tokens_with_categories(code_item_sampled, 'python', tokenizer, return_hf_tokens=True)

        if len(tokens_with_categories_original) == len(hf_tokens_original) and len(tokens_with_categories_sampled) == len(hf_tokens_sampled):
            categories_original = [categories2idx[token_category] for token, token_category in tokens_with_categories_original]
            categories_sampled = [categories2idx[token_category] for token, token_category in tokens_with_categories_sampled]
            results.append((code_item_original, code_item_sampled, categories_original, categories_sampled))
        else:
            results.append(None)  # for unsuccessful ones

    return results


def process_data_multithreaded(data, tokenizer, num_processes=None):
    if not num_processes:
        num_processes = multiprocessing.cpu_count()-1

    chunk_size = len(data['original']) // num_processes
    chunks = [list(zip(data['original'][i:i+chunk_size], data['sampled'][i:i+chunk_size])) for i in range(0, len(data['original']), chunk_size)]

    new_data = {
        'original': [],
        'sampled': [],
        'token_categories_original': [],
        'token_categories_sampled': []
    }

    failed_count = 0
    with multiprocessing.Pool(processes=num_processes) as pool:
        for chunk_results in tqdm(pool.imap_unordered(process_chunk, chunks), total=len(chunks)):
            for result in chunk_results:
                if result:  # if successful
                    code_item_original, code_item_sampled, categories_original, categories_sampled = result
                    new_data['original'].append(code_item_original)
                    new_data['sampled'].append(code_item_sampled)
                    new_data['token_categories_original'].append(categories_original)
                    new_data['token_categories_sampled'].append(categories_sampled)
                else:
                    failed_count += 1

    logger.info('Percentage of unrecognized codes during tagging: {:.2f}%'.format(failed_count / len(data['original']) * 100))

    return new_data


def process_data(data, tokenizer):

    new_data = {
        'original': [],
        'sampled': [],
        'token_categories_original': [],
        'token_categories_sampled': []
    }

    failed_indices = set()
    for idx, (code_item_original, code_item_sampled) in tqdm(enumerate(zip(data['original'], data['sampled'])), total=len(data['original'])):
        for code_item, key_prefix in [(code_item_original, 'original'), (code_item_sampled, 'sampled')]:
            tokens_with_categories, unrecognized_categories, hf_tokens = align_tokens_with_categories(code_item, 'python', tokenizer, return_hf_tokens=True)

            if len(tokens_with_categories) != len(hf_tokens):
                failed_indices.add(idx)
                break

            categories = [categories2idx[token_category] for token, token_category in tokens_with_categories]
            new_data[f'token_categories_{key_prefix}'].append(categories)

    for idx, (code_item_original, code_item_sampled) in enumerate(zip(data['original'], data['sampled'])):
        if idx not in failed_indices:
            new_data['original'].append(code_item_original)
            new_data['sampled'].append(code_item_sampled)

    logger.info('Percentage of unrecognized codes during tagging: {:.2f}%'.format(len(failed_indices) / len(data['original']) * 100))

    if len(set(unrecognized_categories)) > 0:
        logger.info(f'Unrecognized categories: {set(unrecognized_categories)}')

    return new_data


def generate_data(dataset, key, tokenizer, max_num=200, min_len=1, max_len=128, max_comment_num=10, max_def_num=5, cut_def=False, max_todo_num=3):

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

            # I don't like there to have too many 'def' in the code
            # ~100/100000 examples have more than 3 'def'
            if line['solution'].count('def') > max_def_num or line['output'].count('def') > max_def_num:
                max_def_num_count += 1
                continue

            # avoid examples that are too short (less than min_len words)
            # around 2000/100000 examples have around 55 words
            if len(line['solution'].split()) < min_len or len(line['output'].split()) < min_len:
                min_len_count += 1
                continue

            # if the are too many comments, skip
            def count_comment(text):
                return text.count('#')

            if count_comment(line['solution']) > max_comment_num or count_comment(line['output']) > max_comment_num:
                max_comment_num_count += 1
                continue

            # if there are too many TODOs, skip
            def count_todo_comment(text):
                return text.count('# TODO') + text.count('# todo')

            if count_todo_comment(line['solution']) > max_todo_num or count_todo_comment(line['output']) > max_todo_num:
                max_todo_num_count += 1
                continue

            # the number of text.count("'''") and text.count('"""') should be <1
            if line['solution'].count("'''") > 0 or line['solution'].count('"""') > 0 or line['output'].count("'''") > 0 or line['output'].count('"""') > 0:
                function_comment_num_count += 1
                continue

            # cut to 128 tokens
            all_originals.append(' '.join(line['solution'].split(' ')[:max_len]))
            all_samples.append(' '.join(line['output'].split(' ')[:max_len]))

    logger.info(f'Loaded {len(all_originals)} examples before tagging')

    # data = process_data({'original': all_originals, 'sampled': all_samples}, tokenizer)
    data = process_data_multithreaded({'original': all_originals, 'sampled': all_samples}, tokenizer)

    max_num = min(max_num, len(all_originals))

    logger.info(f'{max_def_num_count} examples have more than {max_def_num} "def"')
    logger.info(f'{min_len_count} examples have less than {min_len} words')
    logger.info(f'{max_comment_num_count} examples have more than {max_comment_num} comments')
    logger.info(f'{max_todo_num_count} examples have more than {max_todo_num} TODOs')
    logger.info(f'{function_comment_num_count} examples have more than 1 function comment')
    logger.info(f'Loaded {len(data["original"])} examples after filtering, and will return {max_num} examples')

    # data = {
    #     "original": all_originals[:max_num],
    #     "sampled": all_samples[:max_num]
    # }

    # cut the data based on the max_num
    data = {
        "original": data["original"][:max_num],
        "sampled": data["sampled"][:max_num],
        "token_categories_original": data["token_categories_original"][:max_num],
        "token_categories_sampled": data["token_categories_sampled"][:max_num]
    }

    return data


data = generate_data(args.dataset, args.dataset_key, tokenizer, max_num=args.n_samples, min_len=args.min_len, max_len=args.max_len,
                     max_comment_num=args.max_comment_num, max_def_num=args.max_def_num, cut_def=args.cut_def, max_todo_num=args.max_todo_num)

logger.info(f'Original: {data["original"][0]}')
logger.info(f'Sampled: {data["sampled"][0]}')

token_categories = ['keyword',
                    'identifier',
                    'literal',
                    'operator',
                    'statement_function',
                    'data_structure',
                    'module_block',
                    'delimiters',
                    'other',
                    'comment',
                    'whitespace']

categories2idx = {category: i for i, category in enumerate(token_categories)}
idx2categories = {i: category for i, category in enumerate(token_categories)}


def get_category_frequencies(token_categories):
    """
    Compute the frequencies of each category in a list of token categories.
    """
    freqs = np.zeros(len(categories2idx))
    for categories in token_categories:
        for cat in categories:
            freqs[cat] += 1
    return freqs


def plot_category_distribution(freqs_orig, freqs_sampled, title="Token Category Distribution", save_name="token-category-distribution"):
    """
    Plot the distributions of token categories for original and sampled sentences.
    """

    # normalize frequencies
    freqs_orig = freqs_orig / np.sum(freqs_orig)
    freqs_sampled = freqs_sampled / np.sum(freqs_sampled)

    labels = token_categories
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))
    rects1 = ax.bar(x - width/2, freqs_orig, width, label='Human', color='orange', alpha=0.5, edgecolor='orange')
    rects2 = ax.bar(x + width/2, freqs_sampled, width, label='Machine', color='green', alpha=0.5, edgecolor='green')

    # ax.set_xlabel('Token Category')
    ax.set_ylabel('Frequency')
    # ax.set_title(title)
    # larger font
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.savefig(f'figures/{save_name}.pdf')
    # plt.show()


def compute_category_entropy(freqs):
    """
    Compute the entropy of the token category distribution.
    """
    prob = freqs / np.sum(freqs)
    return -np.sum(prob * np.log2(prob + np.finfo(float).eps))


plt.rcParams.update({'font.size': 24})

# Compute the frequencies for each dataset
freqs_orig = get_category_frequencies(data["token_categories_original"])
freqs_sampled = get_category_frequencies(data["token_categories_sampled"])

# remove those categories with 0 frequency (statement_function, module_block, data_structure, other)
freqs_orig = np.delete(freqs_orig, [4, 5, 6, 8])
freqs_sampled = np.delete(freqs_sampled, [4, 5, 6, 8])

token_categories = ['keyword', 'identifier', 'literal', 'operator', 'delimiters', 'comment', 'whitespace']

# Plot the distributions
plot_category_distribution(freqs_orig, freqs_sampled, title="Token Category Distribution", save_name=f"{args.dataset}-{args.dataset_key}-token-category-distribution")

# get the proportions
freqs_orig = freqs_orig / np.sum(freqs_orig)
freqs_sampled = freqs_sampled / np.sum(freqs_sampled)

# print the frequencies in a pandas dataframe
df = pd.DataFrame({'token_category': token_categories, 'human': freqs_orig, 'machine': freqs_sampled})
df['human'] = df['human'].apply(lambda x: f'{x:.4f}')
df['machine'] = df['machine'].apply(lambda x: f'{x:.4f}')
print(df.to_latex(index=False))
print(args.dataset_key)


data_t10 = {
    'token_category': ['keyword', 'identifier', 'literal', 'operator', 'symbol', 'comment', 'whitespace'],
    'human': [0.0434, 0.4008, 0.1007, 0.0575, 0.2318, 0.0656, 0.1002],
    'machine': [0.0462, 0.3643, 0.1086, 0.0543, 0.2217, 0.0954, 0.1095]
}

data_t02 = {
    'token_category': ['keyword', 'identifier', 'literal', 'operator', 'symbol', 'comment', 'whitespace'],
    'human': [0.0416, 0.3975, 0.1100, 0.0617, 0.2310, 0.0586, 0.0997],
    'machine': [0.0515, 0.3682, 0.1313, 0.0538, 0.2301, 0.0556, 0.1093]
}

# Function to plot category distributions with adjustments as requested


def plot_category_distribution(data1, data2, save_name="token-category-distribution-stacked"):
    labels = data1['token_category']
    x = np.arange(len(labels))
    width = 0.25
    fontsize = 16  # Increased font size for better readability

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot for t=1.0
    freqs_orig = np.array(data1['human']) / sum(data1['human'])
    freqs_sampled = np.array(data1['machine']) / sum(data1['machine'])
    ax1.bar(x - width/2, freqs_orig, width, label='Human', color='orange', alpha=0.6, edgecolor='orange')
    ax1.bar(x + width/2, freqs_sampled, width, label='Machine', color='green', alpha=0.6, edgecolor='green')
    # ax1.text(0.98, 0.98, 'T=1.0', transform=ax1.transAxes, fontsize=fontsize, verticalalignment='top', horizontalalignment='right')
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Proportion at T$=0.2$', fontsize=fontsize)
    ax1.legend(fontsize=fontsize)
    # y tick size
    ax1.set_yticklabels([0, '', 0.1, '', 0.2, '', 0.3, '', 0.4], size=fontsize)

    # Plot for t=0.2
    freqs_orig = np.array(data2['human']) / sum(data2['human'])
    freqs_sampled = np.array(data2['machine']) / sum(data2['machine'])
    ax2.bar(x - width/2, freqs_orig, width, label='Human', color='orange', alpha=0.6, edgecolor='orange')
    ax2.bar(x + width/2, freqs_sampled, width, label='Machine', color='green', alpha=0.6, edgecolor='green')
    # ax2.text(0.98, 0.98, 'T=0.2', transform=ax2.transAxes, fontsize=fontsize, verticalalignment='top', horizontalalignment='right')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=fontsize, rotation=45)
    # bold the x-axis labels
    # for label in ax2.get_xticklabels():
    #     label.set_fontweight('bold')
    ax2.set_ylabel('Proportion at T$=1.0$', fontsize=fontsize)
    # ax2.legend(fontsize=fontsize)
    ax2.set_yticklabels([0, '', 0.1, '', 0.2, '', 0.3, '', 0.4], size=fontsize)

    # plt.xlabel('Token Category', fontsize=fontsize)
    plt.tight_layout()

    # add grid in gray for both x and y axis
    ax1.grid(axis='y', linestyle='-', alpha=0.4)
    ax1.grid(axis='x', linestyle='-', alpha=0.4)
    ax2.grid(axis='y', linestyle='-', alpha=0.4)
    ax2.grid(axis='x', linestyle='-', alpha=0.4)

    plt.savefig(f'./figures/category_distribution.pdf')
    plt.show()


# Now we call the function with the provided data and adjustments
plot_category_distribution(data_t02, data_t10)
