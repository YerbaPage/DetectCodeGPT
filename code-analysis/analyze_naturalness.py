# %%
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from baselines.utils.preprocessing import preprocess_and_save
from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model
# from baselines.sample_generate.generate import generate_data
from baselines.all_baselines import run_all_baselines
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import torch


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
    'dataset_key': "CodeLlama-7b-hf-10000-tp1.0",
    'pct_words_masked': 0.5,
    'pct_identifiers_masked': 0.75,
    'span_length': 2,
    'n_samples': 500,
    'n_perturbation_list': "50",
    'n_perturbation_rounds': 1,
    'base_model_name': "codellama/CodeLlama-7b-hf",
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
    'perturb_type': "random-insert-space+newline", # half of the examples will have newline, half will have new space
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

cache_dir, base_model_name, SAVE_FOLDER = preprocess_and_save(args)
model_config = {}
model_config['cache_dir'] = cache_dir

# mask filling t5 model
# model_config = load_mask_filling_model(args, mask_filling_model_name, model_config)

logger.info(f'args: {args}')

model_config = load_base_model_and_tokenizer(args, model_config)

tokenizer = model_config['base_tokenizer']

from tqdm import tqdm
import multiprocessing
from token_tagging import *

token_categories = ['keyword', 'identifier', 'literal', 'operator', 'statement_function', 'data_structure', 'module_block', 'delimiter_access', 'other', 'comment', 'whitespace']
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

            # add the item if it's not too long (128 tokens)
            # if len(line['solution'].split()) <= 128 and len(line['output'].split()) <= 128:
            #     all_originals.append(line['solution'])
            #     all_samples.append(line['output'])

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
            # TODO: may need to modify this if we implement comment generation in the future
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

            
            # TODO: may need to filter out examples with too many repeated lines or n-grams

            # cut to 128 tokens
            all_originals.append(' '.join(line['solution'].split(' ')[:max_len]))
            all_samples.append(' '.join(line['output'].split(' ')[:max_len]))

    logger.info(f'{max_def_num_count} examples have more than {max_def_num} "def"')
    logger.info(f'{min_len_count} examples have less than {min_len} words')
    logger.info(f'{max_comment_num_count} examples have more than {max_comment_num} comments')
    logger.info(f'{max_todo_num_count} examples have more than {max_todo_num} TODOs')
    logger.info(f'{function_comment_num_count} examples have more than 1 function comment')
    logger.info(f'Loaded {len(all_originals)} examples after filtering, and will return {min(max_num, len(all_originals))} examples')

    # import random
    # random.seed(42)
    # random.shuffle(all_originals)
    # random.shuffle(all_samples)

    data = {
        "original": all_originals[:max_num],
        "sampled": all_samples[:max_num]
    }

    return data

data = generate_data(args.dataset, args.dataset_key, max_num=args.n_samples, min_len=args.min_len, max_len=args.max_len, max_comment_num=args.max_comment_num, max_def_num=args.max_def_num, cut_def=args.cut_def, max_todo_num=args.max_todo_num)

logger.info(f'Original: {data["original"][0]}')
logger.info(f'Sampled: {data["sampled"][0]}')

import torch
import torch.nn.functional as F

def compute_model_outputs(text, args, model_config):
    device_num = torch.cuda.device_count() - 1

    with torch.no_grad():
        if ('13b' in model_config['base_model'].config.name_or_path) or ('20b' in model_config['base_model'].config.name_or_path):
            tokenized = model_config['base_tokenizer'](text, return_tensors="pt").to(f'cuda:{device_num}')
        else:
            tokenized = model_config['base_tokenizer'](text, return_tensors="pt").to(args.DEVICE)

        labels = tokenized['input_ids']
        # drop the "token_type_ids" key if it exists
        if 'token_type_ids' in tokenized.keys():
            tokenized.pop('token_type_ids')
        model_outputs = model_config['base_model'](**tokenized, labels=labels)
        logits = model_outputs.logits[:, :-1]
        ll = -model_outputs.loss.item()
        # logits = model_config['base_model'](**tokenized).logits[:, :-1]
        # ll = -model_config['base_model'](**tokenized, labels=labels).loss.item()

    return tokenized, logits, labels, ll


def compute_metrics(text_list, args, model_config):
    log_likelihoods = []
    entropies = []
    ranks = []
    log_ranks = []

    for text in tqdm(text_list):
        tokenized, logits, labels, ll = compute_model_outputs(text, args, model_config)

        # Compute entropy
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        entropy = -neg_entropy.sum(-1).mean().item()

        # Compute rank
        matches = (logits.argsort(-1, descending=True) == labels[:, 1:].unsqueeze(-1)).nonzero()
        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
        current_ranks, timesteps = matches[:, -1], matches[:, -2]
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"
        current_ranks = current_ranks.float() + 1  # convert to 1-indexed rank
        rank = current_ranks.float().mean().item()
        log_rank = torch.log(current_ranks).float().mean().item()

        # Store results
        log_likelihoods.append(ll)
        entropies.append(entropy)
        ranks.append(rank)
        log_ranks.append(log_rank)

    return log_likelihoods, entropies, ranks, log_ranks

from baselines.utils.run_baseline import get_roc_metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# Calculate metrics for original and sampled data
orig_ll, orig_entropy, orig_rank, orig_log_rank = compute_metrics(data['original'], args, model_config)
sampled_ll, sampled_entropy, sampled_rank, sampled_log_rank = compute_metrics(data['sampled'], args, model_config)

def vislualize_distribution(predictions, title, ax, xlabel):

    ax.hist(predictions['real'], bins=50, density=True, alpha=0.5, color='orange', edgecolor='orange', label='Human')
    ax.hist(predictions['samples'], bins=50, density=True, alpha=0.5, color='green', edgecolor='green', label='Machine')

    mu, std = norm.fit(predictions['real'])
    x = np.linspace(min(predictions['real'], predictions['samples']), max(predictions['real'], predictions['samples']), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, linewidth=3, color='orange')
    mu, std = norm.fit(predictions['samples'])
    x = np.linspace(min(predictions['samples']), max(predictions['samples']), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, linewidth=3, color='green')

    # add gray grids
    ax.grid(b=True, which='major', color='gray', linestyle='-', alpha=0.4)

    if 'rank' in title.lower():
        ax.set_xlim(0, 2.0)
    else:
        ax.set_xlim(-4.0, 0)

    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()


# Visualize the distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# larger font size
plt.rcParams.update({'font.size': 16})

# fig.suptitle("Distributions of Metrics for Original and Sampled Data")

predictions = {
    'real': orig_ll,
    'samples': sampled_ll
}
_, _, auc_ll = get_roc_metrics(predictions['real'], predictions['samples'])
vislualize_distribution(predictions, f"Log Likelihood AUC = {auc_ll:.4f}", axes[0], xlabel="Log Likelihood")

predictions = {
    'real': [x for x in orig_log_rank],
    'samples': [x for x in sampled_log_rank]
}
_, _, auc_log_rank = get_roc_metrics(predictions['real'], predictions['samples'])
vislualize_distribution(predictions, f"Log Rank AUC = {auc_log_rank:.4f}", axes[1], xlabel="Log Rank")

plt.tight_layout()
plt.savefig(f'figures/naturalness_distribution.pdf')
plt.show()

# use pprint to print the auc results in a more readable format
import prettytable as pt
auc_table = pt.PrettyTable()
auc_table.field_names = ["Metric", "AUC"]
auc_table.add_row(["Log Likelihood", f"{auc_ll:.4f}"])
auc_table.add_row(["Log Rank", f"{auc_log_rank:.4f}"])
print(auc_table)


