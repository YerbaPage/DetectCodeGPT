from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import math
from baselines.utils.run_baseline import get_roc_metrics, get_precision_recall_metrics, get_accurancy, run_baseline_threshold_experiment
import functools
import torch
from baselines.supervised import eval_supervised
from baselines.entropy import get_entropy
from baselines.rank import get_ranks, get_rank
from baselines.loss import get_ll, get_lls
import random
import re
import numpy as np
from identifier_tagging import get_identifier
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
    'dataset': "TheVault",
    # 'dataset': "CodeSearchNet",
    'dataset_key': "CodeLlama-7b-hf-10000-tp0.2",
    # 'dataset_key': "CodeLlama-7b-hf-10000-tp1.0",
    'pct_words_masked': 0.5,
    'pct_identifiers_masked': 0.75,
    'span_length': 2,
    'n_samples': 500,
    'n_perturbation_list': "50",
    'n_perturbation_rounds': 1,
    'base_model_name': "codellama/CodeLlama-7b-hf",
    'mask_filling_model_name': "Salesforce/codet5p-770m",
    'batch_size': 50,
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
    'perturb_type': "random-insert-space+newline",
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
model_config = load_mask_filling_model(args, mask_filling_model_name, model_config)

logger.info(f'args: {args}')

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

    logger.info(f'{max_def_num_count} examples have more than {max_def_num} "def"')
    logger.info(f'{min_len_count} examples have less than {min_len} words')
    logger.info(f'{max_comment_num_count} examples have more than {max_comment_num} comments')
    logger.info(f'{max_todo_num_count} examples have more than {max_todo_num} TODOs')
    logger.info(f'{function_comment_num_count} examples have more than 1 function comment')
    logger.info(f'Loaded {len(all_originals)} examples after filtering, and will return {min(max_num, len(all_originals))} examples')

    # statistical analysis
    # import random
    # random.seed(42)
    # random.shuffle(all_originals)
    # random.shuffle(all_samples)

    data = {
        "original": all_originals[:max_num],
        "sampled": all_samples[:max_num]
    }

    return data


data = generate_data(args.dataset, args.dataset_key, max_num=args.n_samples, min_len=args.min_len, max_len=args.max_len,
                     max_comment_num=args.max_comment_num, max_def_num=args.max_def_num, cut_def=args.cut_def, max_todo_num=args.max_todo_num)

logger.info(f'Original: {data["original"][0]}')
logger.info(f'Sampled: {data["sampled"][0]}')

pattern = re.compile(r"<extra_id_\d+>")
pattern_with_space = re.compile(r" <extra_id_\d+> ")


def remove_mask_space(text):
    # find all the mask positions " <extra_id_\d+> ", and remove the space before and after the mask
    matches = pattern_with_space.findall(text)
    for match in matches:
        text = text.replace(match, match.strip())
    return text


def tokenize_and_mask_identifiers(text, args, span_length, pct, ceil_pct=False, buffer_size=1):

    varnames, pos = get_identifier(text, 'python')

    mask_string = ' <<<mask>>> '
    sampled = random.sample(varnames, int(len(varnames)*1))
    logger.info(f"Sampled {len(sampled)} identifiers to mask: {sampled}")

    # Split the text into lines
    lines = text.split('\n')

    # replacements will change the line length, so we need to start from the end
    pos.sort(key=lambda pos: (-pos[0][0], -pos[0][1]))

    # Process each position
    for start, end in pos:
        # Extract line number and pos in line
        line_number, start_pos = start
        _, end_pos = end

        # mask the identifier if it is in the sampled list
        if lines[line_number][start_pos:end_pos] in sampled:
            # Replace the identified section in the line
            lines[line_number] = lines[line_number][:start_pos] + mask_string + lines[line_number][end_pos:]

    # Join the lines back together
    masked_text = '\n'.join(lines)
    # logger.info(f'masked_text: \n{masked_text}')

    tokens = masked_text.split(' ')
    # logger.info(f'tokens: \n{tokens}')

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        # logger.info(f'idx: {idx}, token: {token}')
        if token == mask_string.strip():
            # logger.info(f'filling in {token} with <extra_id_{num_filled}>')
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1

    text = ' '.join(tokens)  # before removing the space before and after the mask

    text = remove_mask_space(text)
    # logger.info(f'text: \n{text}')
    return text


def tokenize_and_mask(text, args, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    # count the number of masks in each text with the pattern "<extra_id_\d+>"
    pattern = re.compile(r"<extra_id_\d+>")
    n_expected = [len(pattern.findall(x)) for x in texts]
    return n_expected


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, model_config, args):
    n_expected = count_masks(texts)
    stop_id = model_config['mask_tokenizer'].encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = model_config['mask_tokenizer'](texts, return_tensors="pt", padding=True).to(args.DEVICE)
    # tokens = model_config['mask_tokenizer'](texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.DEVICE)
    outputs = model_config['mask_model'].generate(**tokens, max_length=512, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id, temperature=args.mask_temperature)
    return model_config['mask_tokenizer'].batch_decode(outputs, skip_special_tokens=False)


def apply_extracted_fills(masked_texts, extracted_fills):
    n_expected = count_masks(masked_texts)
    texts = []
    # logger.info(f"n_expected: {n_expected}")

    for idx, (text, fills, n) in enumerate(zip(masked_texts, extracted_fills, n_expected)):
        if len(fills) < n:
            texts.append('')
        else:
            for fill_idx in range(n):
                text = text.replace(f"<extra_id_{fill_idx}>", fills[fill_idx])
            texts.append(text)

    # logger.info(f"texts: {texts}")
    return texts


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def perturb_texts_(texts, args,  model_config, ceil_pct=False):
    span_length = args.span_length
    pct = args.pct_words_masked
    lambda_poisson = args.span_length
    if args.perturb_type == 'random':
        masked_texts = [tokenize_and_mask(x, args, span_length, pct, ceil_pct) for x in texts]
    elif args.perturb_type == 'identifier-masking':
        masked_texts = [tokenize_and_mask_identifiers(x, args, span_length, pct, ceil_pct) for x in texts]
    elif args.perturb_type == 'random-line-shuffle':
        perturbed_texts = [random_line_shuffle(x, pct) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-newline':
        perturbed_texts = [random_insert_newline(x, pct, lambda_poisson) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-space':
        perturbed_texts = [random_insert_space(x, pct, lambda_poisson) for x in texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-space-newline':
        perturbed_texts = [random_insert_space(x, pct, lambda_poisson) for x in texts]
        perturbed_texts = [random_insert_newline(x, pct, lambda_poisson) for x in perturbed_texts]
        return perturbed_texts
    elif args.perturb_type == 'random-insert-space+newline':
        perturbed_texts_part1 = [random_insert_space(x, pct, lambda_poisson) for x in texts]
        perturbed_texts_part2 = [random_insert_newline(x, pct, lambda_poisson) for x in texts]
        total_num = len(perturbed_texts_part1)
        n1 = int(total_num / 2)
        n2 = total_num - n1
        perturbed_texts_part1 = perturbed_texts_part1[:n1]
        perturbed_texts_part2 = perturbed_texts_part2[:n2]
        return perturbed_texts_part1 + perturbed_texts_part2
    else:
        raise ValueError(f'Unknown perturb_type: {args.perturb_type}')

    raw_fills = replace_masks(masked_texts, model_config, args)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, args, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts, model_config, args)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

        # If it fails for more than 50 texts, then we use the original texts as perturbed texts and inform the user with warning
        if attempts > 50:
            logger.warning(f'WARNING: {len(idxs)} texts have no fills. Using the original texts as perturbed texts.')
            for idx in idxs:
                perturbed_texts[idx] = texts[idx]

    logger.info(f'texts: {texts[0]}')
    logger.info(f'perturbed_texts: {perturbed_texts[0]}')

    return perturbed_texts


def perturb_texts(texts, args,  model_config,  ceil_pct=False):

    def perturb_texts_once(texts, args,  model_config,  ceil_pct=False):

        chunk_size = args.chunk_size
        if '11b' in args.mask_filling_model_name:
            chunk_size //= 2

        outputs = []
        for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
            outputs.extend(perturb_texts_(texts[i:i + chunk_size], args, model_config, ceil_pct=ceil_pct))

        return outputs

    for i in range(args.n_perturbation_rounds):
        texts = perturb_texts_once(texts, args, model_config, ceil_pct=ceil_pct)

    return texts


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def random_line_shuffle(text, pct=0.3):
    '''
    randomly exchange the order of two adjacent lines for pct of the lines, except for the first and last line
    '''
    lines = text.split('\n')
    n_lines = len(lines)
    n_shuffled = int(n_lines * pct)
    shuffled_idxs = np.random.choice(n_lines, n_shuffled, replace=False)
    for idx in shuffled_idxs:
        if idx == n_lines - 1 or idx == 0:
            continue
        lines[idx], lines[idx+1] = lines[idx+1], lines[idx]
    return '\n'.join(lines)


def random_insert_newline(text, pct=0.3, mean=1):
    '''
    randomly insert a newline for pct of the lines
    '''
    lines = text.split('\n')
    n_lines = len(lines)
    n_inserted = int(n_lines * pct)
    inserted_idxs = np.random.choice(n_lines, n_inserted, replace=False)
    for idx in inserted_idxs:
        n_newlines = 1
        # n_newlines = scipy.stats.poisson.rvs(mean) + 1
        lines[idx] = lines[idx] + '\n'*n_newlines
    return '\n'.join(lines)


def random_insert_space(text, pct=0.3, mean=1):
    '''
    randomly insert a space for pct of the lines
    '''
    tokens = text.split(' ')
    n_tokens = len(tokens)
    n_inserted = int(n_tokens * pct)
    inserted_idxs = np.random.choice(n_tokens, n_inserted, replace=False)
    for idx in inserted_idxs:
        n_spaces = scipy.stats.poisson.rvs(mean) + 1
        # n_spaces = 1
        tokens[idx] = tokens[idx] + ' '*n_spaces
    return ' '.join(tokens)


ceil_pct = False
texts = ['''
def remove_mask_space(text, args, **kwargs):
    # find all the mask positions " <extra_id_\d+> ", and remove the space before and after the mask
    pattern = re.compile(r" <extra_id_\d+> ")
    matches = pattern.findall(text)
    for match in matches:
        text = text.replace(match, match.strip())
    return text
''']
span_length = args.span_length
pct = args.pct_words_masked
lambda_poisson = args.span_length

if args.perturb_type == 'random':
    masked_texts = [tokenize_and_mask(x, args, span_length, pct, ceil_pct) for x in texts]
elif args.perturb_type == 'identifier-masking':
    masked_texts = [tokenize_and_mask_identifiers(x, args, span_length, pct, ceil_pct) for x in texts]
elif args.perturb_type == 'random-line-shuffle':
    masked_texts = [random_line_shuffle(x, pct) for x in texts]
elif args.perturb_type == 'random-insert-newline':
    masked_texts = [random_insert_newline(x, pct, lambda_poisson) for x in texts]
elif args.perturb_type == 'random-insert-space':
    masked_texts = [random_insert_space(x, pct, lambda_poisson) for x in texts]
elif args.perturb_type == 'random-insert-space-newline':
    masked_texts = [random_insert_space(x, pct, lambda_poisson) for x in texts]
    masked_texts = [random_insert_newline(x, pct, lambda_poisson) for x in masked_texts]
elif args.perturb_type == 'random-insert-space+newline':
    perturbed_texts_part1 = [random_insert_space(x, pct, lambda_poisson) for x in texts]
    perturbed_texts_part2 = [random_insert_newline(x, pct, lambda_poisson) for x in texts]
    total_num = len(perturbed_texts_part1)
    n1 = int(total_num / 2)
    n2 = total_num - n1
    perturbed_texts_part1 = perturbed_texts_part1[:n1]
    perturbed_texts_part2 = perturbed_texts_part2[:n2]
    masked_texts = perturbed_texts_part1 + perturbed_texts_part2
else:
    raise ValueError(f'Unknown perturb_type: {args.perturb_type}')

raw_fills = replace_masks(masked_texts, model_config, args)
extracted_fills = extract_fills(raw_fills)
perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

logger.info(f'original texts: {texts[0]}')
logger.info(f'masked_texts: {masked_texts[0]}')
logger.info(f'perturbed_texts: {perturbed_texts[0]}')

# from baselines.detectGPT import perturb_texts

original_text = data["original"]
sampled_text = data["sampled"]

perturb_fn = functools.partial(perturb_texts, args=args, model_config=model_config)  # perturbation function
p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(max(n_perturbation_list))])  # perturb sampled text
p_original_text = perturb_fn([x for x in original_text for _ in range(max(n_perturbation_list))])  # perturb original text

results = []
for idx in range(len(original_text)):
    results.append({
        "original": original_text[idx],
        "sampled": sampled_text[idx],
        "perturbed_sampled": p_sampled_text[idx * max(n_perturbation_list): (idx + 1) * max(n_perturbation_list)],
        "perturbed_original": p_original_text[idx * max(n_perturbation_list): (idx + 1) * max(n_perturbation_list)]
    })

selected_index = 1
selected_perturb = 3

print(original_text[selected_index])
# p_original_text[:5]
print(p_original_text[int(args.n_perturbation_list)*selected_index+selected_perturb])
# print the difference between the original and perturbed text
print("\nDifference between original and perturbed text:")
print([x for x in p_original_text[int(args.n_perturbation_list)*selected_index+selected_perturb].split(' ') if x not in original_text[selected_index].split(' ')])

# show the length of the original and perturbed text
print(f"original text length: {len(original_text)}")
print(f"perturbed text length: {len(p_original_text)}")

model_config['mask_model'] = model_config['mask_model'].cpu()
torch.cuda.empty_cache()

# start to load the base scoring model
model_config = load_base_model_and_tokenizer(args, model_config)


for res in tqdm(results, desc="Computing unperturbed log likelihoods"):
    res["original_ll"] = get_ll(res["original"], args, model_config)
    res["sampled_ll"] = get_ll(res["sampled"], args, model_config)

for res in tqdm(results, desc="Computing unperturbed log rank"):
    res["original_logrank"] = get_rank(res["original"], args, model_config, log=True)
    res["sampled_logrank"] = get_rank(res["sampled"], args, model_config, log=True)

for res in tqdm(results, desc="Computing perturbed log likelihoods"):
    p_sampled_ll = get_lls(res["perturbed_sampled"], args, model_config)
    p_original_ll = get_lls(res["perturbed_original"], args, model_config)

    for n_perturbation in n_perturbation_list:
        res[f"perturbed_sampled_ll_{n_perturbation}"] = np.mean([i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)])
        res[f"perturbed_original_ll_{n_perturbation}"] = np.mean([i for i in p_original_ll[:n_perturbation] if not math.isnan(i)])
        res[f"perturbed_sampled_ll_std_{n_perturbation}"] = np.std([i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)]) if len([
            i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)]) > 1 else 1
        res[f"perturbed_original_ll_std_{n_perturbation}"] = np.std([i for i in p_original_ll[:n_perturbation] if not math.isnan(i)]) if len([
            i for i in p_original_ll[:n_perturbation] if not math.isnan(i)]) > 1 else 1

for res in tqdm(results, desc="Computing perturbed log rank"):
    p_sampled_rank = get_ranks(res["perturbed_sampled"], args, model_config, log=True)
    p_original_rank = get_ranks(res["perturbed_original"], args, model_config, log=True)
    for n_perturbation in n_perturbation_list:
        res[f"perturbed_sampled_logrank_{n_perturbation}"] = np.mean([i for i in p_sampled_rank[:n_perturbation] if not math.isnan(i)])
        res[f"perturbed_original_logrank_{n_perturbation}"] = np.mean([i for i in p_original_rank[:n_perturbation] if not math.isnan(i)])

torch.cuda.empty_cache()

print(len(results))  # corresponds to the number of samples, and the result of each sample is stored in a dictionary
print(results[0].keys())  # corresponds to the computed metrics of for each sample


def vislualize_distribution(predictions, title, ax):

    # remove the nans in predictions (the pairs will be removed together)
    reals = []
    samples = []
    for i in range(len(predictions['real'])-1, -1, -1):
        if math.isnan(predictions['real'][i]) or math.isnan(predictions['samples'][i]):
            continue
        else:
            reals.append(predictions['real'][i])
            samples.append(predictions['samples'][i])
    predictions['real'] = reals
    predictions['samples'] = samples

    ax.hist(predictions['real'], bins=30, density=True, alpha=0.5, color='orange', edgecolor='orange', label='Real')
    ax.hist(predictions['samples'], bins=30, density=True, alpha=0.5, color='green', edgecolor='green', label='Samples')

    mu, std = norm.fit(predictions['real'])
    x = np.linspace(min(predictions['real'], predictions['samples']), max(predictions['real'], predictions['samples']), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, linewidth=3, color='orange')
    mu, std = norm.fit(predictions['samples'])
    x = np.linspace(min(predictions['samples']), max(predictions['samples']), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, linewidth=3, color='green')

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# if baseline == 'logrank':
predictions = {'real': [], 'samples': []}
for res in results:
    predictions['real'].append(-res['original_logrank'])
    predictions['samples'].append(-res['sampled_logrank'])
_, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])

print(f"ROC AUC of logrank: {roc_auc}")
vislualize_distribution(predictions, f'Logrank AUC = {roc_auc}', axs[0, 0])


# if baseline == 'LRR':
predictions = {'real': [], 'samples': []}
for res in results:
    predictions['real'].append(-res['original_ll']/res['original_logrank'])
    predictions['samples'].append(-res['sampled_ll']/res['sampled_logrank'])
_, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
print(f'ROC AUC of LRR: {roc_auc}')
vislualize_distribution(predictions, f'LRR AUC = {roc_auc}', axs[0, 1])

# if baseline == 'NPR':
predictions = {'real': [], 'samples': []}
for res in results:
    predictions['real'].append(res[f'perturbed_original_logrank_{n_perturbation}']/res["original_logrank"])
    predictions['samples'].append(res[f'perturbed_sampled_logrank_{n_perturbation}']/res["sampled_logrank"])
_, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
print(f'ROC AUC of NPR: {roc_auc}')
vislualize_distribution(predictions, f'NPR AUC = {roc_auc}', axs[1, 0])

# if baseline == 'DetectGPT':
predictions = {'real': [], 'samples': []}
for res in results:

    real_comp = (res['original_ll'] - res[f'perturbed_original_ll_{n_perturbation}']) / res[f'perturbed_original_ll_std_{n_perturbation}']
    sample_comp = (res['sampled_ll'] - res[f'perturbed_sampled_ll_{n_perturbation}']) / res[f'perturbed_sampled_ll_std_{n_perturbation}']

    # avoid nan
    if math.isnan(real_comp) or math.isnan(sample_comp):
        logger.warning(f"NaN detected, skipping")
        continue

    predictions['real'].append(real_comp)
    predictions['samples'].append(sample_comp)
_, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
print(f'ROC AUC of DetectGPT: {roc_auc}')
vislualize_distribution(predictions, f'DetectGPT AUC = {roc_auc}', axs[1, 1])

plt.tight_layout()
plt.savefig('results.pdf')
