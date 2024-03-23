from collections import defaultdict
import multiprocessing
import matplotlib.pyplot as plt
from collections import Counter
from transformers import RobertaTokenizer, AutoTokenizer
import argparse
from loguru import logger
from tree_sitter import Language, Parser
import os
from tqdm import tqdm
import json
import numpy as np
import pdb
import sys
sys.setrecursionlimit(5000)


if not os.path.exists('build/my-languages.so'):
    Language.build_library(
        # Store the library in the `build` directory
        'build/my-languages.so',

        # Include one or more languages
        [
            './tree-sitter/tree-sitter-python',
            './tree-sitter/tree-sitter-java',
            './tree-sitter/tree-sitter-php',
            './tree-sitter/tree-sitter-go',
            './tree-sitter/tree-sitter-ruby',
            './tree-sitter/tree-sitter-javascript',
        ]
    )
else:
    logger.info('build/my-languages.so already exists, skip building')

PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
PHP_LANGUAGE = Language('build/my-languages.so', 'php')
GO_LANGUAGE = Language('build/my-languages.so', 'go')
RUBY_LANGUAGE = Language('build/my-languages.so', 'ruby')
JAVASCRIPT_LANGUAGE = Language('build/my-languages.so', 'javascript')

# map from language to tree-sitter language
LANGUAGE_MAP = {
    'java': JAVA_LANGUAGE,
    'python': PYTHON_LANGUAGE,
    'php': PHP_LANGUAGE,
    'go': GO_LANGUAGE,
    'ruby': RUBY_LANGUAGE,
    'javascript': JAVASCRIPT_LANGUAGE,
}

parser = Parser()


def get_token_from_position(code_string, start_point, end_point):
    lines = code_string.splitlines()
    token = lines[start_point[0]][start_point[1]:end_point[1]]
    return token


def get_position_as_integer(code, row, col):
    # Converts row, col format to an integer position
    return sum(len(line) + 1 for line in code.splitlines()[:row]) + col  # +1 for newline chars


token_categories = {
    "keyword": ["def", "return", "else", "if", "in", "for", "while", "break", "continue", "pass", "yield", "raise", "import", "from", "as", "global", "nonlocal", "assert", "with", "lambda", "else_clause", "for_in_clause", "try_clause", "except_clause", "finally_clause", "try", "except", "elif", "async", "await", "case", "class", "print", "match", "del", "finally"],
    "identifier": ["identifier", "type_identifier"],
    "literal": ["string_content", "integer", "true", "false", "none", "decimal_integer_literal", "string_literal", "string_fragment", "character_literal", "escape_sequence", "float"],
    "operator": [">", "<", "=", "==", "+", "-", "*", "/", "%", "!", "not", "and", "or", "comparison_operator", ">=", "<=", "is", "is not", "not in", "|", "!=", "&", "^", "**", "//", "<<", ">>", "@", "~", "&=", "^=", "**=", "//=", "/=", "<<=", ">>=", "|=", "->", "<>", '%=', '*=', '+=', '-=', ":="],
    "statement_function": ["expression_statement", "if_statement", "return_statement", "assignment", "print_statement", "while_statement", "for_statement", "break_statement", "continue_statement", "pass_statement", "yield_statement", "raise_statement", "import_statement", "assignment_expression", "binary_expression", "call", "function_definition", "argument_list", "parameters", "decorator", "method_definition", "method_invocation", "exec"],
    "data_structure": ["list_comprehension", "dictionary", "tuple", "set", "array_access"],
    "module_block": ["module", "block", "class_definition", "program", "__future__"],
    "delimiters": [":", ")", "]", "[", "(", ",", "\"", "'", ";", "{", "}", ".", "array_access"],
    "comment": ["comment"],
    "other": ["ellipsis", "type_conversion", "ERROR"]
}


def get_tagged_tokens(code, lang):
    parser.set_language(LANGUAGE_MAP[lang])
    tree = parser.parse(bytes(code, 'utf-8'))

    tagged_tokens_list = []
    others = set()

    def visit_node(node):
        # If node is a leaf (has no children), process the node
        if len(node.children) == 0:
            try:
                token = get_token_from_position(code, (node.start_point[0], node.start_point[1]), (node.end_point[0], node.end_point[1]))
                category_found = False
                for token_type, token_category_list in token_categories.items():
                    if node.type in token_category_list:
                        category = token_type
                        category_found = True
                        break

                if not category_found:
                    category = "whitespace"
                    others.add(node.type)

                start_pos = get_position_as_integer(code, node.start_point[0], node.start_point[1])
                end_pos = get_position_as_integer(code, node.end_point[0], node.end_point[1])
                tagged_tokens_list.append((token, start_pos, end_pos, category))
            except:
                logger.warning(f'Error when processing code: {code}')
                pass

            # logger.debug(f'Found token: {token}, start_pos: {start_pos}, end_pos: {end_pos}, category: {category}, node_type: {node.type}')

        # Always visit child nodes (to reach the leaves)
        for child in node.children:
            visit_node(child)

    visit_node(tree.root_node)

    if len(others) > 0:
        logger.info(f'Other categories: {others}')

    # remove those empty tokens
    tagged_tokens_list = [t for t in tagged_tokens_list if t[0] != '']

    return tagged_tokens_list, others


def load_data(dataset, key, max_num=200, min_len=1, max_len=128, max_comment_num=10, max_def_num=10, cut_def=False, max_todo_num=3):

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

    max_num = min(max_num, len(all_originals))

    logger.info(f'{max_def_num_count} examples have more than {max_def_num} "def"')
    logger.info(f'{min_len_count} examples have less than {min_len} words')
    logger.info(f'{max_comment_num_count} examples have more than {max_comment_num} comments')
    logger.info(f'{max_todo_num_count} examples have more than {max_todo_num} TODOs')
    logger.info(f'{function_comment_num_count} examples have more than 1 function comment')
    logger.info(f'Loaded {len(all_originals)} examples after filtering, and will return {max_num} examples')

    data = {
        "original": all_originals[:max_num],
        "sampled": all_samples[:max_num]
    }

    return data


def get_positions_for_huggingface_tokens(code, tokens):
    positions = []
    start_index = 0
    for token in tokens:
        start = code.find(token, start_index)
        if start == -1:  # skip if token not found
            continue
        end = start + len(token)
        positions.append((start, end))
        start_index = end
    return positions


def assign_category(hf_positions, hf_tokens, ts_tokens):
    hf_categories = []

    for hf_start, hf_end in hf_positions:
        overlapping_categories = []
        for ts_token, ts_start, ts_end, ts_category in ts_tokens:
            if hf_end <= ts_start:
                break
            if (hf_start < ts_end and hf_end > ts_start) or (hf_start == ts_start and hf_end == ts_end):
                overlapping_categories.append(ts_category)

        # for now, just use the most common category if there are overlaps
        if overlapping_categories:
            hf_categories.append(max(set(overlapping_categories), key=overlapping_categories.count))
        else:
            hf_categories.append("whitespace")

    # TODO: may need to handle the spaces and newlines in the future

    # Handle ''' and """ cases
    triple_quotes = ["'''", '"""']
    in_comment_block = False
    for idx, (token, (start, end)) in enumerate(zip(hf_tokens, hf_positions)):
        if token in triple_quotes:
            in_comment_block = not in_comment_block
        if in_comment_block:
            if hf_categories[idx] in ["literal", "whitespace"]:
                hf_categories[idx] = "comment"
            else:
                pass
                # logger.warning(f"Token '{token}' between triple quotes categorized as {hf_categories[idx]}")

    return hf_categories


def align_tokens_with_categories(code, lang, tokenizer, return_hf_tokens=False):

    tree_sitter_tokens, unrecognized_categories = get_tagged_tokens(code, lang)  # using the tag_code function you developed earlier
    huggingface_tokens = tokenizer.tokenize(code)
    cleaned_huggingface_tokens = [i.replace('Ġ', '').replace('Ċ', '\n') for i in huggingface_tokens]
    # Getting start and end positions for each huggingface token in the original code
    hf_positions = get_positions_for_huggingface_tokens(code, cleaned_huggingface_tokens)
    # hf_positions[:30]
    # Assigning categories
    categories = assign_category(hf_positions, hf_tokens=cleaned_huggingface_tokens, ts_tokens=tree_sitter_tokens)

    # You can then combine tokens and their categories:
    tagged_hf_tokens_cleaned = list(zip(cleaned_huggingface_tokens, categories))
    # assert len(huggingface_tokens) == len(tagged_hf_tokens_cleaned)
    tagged_hf_tokens = [(hf_token, tag) for hf_token, (hf_token_cleaned, tag) in zip(huggingface_tokens, tagged_hf_tokens_cleaned)]

    # if not len(huggingface_tokens) == len(tagged_hf_tokens):
    # logger.warning(f'len(huggingface_tokens) != len(tagged_hf_tokens), {len(huggingface_tokens)} != {len(tagged_hf_tokens)}, code: {code}')

    # revert the cleaned tokens back to the original tokens by simply matching the hf_tokens and the tags
    if return_hf_tokens:
        return tagged_hf_tokens, unrecognized_categories, huggingface_tokens

    return tagged_hf_tokens, unrecognized_categories


def analyze_identifiers(code_pairs, lang):

    all_human_codes = code_pairs['original']
    all_machine_codes = code_pairs['sampled']

    # counted_human_identifiers = count_identifiers(all_human_codes, lang)
    # counted_machine_identifiers = count_identifiers(all_machine_codes, lang)

    # use multiprocessing to speed up
    counted_human_identifiers = count_identifiers_multiprocessed(all_human_codes, lang)
    counted_machine_identifiers = count_identifiers_multiprocessed(all_machine_codes, lang)

    # save to file, line by line
    with open(f'./figures/{lang}_human_identifiers.txt', 'w') as f:
        for identifier, count in counted_human_identifiers:
            f.write(f'{identifier}\t{count}\n')
    with open(f'./figures/{lang}_machine_identifiers.txt', 'w') as f:
        for identifier, count in counted_machine_identifiers:
            f.write(f'{identifier}\t{count}\n')

    # for the identifiers that appear in both human and machine codes, compute the log rank difference, sort by the difference, and save to file (min occurrence = 10)
    min_occurrence = 10

    # get the intersection of the two sets
    human_identifiers = set([identifier for identifier, count in counted_human_identifiers if count >= min_occurrence])
    machine_identifiers = set([identifier for identifier, count in counted_machine_identifiers if count >= min_occurrence])
    intersection_identifiers = human_identifiers.intersection(machine_identifiers)
    logger.info(f'len(human_identifiers): {len(human_identifiers)}, len(machine_identifiers): {len(machine_identifiers)}, len(intersection_identifiers): {len(intersection_identifiers)}')

    # compute the log rank difference
    log_rank_diff = []
    for identifier in tqdm(intersection_identifiers, ncols=50):
        human_rank = [i for i, (identifier_, count) in enumerate(counted_human_identifiers) if identifier_ == identifier][0]
        machine_rank = [i for i, (identifier_, count) in enumerate(counted_machine_identifiers) if identifier_ == identifier][0]
        log_rank_diff.append((identifier, np.log(human_rank+1) - np.log(machine_rank+1)))

    # sort by the log rank difference
    log_rank_diff = sorted(log_rank_diff, key=lambda x: x[1], reverse=True)

    # save to file
    with open(f'./figures/{lang}_log_rank_diff.txt', 'w') as f:
        for identifier, diff in log_rank_diff:
            f.write(f'{identifier}\t{diff}\n')


def count_identifiers(codes, lang):
    all_identifiers = []

    for code_item in tqdm(codes, ncols=50):
        tree_sitter_tokens, unrecognized_categories = get_tagged_tokens(code_item, lang)
        all_identifiers += [token for token, start, end, category in tree_sitter_tokens if category == 'identifier']

    # use a counter to count the frequency of each identifier and sort them
    counter = Counter(all_identifiers)
    # convert to a dict
    counter = dict(counter)
    # sort the dict by value
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return counter


def count_tokens(codes, lang):
    all_tokens = []

    for code_item in tqdm(codes, ncols=50):
        tree_sitter_tokens, unrecognized_categories = get_tagged_tokens(code_item, lang)
        all_tokens += [token for token, start, end, category in tree_sitter_tokens]

    # use a counter to count the frequency of each identifier and sort them
    counter = Counter(all_tokens)
    # convert to a dict
    counter = dict(counter)
    # sort the dict by value
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return counter


def worker_count_tokens(chunk, lang):
    all_tokens = []
    for code_item in chunk:
        tree_sitter_tokens, _ = get_tagged_tokens(code_item, lang)
        all_tokens += [token for token, _, _, _ in tree_sitter_tokens]
    return all_tokens


def worker_count_identifiers(chunk, lang):
    all_identifiers = []
    for code_item in chunk:
        tree_sitter_tokens, _ = get_tagged_tokens(code_item, lang)
        all_identifiers += [token for token, _, _, category in tree_sitter_tokens if category == 'identifier']
    return all_identifiers


def worker_helper(args):
    return args[0](*args[1:])


def process_data_multithreaded(data, worker_function, lang, num_processes=None):
    if not num_processes:
        num_processes = multiprocessing.cpu_count() - 1

    chunk_size = len(data) // num_processes
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    results = []
    args_list = [(worker_function, chunk, lang) for chunk in chunks]

    with multiprocessing.Pool(processes=num_processes) as pool:
        for chunk_result in tqdm(pool.imap_unordered(worker_helper, args_list), total=len(chunks)):
            results.extend(chunk_result)

    counter = Counter(results)
    counter = dict(counter)
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return counter


def count_tokens_multiprocessed(codes, lang):
    return process_data_multithreaded(codes, worker_count_tokens, lang)


def count_identifiers_multiprocessed(codes, lang):
    return process_data_multithreaded(codes, worker_count_identifiers, lang)


def analyze_token_divergence_from_ideal(token_counts):
    frequencies = [count for token, count in token_counts]
    ranks = range(1, len(frequencies) + 1)

    # Calculate ideal frequencies based on Zipf's law
    C = frequencies[0]
    alpha = 1  # Typically close to 1 for natural languages
    ideal_freqs = [C / (r**alpha) for r in ranks]

    # Calculate divergence for each token
    divergences = [(token, abs(freq - ideal_freqs[idx])) for idx, (token, freq) in enumerate(token_counts)]

    # Sort tokens by divergence
    divergences = sorted(divergences, key=lambda x: x[1], reverse=True)

    return divergences


def check_zipf_law(code_pairs, lang):
    # Get token counts for both human and machine codes
    # human_token_counts = count_tokens(code_pairs['original'], lang)
    # machine_token_counts = count_tokens(code_pairs['sampled'], lang)

    # use multiprocessing to speed up
    human_token_counts = count_tokens_multiprocessed(code_pairs['original'], lang)
    machine_token_counts = count_tokens_multiprocessed(code_pairs['sampled'], lang)

    # save to file, line by line
    with open(f'./figures/{lang}_human_token_freqs.txt', 'w') as f:
        for token, count in human_token_counts:
            f.write(f'{token}\t{count}\n')

    with open(f'./figures/{lang}_machine_token_freqs.txt', 'w') as f:
        for token, count in machine_token_counts:
            f.write(f'{token}\t{count}\n')

    # Plot Zipf's law for human codes
    _plot_zipf(human_token_counts, "human")

    # Plot Zipf's law for machine codes
    _plot_zipf(machine_token_counts, "machine")

    _plot_combined_zipf(human_token_counts, machine_token_counts, "Zipf's Law for Human vs Machine")

    human_divergences = analyze_token_divergence_from_ideal(human_token_counts)
    machine_divergences = analyze_token_divergence_from_ideal(machine_token_counts)

    # write to file
    with open(f'./figures/{lang}_human_divergences.txt', 'w') as f:
        for token, divergence in human_divergences:
            f.write(f'{token}\t{divergence}\n')

    with open(f'./figures/{lang}_machine_divergences.txt', 'w') as f:
        for token, divergence in machine_divergences:
            f.write(f'{token}\t{divergence}\n')

    human_tokens = dict(human_token_counts)
    machine_tokens = dict(machine_token_counts)

    common_tokens = set(human_tokens.keys()).intersection(set(machine_tokens.keys()))

    divergence = 0
    for token in common_tokens:
        divergence += (human_tokens[token] - machine_tokens[token])**2

    divergence = np.sqrt(divergence)

    logger.info(f"Quantitative divergence between tokens of human and machine codes: {divergence}")


def _plot_combined_zipf(human_token_counts, machine_token_counts, title="Combined Zipf's Law"):
    total_human_tokens = sum([count for _, count in human_token_counts])
    total_machine_tokens = sum([count for _, count in machine_token_counts])

    # Normalize machine frequencies to match human token total for comparison
    human_frequencies = [count for _, count in human_token_counts]
    machine_frequencies = [count / total_machine_tokens * total_human_tokens for _, count in machine_token_counts]

    # Taking the logarithms of the frequencies and the ranks
    ranks_human = np.log10(range(1, len(human_frequencies)+1))
    ranks_machine = np.log10(range(1, len(machine_frequencies)+1))
    log_human_frequencies = np.log10(human_frequencies)
    log_machine_frequencies = np.log10(machine_frequencies)

    plt.figure(figsize=(12, 8))

    # Plot with logarithmic values
    plt.plot(ranks_human, log_human_frequencies, marker="o", linestyle="-", color="blue", linewidth=0.5, markersize=3, label="Human")
    plt.plot(ranks_machine, log_machine_frequencies, marker="x", linestyle="-", color="green", linewidth=0.5, markersize=3, label="Machine")

    # Ideal Zipf's law line
    C = max(human_frequencies[0], machine_frequencies[0])
    ideal_freqs = [np.log10(C/(r**1)) for r in range(1, len(human_frequencies)+1)]  # Use log values for the ideal line
    plt.plot(ranks_human, ideal_freqs, linestyle="--", color="red", linewidth=0.5, label="Ideal Zipf's Line")

    plt.xlabel("Log10(Rank)")
    plt.ylabel("Log10(Frequency)")
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.tight_layout()
    plt.savefig(f'./figures/combined_zipfs_law.pdf')
    plt.show()


def _plot_zipf(token_counts, title):
    frequencies = [count for token, count in token_counts]
    ranks = range(1, len(frequencies)+1)

    plt.figure(figsize=(10, 7))

    # Log-log plot
    plt.loglog(ranks, frequencies, marker="o", label="Observed Frequencies")

    # Ideal Zipf's law line
    C = frequencies[0]
    alpha = 1  # Typically close to 1 for natural languages
    ideal_freqs = [C/(r**alpha) for r in ranks]
    plt.loglog(ranks, ideal_freqs, linestyle="--", color="red", label="Ideal Zipf's Line")

    plt.title(f"Zipf's Law: {title}")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Frequency)")
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.tight_layout()
    plt.savefig(f'./figures/{title}_zipf.pdf')


def check_heaps_law(code_pairs, lang):
    # human_code_lengths, human_vocab_sizes = _get_lengths_and_vocab_sizes(code_pairs['original'], lang)
    # machine_code_lengths, machine_vocab_sizes = _get_lengths_and_vocab_sizes(code_pairs['sampled'], lang)

    # use multiprocessing to speed up
    human_code_lengths, human_vocab_sizes = _get_lengths_and_vocab_sizes_multithreaded(code_pairs['original'], lang)
    machine_code_lengths, machine_vocab_sizes = _get_lengths_and_vocab_sizes_multithreaded(code_pairs['sampled'], lang)

    # Get mean vocab sizes for unique code lengths
    human_code_lengths, human_vocab_sizes = get_mean_vocab_sizes(human_code_lengths, human_vocab_sizes)
    machine_code_lengths, machine_vocab_sizes = get_mean_vocab_sizes(machine_code_lengths, machine_vocab_sizes)

    # Plot Heap's law for human codes
    _plot_heaps(human_code_lengths, human_vocab_sizes, "Human")

    # Plot Heap's law for machine codes
    _plot_heaps(machine_code_lengths, machine_vocab_sizes, "Machine")

    plot_heaps_together(human_code_lengths, human_vocab_sizes, machine_code_lengths, machine_vocab_sizes, "Human vs Machine")


def plot_heaps_together(human_lengths, human_vocab, machine_lengths, machine_vocab, title):
    plt.figure(figsize=(10, 7))

    log_human_lengths = np.log10(human_lengths)
    log_human_vocab = np.log10(human_vocab)
    log_machine_lengths = np.log10(machine_lengths)
    log_machine_vocab = np.log10(machine_vocab)

    # Scatter plot
    plt.scatter(log_human_lengths, log_human_vocab, marker="o", label="Human", color="blue", s=3)
    plt.scatter(log_machine_lengths, log_machine_vocab, marker="o", label="Machine", color="green", s=3)

    plt.xlim(0, 3)
    plt.ylim(0, 2)

    plt.xlabel("Log10(Text Length)")
    plt.ylabel("Log10(Vocabulary Size)")
    plt.legend(loc='upper left')
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.tight_layout()
    plt.savefig(f'./figures/combined_heaps_law.pdf')
    plt.show()


def get_mean_vocab_sizes(code_lengths, vocab_sizes):
    length_to_vocab = defaultdict(list)

    for length, vocab in zip(code_lengths, vocab_sizes):
        length_to_vocab[length].append(vocab)

    # Get unique code lengths
    unique_lengths = list(length_to_vocab.keys())

    # Compute mean vocab sizes
    mean_vocab_sizes = [np.mean(length_to_vocab[length]) for length in unique_lengths]

    return unique_lengths, mean_vocab_sizes


def _get_lengths_and_vocab_sizes(codes, lang):
    code_lengths = []
    vocab_sizes = []

    for code_item in tqdm(codes, ncols=50):
        tree_sitter_tokens, _ = get_tagged_tokens(code_item, lang)
        tokens = [token for token, _, _, _ in tree_sitter_tokens]
        code_lengths.append(len(tokens))
        vocab_sizes.append(len(set(tokens)))

    return code_lengths, vocab_sizes


def worker_lengths_and_vocab_sizes(args):
    chunk, lang = args
    code_lengths = []
    vocab_sizes = []
    for code_item in chunk:
        tree_sitter_tokens, _ = get_tagged_tokens(code_item, lang)
        tokens = [token for token, _, _, _ in tree_sitter_tokens]
        code_lengths.append(len(tokens))
        vocab_sizes.append(len(set(tokens)))
    return code_lengths, vocab_sizes


def _get_lengths_and_vocab_sizes_multithreaded(data, lang, num_processes=None):
    if not num_processes:
        num_processes = multiprocessing.cpu_count() - 1

    chunk_size = len(data) // num_processes
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    all_code_lengths = []
    all_vocab_sizes = []
    args_list = [(chunk, lang) for chunk in chunks]

    with multiprocessing.Pool(processes=num_processes) as pool:
        for code_lengths, vocab_sizes in tqdm(pool.imap_unordered(worker_lengths_and_vocab_sizes, args_list), total=len(chunks)):
            all_code_lengths.extend(code_lengths)
            all_vocab_sizes.extend(vocab_sizes)

    return all_code_lengths, all_vocab_sizes


def _plot_heaps(code_lengths, vocab_sizes, title):
    plt.figure(figsize=(10, 7))

    # import pdb
    # pdb.set_trace()

    # Log-log plot
    # plt.loglog(code_lengths, vocab_sizes, marker="o", label="Observed Vocab Sizes")
    log_code_lengths = np.log10(code_lengths)
    log_vocab_sizes = np.log10(vocab_sizes)

    # scatter plot
    # smaller points
    plt.scatter(log_code_lengths, log_vocab_sizes, marker="o", label="Observed Vocab Sizes", s=2)

    # x lim 0~3
    # y lim 0~2
    plt.xlim(0, 3)
    plt.ylim(0, 2)

    # Ideal Heap's law line (optional)
    # To derive the parameters K and beta, a regression on the log-log plot can be performed.
    # Here, we're just plotting the observed values without the ideal line.

    plt.title(f"Heap's Law: {title}")
    plt.xlabel("Log10(Code Length)")
    plt.ylabel("Log10(Vocab Size)")
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.tight_layout()
    plt.savefig(f'./figures/{title}_heaps.pdf')


if __name__ == '__main__':

    max_num = 100000
    code_pairs = load_data(dataset='CodeSearchNet', key='CodeLlama-7b-hf-5000-tp1.0-nostop', max_num=max_num)

    # original_codes = code_pairs['original']
    # code_item = original_codes[10]

    lang = 'python'

    # analyze all the identifiers
    # analyze_identifiers(code_pairs, lang)
    plt.rcParams.update({'font.size': 20})

    # check the zipf's law
    check_zipf_law(code_pairs, lang)

    # check the heap's law
    check_heaps_law(code_pairs, lang)
