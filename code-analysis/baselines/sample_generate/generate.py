
import torch
import numpy as np
import datasets
import random
from . import custom_datasets


def generate_samples(args, model_config, raw_data, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(args, model_config, original_text)

        for o, s in zip(original_text, sampled_text):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(custom_datasets.SEPARATOR, ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)

    return data


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]
# trim to shorter length


def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(args, model_config, texts):
    prompt_tokens = args.prompt_len
    min_words = args.min_words
    DEVICE = args.DEVICE
    # encode each text as a list of token ids
    if args.dataset == 'pubmed':
        texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
        all_encoded = model_config['base_tokenizer'](texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        all_encoded = model_config['base_tokenizer'](texts, return_tensors="pt", padding=True).to(DEVICE)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    decoded = ['' for _ in range(len(texts))]

    # sample from the model until we get a sample with at least min_words words for each example
    # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
    tries = 0
    while (m := min(len(x.split()) for x in decoded)) < min_words:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

        sampling_kwargs = {}
        if args.do_top_p:
            sampling_kwargs['top_p'] = args.top_p
        elif args.do_top_k:
            sampling_kwargs['top_k'] = args.top_k
        min_length = args.min_len
        outputs = model_config['base_model'].generate(**all_encoded, min_length=min_length, max_length=200, temperature=args.temperature, do_sample=True, **sampling_kwargs,
                                                      pad_token_id=model_config['base_tokenizer'].eos_token_id, eos_token_id=model_config['base_tokenizer'].eos_token_id)
        decoded = model_config['base_tokenizer'].batch_decode(outputs, skip_special_tokens=True)
        tries += 1
        if tries > 3:
            break
    return decoded


def generate_data(args, model_config):
    # load data
    dataset = args.dataset
    key = args.dataset_key
    n_samples = args.n_samples
    batch_size = args.batch_size
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, model_config['cache_dir'])
    else:
        data = datasets.load_dataset(dataset, split='train', cache_dir=model_config['cache_dir'])[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.seed(0)
    random.shuffle(data)

    data = data[:5_000]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = model_config['preproc_tokenizer'](data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return generate_samples(args, model_config, data[:n_samples], batch_size=batch_size)


def strip_newlines(text):
    return ' '.join(text.split())
