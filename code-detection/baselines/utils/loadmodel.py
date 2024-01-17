import transformers
import torch
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
import os


def load_base_model_and_tokenizer(args, model_config):
    name = args.base_model_name
    print(f'Loading BASE model {args.base_model_name}...')
    base_model_kwargs = {}
    if ('gpt-j' in name) or ('neox' in name) or ('13b' in name):
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in name:
        base_model_kwargs.update(dict(revision='float16'))

    if '13b' in name:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=model_config['cache_dir'], device_map="auto")
    elif 'codet5p' in name:
        base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=model_config['cache_dir'], device_map="auto", torch_dtype=torch.float16,
                                                                        trust_remote_code=True, decoder_start_token_id=50256, pad_token_id=50256)
    elif '20b' in name:
        config = transformers.AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        with init_empty_weights():
            base_model = transformers.AutoModelForCausalLM.from_config(config)
        base_model = load_checkpoint_and_dispatch(
            base_model,  model_config['cache_dir'], device_map="auto", no_split_module_classes=["GPTNeoXLayer"]
        )
    elif "llama" in name.lower():
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=model_config['cache_dir'], trust_remote_code=True, torch_dtype=torch.float16)
        base_model.to(args.DEVICE)
    else:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=model_config['cache_dir'], trust_remote_code=True)
        base_model.to(args.DEVICE)

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    if 'llama' in name.lower():
        base_tokenizer = transformers.LlamaTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=model_config['cache_dir'])
    else:
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=model_config['cache_dir'])
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    model_config['base_model'] = base_model
    model_config['base_tokenizer'] = base_tokenizer
    return model_config


def load_mask_filling_model(args, mask_filling_model_name, model_config):

    print(f'Loading mask filling model {mask_filling_model_name}...')
    if 'incoder' in mask_filling_model_name:
        mask_model = transformers.AutoModelForCausalLM.from_pretrained(mask_filling_model_name, cache_dir=model_config['cache_dir'])
    else:
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, cache_dir=model_config['cache_dir'])
    # mask_model.parallelize()
    # to device
    mask_model.to(args.DEVICE)

    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512

    # preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=model_config['cache_dir'])
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=model_config['cache_dir'])
    if 'incoder' in mask_filling_model_name:
        mask_tokenizer.pad_token = "<pad>"
        mask_tokenizer.paddding_side = "left"

    # if args.dataset in ['english', 'german']:
        # preproc_tokenizer = mask_tokenizer

    # model_config['preproc_tokenizer'] = preproc_tokenizer
    model_config['mask_tokenizer'] = mask_tokenizer
    model_config['mask_model'] = mask_model
    return model_config
