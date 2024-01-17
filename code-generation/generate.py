import time
import openai
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from loguru import logger
import gzip
import json
import pdb
from tqdm import tqdm
import os
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def load_data(path='./benchmark/humaneval-x', language='python', max_num=10000):

    all_prompts = []
    all_solutions = []

    if 'humaneval' in path:
        path_to_data = f'{path}/{language}/data/humaneval_{language}.jsonl.gz'

        logger.info(f'Loading data from {path_to_data}')

        with gzip.open(path_to_data, 'rb') as f:

            for line in f:
                data = json.loads(line)

                # # dict_keys(['task_id', 'prompt', 'canonical_solution', 'test', 'text', 'declaration', 'example_test'])
                # for key, value in data.items():
                #     logger.info(f'\n{key}: {value}')
                #     # the 'test' contains test cases for evaluation
                #     # text contains the function docstring, in the prompt
                #     # declaration contains the function declaration part, also in the prompt
                #     # example_test contains the test cases for the example function, in the prompt

                all_prompts.append(data['prompt'])
                all_solutions.append(data['canonical_solution'])

    elif 'CodeSearchNet' in path:

        path_to_data = f'{path}/{language}/train.jsonl'

        logger.info(f'Loading data from {path_to_data}')

        failed = 0
        success = 0

        max_prompt_len = 128
        min_prompt_len = 5

        max_solution_len = 256
        min_solution_len = 5

        with open(path_to_data, 'r') as f:

            count = 0
            for line in tqdm(f):

                data = json.loads(line)

                # {'repo': 'smdabdoub/phylotoast', 'path': 'phylotoast/util.py', 'func_name': 'split_phylogeny', 'original_string': 'def split_phylogeny(p, level="s"):\n    """\n    Return either the full or truncated version of a QIIME-formatted taxonomy string.\n\n    :type p: str\n    :param p: A QIIME-formatted taxonomy string: k__Foo; p__Bar; ...\n\n    :type level: str\n    :param level: The different level of identification are kingdom (k), phylum (p),\n                  class (c),order (o), family (f), genus (g) and species (s). If level is\n                  not provided, the default level of identification is species.\n\n    :rtype: str\n    :return: A QIIME-formatted taxonomy string up to the classification given\n            by param level.\n    """\n    level = level+"__"\n    result = p.split(level)\n    return result[0]+level+result[1].split(";")[0]', 'language': 'python', 'code': 'def split_phylogeny(p, level="s"):\n    """\n    Return either the full or truncated version of a QIIME-formatted taxonomy string.\n\n    :type p: str\n    :param p: A QIIME-formatted taxonomy string: k__Foo; p__Bar; ...\n\n    :type level: str\n    :param level: The different level of identification are kingdom (k), phylum (p),\n                  class (c),order (o), family (f), genus (g) and species (s). If level is\n                  not provided, the default level of identification is species.\n\n    :rtype: str\n    :return: A QIIME-formatted taxonomy string up to the classification given\n            by param level.\n    """\n    level = level+"__"\n    result = p.split(level)\n    return result[0]+level+result[1].split(";")[0]', 'code_tokens': ['def', 'split_phylogeny', '(', 'p', ',', 'level', '=', '"s"', ')', ':', 'level', '=', 'level', '+', '"__"', 'result', '=', 'p', '.', 'split', '(', 'level', ')', 'return', 'result', '[', '0', ']', '+', 'level', '+', 'result', '[', '1', ']', '.', 'split', '(', '";"', ')', '[', '0', ']'], 'docstring': 'Return either the full or truncated version of a QIIME-formatted taxonomy string.\n\n    :type p: str\n    :param p: A QIIME-formatted taxonomy string: k__Foo; p__Bar; ...\n\n    :type level: str\n    :param level: The different level of identification are kingdom (k), phylum (p),\n                  class (c),order (o), family (f), genus (g) and species (s). If level is\n                  not provided, the default level of identification is species.\n\n    :rtype: str\n    :return: A QIIME-formatted taxonomy string up to the classification given\n            by param level.', 'docstring_tokens': ['Return', 'either', 'the', 'full', 'or', 'truncated', 'version', 'of', 'a', 'QIIME', '-', 'formatted', 'taxonomy', 'string', '.'], 'sha': '0b74ef171e6a84761710548501dfac71285a58a3', 'url': 'https://github.com/smdabdoub/phylotoast/blob/0b74ef171e6a84761710548501dfac71285a58a3/phylotoast/util.py#L159-L177', 'partition': 'train'}

                data['original_string'] = data['original_string'].replace("'''", '"""')
                try:
                    prompt = data['original_string'].split('"""')[0] + '"""' + data['original_string'].split('"""')[1] + '"""'
                    solution = data['original_string'].split('"""')[2]
                    success += 1
                except:
                    # logger.info(f'Error in prompt/solution extraction for {data["original_string"]}')
                    # pdb.set_trace()
                    failed += 1

                # logger.info(f'prompt: \n{prompt}')
                # logger.info(f'solution: \n{solution}')

                if len(prompt.split()) > max_prompt_len or len(prompt.split()) < min_prompt_len:
                    continue

                if len(solution.split()) > max_solution_len or len(solution.split()) < min_solution_len:
                    continue

                all_prompts.append(prompt)
                all_solutions.append(solution)
                # keep the original version because I've ran other experiments already
                # count += 1
                # if count >= max_num:
                #     break

        logger.info(f'Failed: {failed}, Success: {success}')

    elif "TheVault" in path:

        path_to_data = f'{path}/{language}/small_train.jsonl'

        logger.info(f'Loading data from {path_to_data}')

        failed = 0
        success = 0

        max_prompt_len = 128
        min_prompt_len = 5

        max_solution_len = 256
        min_solution_len = 5

        with open(path_to_data, 'r') as f:

            count = 0
            for line in tqdm(f):

                data = json.loads(line)

                data['original_string'] = data['original_string'].replace("'''", '"""')
                try:
                    prompt = data['original_string'].split('"""')[0] + '"""' + data['original_string'].split('"""')[1] + '"""'
                    solution = data['original_string'].split('"""')[2]
                    success += 1
                except:
                    # logger.info(f'Error in prompt/solution extraction for {data["original_string"]}')
                    # pdb.set_trace()
                    failed += 1

                # logger.info(f'prompt: \n{prompt}')
                # logger.info(f'solution: \n{solution}')

                if len(prompt.split()) > max_prompt_len or len(prompt.split()) < min_prompt_len:
                    continue

                if len(solution.split()) > max_solution_len or len(solution.split()) < min_solution_len:
                    continue

                # if there is almost no docstring, skip (this have been done in the "try" part above")
                # if len(data['original_string'].split('"""')[1]) < 3:
                #     continue

                all_prompts.append(prompt)
                all_solutions.append(solution)
                # keep the original version because I've ran other experiments already
                # count += 1
                # if count >= max_num:
                #     break

        logger.info(f'Failed: {failed}, Success: {success}')

    logger.info(f'Loaded {len(all_prompts)} prompts and {len(all_solutions)} solutions')

    # analyze the lengths
    prompt_lengths = [len(prompt.split()) for prompt in all_prompts]
    solution_lengths = [len(solution.split()) for solution in all_solutions]
    logger.info(f'Prompt lengths: min: {min(prompt_lengths)}, max: {max(prompt_lengths)}, mean: {np.mean(prompt_lengths)}, std: {np.std(prompt_lengths)}')
    logger.info(f'Solution lengths: min: {min(solution_lengths)}, max: {max(solution_lengths)}, mean: {np.mean(solution_lengths)}, std: {np.std(solution_lengths)}')

    # todo: remove this sampling because I added max_num in preprocessing part
    if len(all_prompts) > max_num:
        # sample a subset of the data
        # fix the random seed for reproducibility

        seed = 42
        np.random.seed(seed)
        indices = np.random.choice(len(all_prompts), max_num, replace=False)
        all_prompts = [all_prompts[i] for i in indices]
        all_solutions = [all_solutions[i] for i in indices]

        prompt_lengths = [len(prompt.split()) for prompt in all_prompts]
        solution_lengths = [len(solution.split()) for solution in all_solutions]

        logger.info(f'Sampled {len(all_prompts)} prompts and {len(all_solutions)} solutions')
        logger.info(f'Prompt lengths: min: {min(prompt_lengths)}, max: {max(prompt_lengths)}, mean: {np.mean(prompt_lengths)}, std: {np.std(prompt_lengths)}')
        logger.info(f'Solution lengths: min: {min(solution_lengths)}, max: {max(solution_lengths)}, mean: {np.mean(solution_lengths)}, std: {np.std(solution_lengths)}')

    return all_prompts, all_solutions


def truncate(completion):

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            re.escape('<|endoftext|>')
        ]
    ]

    # prints = list(re.finditer('^print', completion, re.MULTILINE))
    # if len(prints) > 1:
    #     completion = completion[:prints[1].start()]

    # defs = list(re.finditer('^def', completion, re.MULTILINE))
    # logger.info(f'defs: {defs}')
    # if len(defs) > 1:
    #     completion = completion[:defs[1].start()]

    # todo: decide whether to remove the def lines (found some examples that have multiple def lines)
    # remove the 'def' lines
    # defs = list(re.finditer('    def ', completion, re.MULTILINE))
    # logger.info(f'defs: {defs}')
    # if len(defs) >= 1:
    #     completion = completion[:defs[0].start()]

    # # remove the '@' lines
    # ats = list(re.finditer('^    @', completion, re.MULTILINE))
    # logger.info(f'ats: {ats}')
    # if len(ats) > 0:
    #     completion = completion[:ats[0].start()]

    # todo: decide whether to remove the comments
    # # remove all the lines that start with '    #'
    # # split the completion into lines
    # lines = completion.splitlines()
    # # remove the lines that start with '    #'
    # lines = [line for line in lines if not line.startswith('    #')]
    # # join the lines back together
    # completion =

    start_pos = 0

    # # remove the comments
    # comments = list(re.finditer('^    #', completion, re.MULTILINE))

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def generate_hf(model_name, prompts, solutions, batch_size=16, max_length_sample=128, max_length=128, do_sample=True, top_p=0.95, temperature=0.2):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if 'codegen-' in model_name.lower():
        tokenizer.pad_token_id = 50256
        tokenizer.padding_side = 'left'
    elif 'santa' in model_name.lower():
        tokenizer.pad_token_id = 49156  # https://huggingface.co/bigcode/santacoder/blob/main/special_tokens_map.json
        logger.info(f'pad_token: {tokenizer.pad_token}')
        tokenizer.padding_side = 'left'
    elif 'parrot' in model_name.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f'pad_token: {tokenizer.pad_token}')
        tokenizer.padding_side = 'left'
    elif "incoder" in model_name.lower():
        tokenizer.pad_token_id = 1
        tokenizer.padding_side = 'left'
    elif "phi-1" in model_name.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f'pad_token: {tokenizer.pad_token}')
        tokenizer.padding_side = 'left'

    if 't5p' in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                      torch_dtype=torch.float16,
                                                      trust_remote_code=True)
    # todo: not sure wizard for tp16
    elif "llama" in model_name.lower() or "wizard" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
    elif "codegen2" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision="main")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    all_outputs = []
    model = model.to(device)

    if 'starcoder' in model_name.lower() or "llama" in model_name.lower() or "wizard" in model_name.lower() or "codegen2" in model_name.lower():
        input_ids = [tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids for prompt in prompts]

        def_id = tokenizer('def', add_special_tokens=False).input_ids[0]
        try:
            def_with_space_id = tokenizer('def', add_prefix_space=True, add_special_tokens=False).input_ids[0]
        except:
            def_with_space_id = tokenizer(' def', add_special_tokens=False).input_ids[0]

        eos_id_list = [tokenizer.eos_token_id, def_id, def_with_space_id]
        logger.info(f'eos_id_list: {eos_id_list}')

        for input_ids in tqdm(input_ids, ncols=50):
            input_ids = input_ids.to(device)
            input_ids_len = input_ids.shape[1]
            logger.info(f'input_ids_len: {input_ids_len}')

            if max_length_sample >= 256:
                outputs = model.generate(input_ids, do_sample=do_sample, max_length=max_length_sample+input_ids_len, top_p=top_p, temperature=temperature, pad_token_id=tokenizer.pad_token_id, use_cache=True, eos_token_id=eos_id_list)
            else:
                outputs = model.generate(input_ids, do_sample=do_sample, max_length=max_length_sample+input_ids_len, top_p=top_p, temperature=temperature, pad_token_id=tokenizer.pad_token_id, use_cache=True)

            decoded_output = tokenizer.decode(outputs[0, input_ids_len:])
            # logger.info(f'decoded_output: {decoded_output}')
            all_outputs.append(decoded_output)
            outputs = all_outputs
    else:
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids
        input_ids_len = input_ids.shape[1]
        logger.info(f'input_ids_len: {input_ids_len}')

        # create a dataset from the samples
        dataset = torch.utils.data.TensorDataset(input_ids)

        if batch_size >= 4:
            num_workers = batch_size // 2
        else:
            num_workers = 1

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

        # todo: when to use bad_words_ids?
        # bad_words = ['#', '"""', "'''"]
        # bad_words_ids = tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids
        # logger.info(f'bad_words_ids: {bad_words_ids}')

        for input_ids in tqdm(dataloader, ncols=50):

            input_ids = input_ids[0].to(device)
            outputs = model.generate(input_ids, do_sample=do_sample, max_length=max_length_sample+input_ids_len, top_p=top_p, temperature=temperature, pad_token_id=tokenizer.pad_token_id, use_cache=True)
            
            all_outputs.append(outputs)

        samples = torch.cat(all_outputs, dim=0)
        outputs = tokenizer.batch_decode(samples[:, input_ids_len:, ...])

    # truncate the outputs (based on the original code of CodeGen)
    outputs = [truncate(output) for output in outputs]

    logger.info(f'Generated {len(outputs)} samples')

    logger.info("Showing first 3 samples")

    for i in range(3):
        logger.info(f'Example {i}:')
        logger.info(f'Prompt: \n{prompts[i]}')
        logger.info(f'Output: \n{outputs[i]}')
        logger.info(f'Solution: \n{solutions[i]}')

    # pdb.set_trace()
    return prompts, outputs, solutions


def generate_openai(model_name, prompts, solutions, max_tokens=128, batch_size=10):

    # generate the outputs using the openai api

    # get the api key from the environment variable
    # os.environ["OPENAI_API_KEY"] = "PUT YOUR API KEY HERE"
    api_key = os.getenv('OPENAI_API_KEY')

    # create the openai object
    openai.api_key = api_key

    # generate the outputs
    outputs = []
    logger.info(f'Generating outputs with openai api')


    RPM = 20
    for i in tqdm(range(0, len(prompts), batch_size), ncols=50):
        prompt = prompts[i:i+batch_size]
        solution = solutions[i:i+batch_size]

        # generate the output
        # https://platform.openai.com/docs/guides/rate-limits/overview
        # logger.info(f'prompt: \n{prompt}')
        # logger.info(f'solution: \n{solution}')

        logger.info('sending request to openai api...')

        # response = openai.Completion.create(
        #     engine=model_name,
        #     prompt=prompt,
        #     max_tokens=256,
        #     top_p=0.95)

        # try for infinite times until the request is successful
        while True:
            retry_count = 0
            try:
                response = openai.Completion.create(
                    engine=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    top_p=0.95)
                break
            # show the error message and retry after certain time
            except Exception as e:
                # show the error message
                logger.info(f'openai api request failed, error: {e}')
                retry_count += 1
                logger.info(f'openai api request failed, retrying... {retry_count}')
                time.sleep(60/RPM)

        logger.info('response received')
        logger.info(f'response: \n{response}')

        output = [choice['text'] for choice in response.choices]

        outputs.extend(output)

        time.sleep(60/RPM)

    outputs = [truncate(output) for output in outputs]
    logger.info(f'Generated {len(outputs)} samples')
    output_lengths = [len(output.split()) for output in outputs]

    logger.info(f'output lengths: min: {min(output_lengths)}, max: {max(output_lengths)}, mean: {np.mean(output_lengths)}, std: {np.std(output_lengths)}')

    return prompts, outputs, solutions


if __name__ == "__main__":

    # load the data
    # path = './benchmark/humaneval-x'
    # path = 'data/CodeSearchNet'
    # path = "data/TheVault"

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="./benchmark/humaneval-x")
    parser.add_argument('--max_num', type=int, default=100000)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--model_name', type=str, default='codeparrot/codeparrot')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    logger.info(f'args: {args}')

    path = args.path
    max_num = args.max_num
    temperature = args.temperature
    model_name = args.model_name
    batch_size = args.batch_size

    # max_num = 100000
    prompts, solutions = load_data(path=path, language='python', max_num=max_num)

    # batch_size = 32
    # model_size = '6B'  # '350M' '2B' # '6B' # '16B'
    # model_name = f"Salesforce/codegen-{model_size}-multi"
    # model_name = "bigcode/santacoder"
    # model_name = 'codeparrot/codeparrot'
    # temperature = 0.2

    # model_name = 'code-davinci-002'

    if 'davinci' in model_name:
        prompts, outputs, solutions = generate_openai(model_name, prompts, solutions, max_tokens=200, batch_size=10)

    else:
        prompts, outputs, solutions = generate_hf(model_name, prompts, solutions, max_length_sample=args.max_length,
                                                  max_length=128, do_sample=True, top_p=0.95, temperature=temperature, batch_size=batch_size)

        model_name = model_name.split('/')[-1]

    logger.info(f'Generated {len(outputs)} outputs')

    # write the outputs to a file and together with the prompts and solutions

    save_prefix = f'output/{path.split("/")[-1]}'

    if args.max_length >= 256:
        file_name = f'{save_prefix}/{model_name}-{max_num}-tp{temperature}-nostop/outputs.txt'
        if not os.path.exists(f'{save_prefix}/{model_name}-{max_num}-tp{temperature}-nostop'):
            os.makedirs(f'{save_prefix}/{model_name}-{max_num}-tp{temperature}-nostop')
    else:
        file_name = f'{save_prefix}/{model_name}-{max_num}-tp{temperature}/outputs.txt'
        if not os.path.exists(f'{save_prefix}/{model_name}-{max_num}-tp{temperature}'):
            os.makedirs(f'{save_prefix}/{model_name}-{max_num}-tp{temperature}')
    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, 'w+') as f:
        for i in range(len(outputs)):
            results = {'prompt': prompts[i], 'output': outputs[i], 'solution': solutions[i]}
            f.write(json.dumps(results))
            f.write('\n')

    # another version that is more clear with printing
    file_name = f'{save_prefix}/{model_name}-{max_num}-tp{temperature}/outputs_v2.txt'

    # delete the file if it exists
    if os.path.exists(file_name):
        os.remove(file_name)

    for i in range(len(outputs)):
        # print the output directly
        print("-"*20, file=open(file_name, 'a'))
        print(f'Prompt: \n{prompts[i]}', file=open(file_name, 'a'))
        print("-"*10, file=open(file_name, 'a'))
        print(f'Output: \n{outputs[i]}', file=open(file_name, 'a'))
        print("-"*10, file=open(file_name, 'a'))
        print(f'Solution: \n{solutions[i]}', file=open(file_name, 'a'))

    logger.info(f'Finished writing to {file_name}')
