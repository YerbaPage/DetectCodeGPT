from transformers import RobertaTokenizer
import argparse
from loguru import logger
from tree_sitter import Language, Parser
import os
from tqdm import tqdm
import json
import numpy as np
import pdb


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


def get_identifier(code, lang):
    pos = []
    identifiers = []

    def traverse(root):
        if root is None:
            return
        for child in root.children:
            if child.type == 'identifier':
                if child.start_byte > 0 and code[child.start_byte - 1] == '.':
                    continue
                # the token 'self' is not an identifier to perturb
                if get_identifier_from_position(code, child.start_point, child.end_point) == 'self':
                    continue
                
                start_point = child.start_point
                end_point = child.end_point
                pos.insert(0, (start_point, end_point))
            traverse(child)


    parser.set_language(LANGUAGE_MAP[lang])
    tree = parser.parse(bytes(code, 'utf-8'))
    traverse(tree.root_node)
    for id in pos:
        identifiers.append(get_identifier_from_position(code, id[0], id[1]))
    # return identifiers, and the position of all identifiers
    return list(set(identifiers)), pos


def get_identifier_from_position(code_string, start_point, end_point):

    lines = code_string.splitlines()

    identifier = lines[start_point[0]][start_point[1]:end_point[1]]
    # logger.info(f'identifier: {identifier}\n\n')
    return identifier


def load_data(path, language='python', max_num=10000):

    all_data = []
    path_to_data = f'{path}/{language}/train.jsonl'

    logger.info(f'Loading data from {path_to_data}')

    failed = 0
    success = 0

    max_len = 128
    min_len = 5

    with open(path_to_data, 'r') as f:
        count = 0
        for line in tqdm(f):

            if count >= max_num:
                break

            data = json.loads(line)
            data['original_string'] = data['original_string'].replace("'''", '"""')
            try:
                prompt = data['original_string'].split('"""')[0] + '"""' + data['original_string'].split('"""')[1] + '"""'
                solution = data['original_string'].split('"""')[2]
                success += 1
            except:
                failed += 1

            all_data.append(solution)
            count += 1

    logger.info(f'Failed: {failed}, Success: {success}')

    all_lengths = [len(data.split()) for data in all_data]
    logger.info(f'All lengths: min: {min(all_lengths)}, max: {max(all_lengths)}, mean: {np.mean(all_lengths)}, std: {np.std(all_lengths)}')

    return all_data

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--language_name', type=str, default='java')
    params = argparser.parse_args()

    # code_item = "def test():\n    print('hello world')"
    texts = ['''
def remove_mask_space(text, args, **kwargs):
    # find all the mask positions " <extra_id_\d+> ", and remove the space before and after the mask
    pattern = re.compile(r" <extra_id_\d+> ")
    matches = pattern.findall(text)
    for match in matches:
        text = text.replace(match, match.strip())
    return text
''']
    texts = ['''
tags to be added to the image.
        :type extra_tags: list of unicode | str
        :param add_latest: If True, the latest tag will be added to the list of tags.
        :type add_latest: bool
        :return: The image id.
        :rtype: unicode | str
        """
        if not isinstance(main_tag, six.string_types):
            raise TypeError('main_tag must be a string')
        if not isinstance(extra_tags, list):
''']
    code_item = texts[0]

    # max_num = 100
    # original_codes = load_data(path=path, language='python', max_num=max_num)
    # code_item = original_codes[0]

    varnames, pos = get_identifier(code_item, 'python')

    logger.info(f'code_item: \n{code_item}')
    logger.info(f'varnames: \n{varnames}')
    logger.info(f'pos: \n{pos}')
