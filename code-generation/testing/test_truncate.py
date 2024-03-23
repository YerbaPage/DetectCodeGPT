import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import gzip
import json
import pdb
from tqdm import tqdm
import os
import re
import numpy as np
import openai
import time


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

    # remove the 'def' lines
    defs = list(re.finditer('    def ', completion, re.MULTILINE))
    logger.info(f'defs: {defs}')
    if len(defs) >= 1:
        completion = completion[:defs[0].start()]

    # remove the '@' lines
    ats = list(re.finditer('^    @', completion, re.MULTILINE))
    logger.info(f'ats: {ats}')
    if len(ats) > 0:
        completion = completion[:ats[0].start()]


    start_pos = 0

    # # remove the comments
    # comments = list(re.finditer('^    #', completion, re.MULTILINE))

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


output = {"prompt": "def getModName(self):\n        \"\"\"\n        Return the lowercased name of this module.\n\n        Notes:\n            This pulls the ``mod_name`` attribute on the class. This allows\n            an implementer to set a arbitrary name for the module.  If this\n            attribute is not set, it defaults to\n            ``self.__class__.__name__.lower()`` and sets ``mod_name`` to\n            that value.\n\n        Returns:\n            (str): The module name.\n        \"\"\"",
          "output": "\n\n        if not hasattr(self, 'mod_name'):\n            mod_name = self.__class__.__name__.lower()\n            setattr(self, 'mod_name', mod_name)\n            return mod_name\n        return self.mod_name\n\n    @getModName.setter\n    def getModName(self, name):\n        \"\"\"\n        Set the module's name.\n\n        Args:\n            name (str): The name to set the module's name to.\n\n        \"\"\"\n        setattr(self, 'mod_name', name)\n\n    def getReadables(self):\n        \"\"\"\n        Return a list of the readable values of this module.\n\n        Notes:\n            This pulls the ``readables`` list on the class. This allows\n            an implementer to define a set of readable values. If this\n            attribute is not set, it defaults to an empty list, then sets\n            ``readables`` to that list.\n\n        Returns:\n            (list): A list of readable values.\n\n        \"\"\"\n        if not hasattr(self, 'readables'):\n            readables = []", "solution": "\n        ret = self.mod_name\n        if ret is None:\n            ret = self.__class__.__name__\n        return ret.lower()"}['output']

logger.info(f'output: \n{output}')

output = truncate(output)

logger.info(f'truncated output: \n{output}')
