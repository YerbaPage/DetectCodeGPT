
import torch
import time
import math
import os

# get the average rank of each observed token sorted by model likelihood


def get_rank(text, args, model_config, log=False):

    with torch.no_grad():
        tokenized = model_config['base_tokenizer'](text, return_tensors="pt").to(args.DEVICE)
        # drop the "token_type_ids" key if it exists
        if 'token_type_ids' in tokenized.keys():
            tokenized.pop('token_type_ids')
        logits = model_config['base_model'](**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


def get_ranks(texts, args, model_config, log=True):
    return [get_rank(text, args, model_config, log=log) for text in texts]
