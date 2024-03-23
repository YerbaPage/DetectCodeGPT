
import torch
import torch.nn.functional as F


# get average entropy of each token in the text
def get_entropy(text, args, model_config):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = model_config['base_tokenizer'](text, return_tensors="pt").to(args.DEVICE)  # input_ids + mask
        logits = model_config['base_model'](**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()
