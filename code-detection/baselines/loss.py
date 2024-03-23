import torch
import time
import os

# Get the log likelihood of each text under the base_model
def get_ll(text, args, model_config):
    device_num = torch.cuda.device_count()-1 
            
    with torch.no_grad():
        if ('13b' in model_config['base_model'].config.name_or_path) or ('20b' in model_config['base_model'].config.name_or_path):
            tokenized = model_config['base_tokenizer'](text, return_tensors="pt").to(f'cuda:{device_num}')
        else:
            tokenized = model_config['base_tokenizer'](text, return_tensors="pt").to(args.DEVICE)

        # drop the "token_type_ids" key if it exists
        if 'token_type_ids' in tokenized.keys():
            tokenized.pop('token_type_ids')
        labels = tokenized['input_ids']
        
        time.sleep(0.01)

        return -model_config['base_model'](**tokenized, labels=labels).loss.item()

    
def get_lls(texts, args, model_config):
    return [get_ll(text, args, model_config) for text in texts]
    