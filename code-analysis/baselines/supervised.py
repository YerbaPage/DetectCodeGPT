
import transformers
import torch
import tqdm
from .utils.run_baseline import get_roc_metrics


def eval_supervised(data, args,  model_config, model):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=model_config['cache_dir']).to(args.DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=model_config['cache_dir'])

    real, fake = data['original'], data['sampled']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // args.batch_size), desc="Evaluating real"):
            batch_real = real[batch * args.batch_size:(batch + 1) * args.batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:, 0].tolist())

        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // args.batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * args.batch_size:(batch + 1) * args.batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:, 0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }
    _, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])

    print(f"{model}_threshold ROC AUC: {roc_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {'name': model,
            'roc_auc': roc_auc,
            }
