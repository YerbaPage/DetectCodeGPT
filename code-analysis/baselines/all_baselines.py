from .loss import get_ll, get_lls
from .detectGPT import perturb_texts
from .rank import get_ranks, get_rank
from .entropy import get_entropy
from .supervised import eval_supervised
import torch
import numpy as np
import tqdm
import functools
from .utils.run_baseline import get_roc_metrics, get_precision_recall_metrics, get_accurancy, run_baseline_threshold_experiment
import math


def run_all_baselines(data, args, n_perturbation_list, model_config, baselines=['likelihood', 'logrank', 'simple', 'perturb_prob', 'perturb_logrank']):
    torch.manual_seed(0)
    np.random.seed(0)
    original_text = data["original"]
    sampled_text = data["sampled"]
    results = []
    if ('DetectGPT' in baselines) or ('NPR' in baselines):
        perturb_fn = functools.partial(perturb_texts, args=args, model_config=model_config)
        p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(max(n_perturbation_list))])
        p_original_text = perturb_fn([x for x in original_text for _ in range(max(n_perturbation_list))])
        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "sampled": sampled_text[idx],
                "perturbed_sampled": p_sampled_text[idx * max(n_perturbation_list): (idx + 1) * max(n_perturbation_list)],
                "perturbed_original": p_original_text[idx * max(n_perturbation_list): (idx + 1) * max(n_perturbation_list)]
            })
    else:
        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "sampled": sampled_text[idx]
            })
    if ('likelihood' in baselines) or ('DetectGPT' in baselines) or ('LRR' in baselines):
        for res in tqdm.tqdm(results, desc="Computing unperturbed log likelihoods"):
            res["original_ll"] = get_ll(res["original"], args, model_config)
            res["sampled_ll"] = get_ll(res["sampled"], args, model_config)
    if ('logrank' in baselines) or ('NPR' in baselines) or ('simple' in baselines):
        for res in tqdm.tqdm(results, desc="Computing unperturbed log rank"):
            res["original_logrank"] = get_rank(res["original"], args, model_config, log=True)
            res["sampled_logrank"] = get_rank(res["sampled"], args, model_config, log=True)
    if 'DetectGPT' in baselines:
        for res in tqdm.tqdm(results, desc="Computing perturbed log likelihoods"):
            p_sampled_ll = get_lls(res["perturbed_sampled"], args, model_config)
            p_original_ll = get_lls(res["perturbed_original"], args, model_config)
            for n_perturbation in n_perturbation_list:

                res[f"perturbed_sampled_ll_{n_perturbation}"] = np.mean([i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)])
                res[f"perturbed_original_ll_{n_perturbation}"] = np.mean([i for i in p_original_ll[:n_perturbation] if not math.isnan(i)])
                res[f"perturbed_sampled_ll_std_{n_perturbation}"] = np.std([i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)]) if len([
                    i for i in p_sampled_ll[:n_perturbation] if not math.isnan(i)]) > 1 else 1
                res[f"perturbed_original_ll_std_{n_perturbation}"] = np.std([i for i in p_original_ll[:n_perturbation] if not math.isnan(i)]) if len([
                    i for i in p_original_ll[:n_perturbation] if not math.isnan(i)]) > 1 else 1
    if 'NPR' in baselines:
        for res in tqdm.tqdm(results, desc="Computing perturbed log rank"):
            p_sampled_rank = get_ranks(res["perturbed_sampled"], args, model_config, log=True)
            p_original_rank = get_ranks(res["perturbed_original"], args, model_config, log=True)
            for n_perturbation in n_perturbation_list:
                res[f"perturbed_sampled_logrank_{n_perturbation}"] = np.mean([i for i in p_sampled_rank[:n_perturbation] if not math.isnan(i)])
                res[f"perturbed_original_logrank_{n_perturbation}"] = np.mean([i for i in p_original_rank[:n_perturbation] if not math.isnan(i)])

    baseline_outputs = []
    for baseline in baselines:
        if baseline == 'likelihood':
            predictions = {'real': [], 'samples': []}
            for res in results:
                predictions['real'].append((res['original_ll']))
                predictions['samples'].append((res['sampled_ll']))
            _, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
            baseline_outputs.append({'name': f'{baseline}_threshold', 'roc_auc': roc_auc, })

        if baseline == 'rank':
            def rank_criterion(text): return -get_rank(text, args, model_config, log=False)
            baseline_outputs.append(run_baseline_threshold_experiment(rank_criterion, "rank", data, args))

        if baseline == 'logrank':
            predictions = {'real': [], 'samples': []}
            for res in results:
                predictions['real'].append(-res['original_logrank'])
                predictions['samples'].append(-res['sampled_logrank'])
            _, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
            baseline_outputs.append({'name': f'{baseline}_threshold', 'roc_auc': roc_auc, })

        if baseline == 'entropy':
            def entropy_criterion(text): return get_entropy(text, args, model_config)
            baseline_outputs.append(run_baseline_threshold_experiment(entropy_criterion, "entropy", data, args))

        if baseline == 'LRR':
            predictions = {'real': [], 'samples': []}
            for res in results:
                predictions['real'].append(-res['original_ll']/res['original_logrank'])
                predictions['samples'].append(-res['sampled_ll']/res['sampled_logrank'])
            _, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
            baseline_outputs.append({'name': f'{baseline}_threshold', 'roc_auc': roc_auc, })

        if baseline == 'supervised_roberta_base':
            baseline_outputs.append(eval_supervised(data, args, model_config, model='roberta-base-openai-detector'))

        if baseline == 'supervised_roberta_large':
            baseline_outputs.append(eval_supervised(data, args, model_config, model='roberta-large-openai-detector'))

        if baseline == 'DetectGPT':
            for n_perturbation in n_perturbation_list:
                predictions = {'real': [], 'samples': []}
                for res in results:

                    predictions['real'].append((res['original_ll'] - res[f'perturbed_original_ll_{n_perturbation}']) / res[f'perturbed_original_ll_std_{n_perturbation}'])
                    predictions['samples'].append((res['sampled_ll'] - res[f'perturbed_sampled_ll_{n_perturbation}']) / res[f'perturbed_sampled_ll_std_{n_perturbation}'])
                _, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])

                name = f'perturbation_{n_perturbation}_{baseline}'
                baseline_outputs.append({'name': name, 'roc_auc': roc_auc, })

        if baseline == 'NPR':
            for n_perturbation in n_perturbation_list:
                predictions = {'real': [], 'samples': []}
                for res in results:
                    predictions['real'].append(res[f'perturbed_original_logrank_{n_perturbation}']/res["original_logrank"])
                    predictions['samples'].append(res[f'perturbed_sampled_logrank_{n_perturbation}']/res["sampled_logrank"])
                _, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
                name = f'perturbation_{n_perturbation}_{baseline}'
                baseline_outputs.append({'name': name, 'roc_auc': roc_auc, })

    return baseline_outputs
