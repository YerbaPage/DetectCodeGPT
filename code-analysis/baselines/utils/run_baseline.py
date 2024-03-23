import torch
import numpy as np
import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import math
from loguru import logger


def run_baseline_threshold_experiment(criterion_fn, name, data, args):
    torch.manual_seed(0)
    np.random.seed(0)
    predictions = {'real': [], 'samples': []}
    for batch in tqdm.tqdm(range(len(data["original"]) // args.batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * args.batch_size:(batch + 1) * args.batch_size]
        sampled_text = data["sampled"][batch * args.batch_size:(batch + 1) * args.batch_size]

        for idx in range(len(original_text)):
            predictions['real'].append(criterion_fn(original_text[idx]))
            predictions['samples'].append(criterion_fn(sampled_text[idx]))
    _, _, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    _, _, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    acc = get_accurancy(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}, Acc: {acc}")

    return {'name': f'{name}_threshold',
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'acc': acc,
            }


def get_roc_metrics(real_preds, sample_preds):

    # filter out any NaNs
    new_real_preds = []
    new_sample_preds = []
    for i in range(len(real_preds)):
        if not np.isnan(real_preds[i]) and not np.isnan(sample_preds[i]):
            new_real_preds.append(real_preds[i])
            new_sample_preds.append(sample_preds[i])

    logger.info(f"Filtered out {len(real_preds) - len(new_real_preds)} NaNs")

    # fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    fpr, tpr, _ = roc_curve([0] * len(new_real_preds) + [1] * len(new_sample_preds), new_real_preds + new_sample_preds)
    roc_auc = auc(fpr, tpr)

    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):

    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def get_accurancy(real_preds, sample_preds):

    return sum(np.array(real_preds) < np.array(sample_preds))/len(sample_preds)


# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
          "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
          "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# save the ROC curve for each experiment, given a list of output dictionaries, one for each experiment, using colorblind-friendly colors


def save_roc_curves(experiments):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")

# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed


def save_ll_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
        except:
            pass
# save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed


def save_llr_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]

            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        except:
            pass
