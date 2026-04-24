import torch

from compositional import metrics

def compute_scores(mask_a, mask_b):
    iou = metrics.iou(mask_a, mask_b)
    activation_coverage = metrics.activations_coverage(
        mask_a, mask_b
    )
    detection_accuracy = metrics.detection_accuracy(
        mask_a, mask_b
    )
    samples_coverage = metrics.samples_coverage(
        mask_a, mask_b
    )

    explanation_coverage = metrics.explanation_coverage(
        mask_a, mask_b)

    dict_results = {
        "iou": iou.item(),
        "activation_coverage": activation_coverage.item(),
        "label_coverage": detection_accuracy.item(),
        "samples_coverage": samples_coverage.item(),
        "explanation_coverage": explanation_coverage.item(),
    }
    return dict_results

def is_specialization(samples_formula_a, samples_formula_b):
    if is_equivalent(samples_formula_a, samples_formula_b):
        return False
    TP = torch.sum(samples_formula_a & samples_formula_b)
    FP = torch.sum(samples_formula_a & ~samples_formula_b)
    return TP > 0 and FP == 0

def rate_specialization(samples_formula_a, samples_formula_b):
    TP = torch.sum(samples_formula_a & samples_formula_b)
    FP = torch.sum(samples_formula_a & ~samples_formula_b)
    return 1 - (FP / (TP + FP))

def is_generalization(samples_formula_a, samples_formula_b):
    if is_equivalent(samples_formula_a, samples_formula_b):
        return False
    TP = torch.sum(samples_formula_a & samples_formula_b)
    FN = torch.sum(~samples_formula_a & samples_formula_b)
    return TP > 0 and FN == 0

def rate_generalization(samples_formula_a, samples_formula_b):
    TP = torch.sum(samples_formula_a & samples_formula_b)
    FN = torch.sum(~samples_formula_a & samples_formula_b)
    return 1 - (FN / (TP + FN))

def is_equivalent(samples_formula_a, samples_formula_b):
    return torch.all(samples_formula_a == samples_formula_b)