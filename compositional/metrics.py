"""Metrics to evaluate the quality of the explanations."""

import functools

import torch

from . import model_utils, mask_utils
from . import constants as C


@functools.lru_cache(maxsize=10)
def compute_hits(vector):
    """Compute the number of ones in the given vector.
    Args:
        vector (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        hits (int): The number of ones in the given vector.
    """
    return torch.count_nonzero(vector)


def iou(vector1, vector2):
    """Compute the intersection over union between two vectors.
    Args:
        vector1 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        vector2 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        iou (float): The intersection over union between the two vectors.
    """
    intersection = torch.count_nonzero(vector1 & vector2)
    v1_size = compute_hits(vector1)
    v2_size = compute_hits(vector2)
    score = intersection / max(v1_size + v2_size - intersection, C.EPSILON)
    return score

def counter_iou(*, label_mask, bitmaps, counter_bitmaps):
    # Compute the IoU wrt the activation range considered
    formula_iou = iou(label_mask, bitmaps)

    # Compute the IoU wrt the other activations (~bitmaps)
    counter_iou = iou(label_mask, counter_bitmaps)
    
    # There must be a greater alignment in the activations considered than in all the other activations
    diff = max(formula_iou - counter_iou, 0)
    max_iou = max(formula_iou, counter_iou)
    diff = diff / (max_iou + 1e-8) # Normalize diff to avoid scale weighting
    
    # Compute the weighted Iou
    iou_score = formula_iou * diff
    #print(f"Formula IoU: {formula_iou}, Counter IoU: {counter_iou}, Diff: {diff}, IoU used for beam: {iou_score}")
    return iou_score

def diff_ratio_iou_adv_iou(iou, adv_iou):
    if iou == 0:
        return 0
    if adv_iou == 0:
        return 1
    diff = max(iou - adv_iou, 0)
    # normalize diff in a way that if vanilla_iou and adv_iou are small and their difference is small is comparable to a bigger difference but with vanilla_iou and adv_iou larger. Basically, we want to express the difference in percentage and independently from the size of iou
    diff = diff / iou
    return diff

def weighted_iou(segmentations, bitmaps, weights):
    """Compute the intersection over union between two vectors.
    Args:
        segmentations (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        iou (float): The intersection over union between the two vectors.
    """
    intersection_map = segmentations & bitmaps
    weighted_intersection = (intersection_map * weights).sum()
    #weighted_segmentations = (segmentations * weights).sum()
    weighted_bitmaps = (bitmaps * weights).sum()

    score = weighted_intersection / max(segmentations.sum() + weighted_bitmaps - weighted_intersection, C.EPSILON)
    return score


def sample_iou(vector1, vector2):
    """Compute the intersection over union between two vectors.
    Args:
        vector1 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        vector2 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        iou (float): The intersection over union between the two vectors.
    """
    intersection = torch.count_nonzero(vector1 & vector2, 1)
    v1_size = torch.count_nonzero(vector1, 1)
    v2_size = torch.count_nonzero(vector2, 1)
    score = intersection / (v1_size + v2_size - intersection + C.EPSILON)
    return score


def activations_coverage(activations, segmentations):
    """Compute the activation coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
    Returns:
        activation_coverage (float): The activation coverage.
    """
    return torch.count_nonzero(
        activations & segmentations
    ) / torch.count_nonzero(activations)

def explanation_coverage(activations, segmentations):
    return get_num_nonzerosamples(
        activations & segmentations) / (
            activations.sum(1) > 0).sum()
    

def detection_accuracy(activations, segmentations):
    """Compute the segmentations coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
    Returns:
        segmentations_coverage (float): The segmentations coverage.
    """
    return torch.count_nonzero(
        activations & segmentations
    ) / torch.count_nonzero(segmentations)


def samples_coverage(activations, segmentations):
    """Compute the samples coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H, W) where N is
        the number of sample.
    Returns:
        samples_coverage (float): The samples coverage.
    """
    samples_overlap = (
        torch.sum(activations & segmentations, 1, dtype=torch.int32) > 0
    )
    segmentation_in = torch.sum(segmentations, 1, dtype=torch.int32) > 0
    return torch.sum(samples_overlap) / torch.sum(segmentation_in)


def avg_mask_size(mask):
    """Compute the average mask size for the given mask considering
    only the samples where the mask has at least one pixel.
    Args:
        mask (torch.Tensor): A tensor of shape (N, F) where N is
            the number of sample.
    Returns:
        avg_size (float): The average segmentation size.
    """
    assert len(mask.shape) == 2
    samples_size = torch.sum(mask, 1, dtype=torch.int32)
    samples_size = samples_size[samples_size > 0]
    samples_size = samples_size / mask.shape[1]
    return torch.mean(samples_size.float())


def avg_overlapping(activations, segmentations):
    """Compute the average overlapping between the given activations and
    segmentations by considering only the samples where the intersection is
    greater than zero.
    Args:
        activations (torch.Tensor): A tensor of shape (N, F) where N is the
            number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, F) where N is the
            number of sample.
    Returns:
        avg_overlapping (float): The average overlapping.
    """
    overlapping = torch.sum(activations & segmentations, 1, dtype=torch.int32)
    assert len(activations.shape) == 2

    overlapping = overlapping[overlapping > 0]
    overlapping = overlapping / activations.shape[1]
    return torch.mean(overlapping)


def get_num_nonzerosamples(mask):
    """Compute the number of samples with at least one pixel.
    Args:
        mask (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        num_nonzerosamples (int): The number of samples with at
            least one pixel.
    """
    return torch.sum(torch.sum(mask, 1, dtype=torch.int32) > 0)


def cosine_concept_masking_score(
        activation_before_masking, activation_after_masking, activation_range):
    """Compute the concept masking for the given activations.
    Args:
        activation_before_masking (torch.Tensor): A tensor of shape (N, H, W)
            where N is the number of sample.
        activation_after_masking (torch.Tensor): A tensor of shape (N, H, W)
            where N is the number of sample.
    Returns:
        concept_masking (float): The concept masking.
    """

    activation_before_masking = activation_before_masking.flatten(1)
    bitmaps = torch.where(
                    (activation_before_masking > activation_range[0])
                    & (activation_before_masking < activation_range[1]),
                    1, 0)
    activation_after_masking = activation_after_masking.flatten(1)
    activation_before_masking = activation_before_masking * bitmaps
    activation_after_masking = activation_after_masking * bitmaps
    return torch.mean(torch.nn.functional.cosine_similarity(
        activation_before_masking, activation_after_masking
    ))


def get_concept_masking(*,
    activations,
    mask_shape,
    label_mask,
    loader,
    model,
    layer_name,
    unit,
    input_size,
    activation_range,
):
    """Compute the concept masking for the given label.
    Args:
        label (torch.Tensor):
    Returns:
        concept_masking (float):
    """
    samples_with_label = torch.nonzero(label_mask.sum(1) > 0).flatten()
    label_mask = label_mask.reshape(-1, mask_shape[0], mask_shape[1])
    no_mask_activations = activations[samples_with_label].unsqueeze(1)
    random_masks = torch.rand(label_mask.shape, device=label_mask.device)
    mask_to_apply = torch.where(label_mask == 0, random_masks, label_mask)
    masked_activations = model_utils.apply_concept_masking(
        loader,
        model,
        [layer_name],
        units=[unit],
        mask=mask_to_apply,
        image_size=input_size,
    )[0][samples_with_label]
    score = cosine_concept_masking_score(
        no_mask_activations, masked_activations, activation_range)
    return score


def diff_adv_iou(formula_mask, bitmaps):
    std_iou = iou(formula_mask, bitmaps)
    adv_iou = iou(formula_mask, ~bitmaps)
    diff = max(std_iou - adv_iou, 0)
    # normalize diff in a way that if vanilla_iou and adv_iou are small and their difference is small is comparable to a bigger difference but with vanilla_iou and adv_iou larger. Basically, we want to express the difference in percentage and independently from the size of iou
    diff = diff / (std_iou + 1e-5)
    return diff


def compute_metrics(metrics, label, masks, bitmaps):
    dict_results = {}
    label_mask = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
        label, masks)).to(bitmaps.device)
    for metric in metrics:
        if metric == "iou":
            metric_function = iou
        elif metric == "activations_coverage":
            metric_function = activations_coverage
        elif metric == "detection_accuracy":
            metric_function = detection_accuracy
        elif metric == "samples_coverage":
            metric_function = samples_coverage
        elif metric == "explanation_coverage":
            metric_function = explanation_coverage
        elif metric == "diff_adv_iou":
            metric_function = diff_adv_iou
        else:
            raise ValueError(f"Unknown metric {metric}")
        metric_value = metric_function(label_mask, bitmaps).item()
        dict_results[metric] = metric_value
    return dict_results


def compute_metric_summary(metric, results):
    list_metric = []
    for result in results:
        unit, cluster_index, activation_range, best_label, string_label, metrics_values = result
        if metric in metrics_values:
            metric_value = metrics_values[metric]
            list_metric.append(float(metric_value))
    summary = {
        f"avg_{metric}": sum(list_metric) / len(list_metric) if list_metric else -1,
        f"stdev_{metric}": torch.std(torch.tensor(list_metric)).item() if list_metric else -1,
    }  
    return summary


def compute_avg_metric(metric, results):
    metric_counter = 0
    for result in results:
        unit, cluster_index, activation_range, best_label, string_label, metrics_values = result
        if metric in metrics_values:
            metric_value = metrics_values[metric]
            metric_counter += float(metric_value)
    avg_metric = metric_counter / len(results) if results else -1
    summary = {
        f"percentage_{metric}": avg_metric,
        }  
    return summary

def compute_metrics_summary(results):
    summary_iou = compute_metric_summary("iou", results)
    summary_act_coverage = compute_metric_summary("activations_coverage", results)
    summary_det_accuracy = compute_metric_summary("detection_accuracy", results)
    summary_sample_coverage = compute_metric_summary("samples_coverage", results)
    summary_explanation_coverage = compute_metric_summary("explanation_coverage", results)
    summary_apply = compute_metric_summary("apply", results)
    summary_diff_iou = compute_metric_summary("diff_adv_iou", results)
    summary_visited = compute_metric_summary("visited", results)
    summary_expanded = compute_metric_summary("expanded", results)
    summary_estimated = compute_metric_summary("estimated", results)
    summary_time_taken = compute_metric_summary("time_taken", results)
    summary_is_not_incremental = compute_avg_metric("is_not_incremental", results)
    summary_descriptivity = compute_metric_summary("descriptivity", results)
    summary_firing_rate = compute_metric_summary("firing_rate", results)
    summary_p_artifact = compute_metric_summary("p_artifact", results)
    summary_reliable = compute_avg_metric("realiable", results)
    summary = {**summary_iou, **summary_act_coverage, **summary_det_accuracy, **summary_sample_coverage, **summary_explanation_coverage, **summary_apply, **summary_diff_iou, **summary_visited, **summary_expanded, **summary_estimated, **summary_time_taken, **summary_is_not_incremental, **summary_descriptivity, **summary_firing_rate, **summary_p_artifact, **summary_reliable}
    return summary