"""
Utility functions.
"""
import os 
import pickle
from tqdm import tqdm
import random
from timeit import default_timer as timer

import numpy as np
import torch
from numpy.random import RandomState
import scipy.sparse as sparse

from compositional import activation_utils, mask_utils, metrics
from compositional import algorithms
from src.model_wrapper import Place365Model, DenseNetPlace365, ImageNetModel, NLPModelWrapper, UntrainedModel
from compositional import formula as F
from compositional import constants as C


def is_incremental(label):
    if isinstance(label, int):
        return True
    elif isinstance(label, F.Leaf):
        return True
    elif isinstance(label, F.UnaryNode):
        return is_incremental(label.val)
    elif isinstance(label, F.BinaryNode):
        return is_incremental(label.left) and (isinstance(label.right, F.Leaf) or isinstance(label.right, F.UnaryNode))
    else:
        raise ValueError(f"Label type {type(label)} not supported")


def set_seed(seed: int) -> RandomState:
    """Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # set to false for reproducibility, True to boost performance
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    return g


# reference: https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    """Method to set seed for each worker.
    Args:
        worker_id (int): Id of the worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_to_sparse(vector):
    """
    Convert a torch tensor to a sparse matrix.

    Args:
        vector (torch.Tensor): tensor to convert

    Returns:
        scipy.sparse.csr_matrix: sparse matrix
    """
    if len(vector.shape) != 2:
        vector = torch.reshape(vector, (vector.shape[0], -1))
    return sparse.csr_matrix(vector.numpy())


def sparse_to_torch(vector):
    """
    Convert a sparse matrix to a torch tensor.

    Args:
        vector (scipy.sparse.csr_matrix): sparse matrix to convert

    Returns:
        torch.Tensor: tensor
    """
    return torch.from_numpy(vector.toarray())


    



def get_result_dir(root, heuristic_name, configuration, dataset_name, segmentor_name, mask_shape, used_classes, probed_model, granularity=None, additional_dataset=None):
    if 'wordnet_' in configuration:
        results_dir = f"{root}/{used_classes}/{dataset_name}/{probed_model}/{segmentor_name}/{configuration}/{mask_shape}"
    elif 'categorical' in configuration and segmentor_name not in ['human', 'mask2former']:
        results_dir = f"{root}/{configuration}/granularity_{granularity}_{additional_dataset}/{dataset_name}/{probed_model}/{segmentor_name}/{mask_shape}"
    else:
        if heuristic_name == 'optimal':
            root = f"{root}/optimal"
        elif heuristic_name == 'beam_optimal':
            root = f"{root}/beam_optimal"
        elif heuristic_name == 'none':
            root = f"{root}/no_heuristic"
        results_dir = f"{root}/{used_classes}/{dataset_name}/{probed_model}/{segmentor_name}/{mask_shape}"
    return results_dir

def get_optimal_info_dir(root, heuristic_name, configuration, dataset_name, segmentor_name, mask_shape, used_classes, probed_model, granularity=None, additional_dataset=None):
    optimal_dir = f"{root}/{used_classes}/{dataset_name}/{probed_model}/{segmentor_name}/{mask_shape}"
    return optimal_dir



# def compute_compositional_explanations(model,masks, masks_info, disjoint_info, activations, units, config_experiment, config_compositional, *, interpolation_mode='bilinear', beam_variant=None, verbose=True, constraints=None, dataset=None):
#     exit()
#     num_clusters = config_compositional['num_clusters']
#     mask_shape = config_experiment['mask_shape']
#     length = config_compositional['length']
#     layer_name = config_compositional['layer_name']
#     device = config_experiment['device']
#     results_dir = config_experiment['results_dir']
#     beam_limit = config_compositional['beam_limit']
#     heuristic = config_compositional['heuristic']
#     results = []
#     counter_variant = config_compositional['counter_variant']
#     exit()

#     for unit in tqdm(
#                 units, desc="Computing Compostional explanations per unit"
#             ):
#         if len(activations.shape) == 4:
#             unit_activations = activations[:,unit,:,:]
#         elif len(activations.shape) == 2:
#             unit_activations = activations[:,unit]
#         else:
#             raise ValueError(f"Activations shape {activations.shape} not supported")

#         # Filter out extreme cases where there are too few activations
#         nonzero_activations = torch.count_nonzero(unit_activations)
#         if nonzero_activations < num_clusters:
#             print(f"Unit {unit} cluster {cluster_index} has very few activations ({nonzero_activations}), skipping. Please consider changing the number of clusters or checking the activation ranges.")
#             continue
#         exit()
#         # Compute activation range to be kept in the masks
#         activation_ranges = activation_utils.compute_activation_ranges(
#             unit_activations, num_clusters)
#         for cluster_index, activation_range in enumerate(
#                     sorted(activation_ranges)
#                 ):
#             beam_dir = 'optimal' if heuristic == 'optimal' else beam_limit
#             dir_current_results = (
#                 f"{results_dir}/{beam_dir}/length_{length}/"
#                 + f"{layer_name}/{unit}/{activation_range}"
#             )
#             if beam_variant == 'old':
#                 dir_current_results = dir_current_results + '/old_beam'
#             elif beam_variant == 'new':
#                 dir_current_results = dir_current_results + '/new_beam'
#             elif beam_variant is None:
#                 dir_current_results = dir_current_results + '/baseline'
#             if counter_variant:
#                 dir_current_results = dir_current_results + '/counter_variant'
#             if not os.path.exists(dir_current_results):
#                 os.makedirs(dir_current_results)
#             file_algo_results = (
#                 f"{dir_current_results}/" + f"{length}.pickle"
#             )



#             # Compute binary masks
#             if unit_activations.shape != masks[1].shape:
#                 bitmaps = activation_utils.compute_bitmaps(
#                     unit_activations,
#                     activation_range,
#                     mask_shape=mask_shape,
#                 )
#             else:
#                 #print("Activations already in the right shape, no need to upsample")
#                 bitmaps = torch.where(
#                     (unit_activations > activation_range[0]) & (unit_activations < activation_range[1]), 
#                     True, False)


#             bitmaps = bitmaps.to(device)
#             if True or not os.path.exists(file_algo_results):

#                 #bitmaps_perc = bitmaps.sum()/bitmaps.numel()
#                 #random_bitmaps = torch.bernoulli(torch.full(bitmaps.shape, bitmaps_perc)).bool().to(device)

#                 if dataset is not None and bitmaps.sum() < 500:
#                     #print(f"Too few activations for unit {unit} cluster {cluster_index} with range {activation_range}. Only {bitmaps.sum()} activations found. Please consider changing the number of clusters or checking the activation ranges.")
#                     continue
#                 start_time = timer()
#                 (
#                     best_label,
#                     best_iou,
#                     visited,
#                     expanded,
#                     estimated
#                 ) = algorithms.get_heuristic_scores(
#                     masks,
#                     bitmaps,
#                     segmentations_info=masks_info,
#                     disjoint_info=disjoint_info,
#                     heuristic=heuristic,
#                     length=length,
#                     beam_size=beam_limit,
#                     max_size_mask=mask_shape[0]*mask_shape[1],
#                     mask_shape=mask_shape,
#                     device=device,
#                     labels=config_experiment['segmentor_labels'],
#                     beam_variant=beam_variant,
#                     constraints=constraints,
#                     counter_variant=counter_variant
#                 )
#                 exit()
#                 end_time = timer()

#                 # # RANDOM
#                 # (
#                 #     random_label,
#                 #     random_iou,
#                 #     _,
#                 #     _,
#                 #     _
#                 # ) = algorithms.get_heuristic_scores(
#                 #     masks,
#                 #     random_bitmaps,
#                 #     segmentations_info=masks_info,
#                 #     disjoint_info=disjoint_info,
#                 #     heuristic=heuristic,
#                 #     length=length,
#                 #     beam_size=beam_limit,
#                 #     max_size_mask=mask_shape[0]*mask_shape[1],
#                 #     mask_shape=mask_shape,
#                 #     device=device,
#                 #     neuron_weights=neuron_weights,
#                 #     labels=config_experiment['segmentor_labels'],
#                 #     beam_variant=beam_variant,
#                 #     constraints=constraints,
#                 # )
#                 # is_default_expl =  random_label == best_label
#                 # string_random_label = F.get_formula_str(random_label, config_experiment['segmentor_labels'])


                


#                 time_taken = end_time - start_time
#                 string_label = F.get_formula_str(best_label, config_experiment['segmentor_labels'])
#                 with open(file_algo_results, "wb") as file:
#                     pickle.dump((best_label, string_label, best_iou, visited, expanded, estimated, time_taken), file)
#                 if best_label is not None and dataset is not None:
#                     unit_activations = unit_activations.to('cpu')
#                     # Extract the top k highest activating samples for the unit and cluster
#                     topk= 5
#                     if len(bitmaps.shape) == 3:
#                         only_relevant = bitmaps.sum(dim=(1,2)) > 0
#                     elif len(bitmaps.shape) ==2:
#                         only_relevant = bitmaps.sum(dim=(1)) > 0
#                     elif len(bitmaps.shape) ==1:
#                         only_relevant = bitmaps > 0
#                     else:
#                         raise ValueError(f"Bitmaps shape {bitmaps.shape} not supported")
#                     relevant_activations = unit_activations*only_relevant.to(unit_activations.device)
#                     topk_values, topk_indices = torch.topk(relevant_activations, k=min(topk, relevant_activations.shape[0]), dim=0)
#                     print(f"Top-{topk} activating samples for unit {unit} cluster {cluster_index} ({string_label}):")
#                     for idx in topk_indices:
#                         pre_sentence, hyp_sentence = dataset.print_sentence_at_idx(idx.item())
#                         print(f"Pre: {pre_sentence} | Hyp: {hyp_sentence}") 
#             else:
#                 with open(file_algo_results, "rb") as file:
#                     loaded_info = pickle.load(file)
#                     if len(loaded_info) == 7:
#                         best_label, string_label, best_iou, visited, expanded, estimated, time_taken = loaded_info
#                     elif len(loaded_info) == 6:
#                         best_label, string_label, best_iou, visited, expanded, estimated = loaded_info
#                         time_taken = -1
#                     elif len(loaded_info) == 5:
#                         best_label, string_label, best_iou, visited, expanded = loaded_info
#                         estimated = -1
#                         time_taken = -1

#                 if string_label != F.get_formula_str(best_label, config_experiment['segmentor_labels']):
#                     raise ValueError(f"The mapping used during the computation of explanation is different from the one used during the collection. String label does not match the formula {string_label} - {F.get_formula_str(best_label, config_experiment['segmentor_labels'])}")

#             if best_label is None:
#                 if verbose:
#                     print(
#                             f"Unit: {unit} - "
#                             + f"Cluster: {cluster_index} - "
#                             + f"No valid explanation found. Best IoU is 0. "
#                             )
#                 continue
#             # Compute Counter Iou
#             formula_mask = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
#                     best_label, masks)).to(bitmaps.device)
#             vanilla_iou = metrics.iou(formula_mask, bitmaps).item()
#             counter_iou = metrics.counter_iou(label_mask=formula_mask, bitmaps=bitmaps, counter_bitmaps=~bitmaps).item()
#             if verbose:
#                 if best_iou == 0:
#                     print(
#                             f"Unit: {unit} - "
#                             + f"Cluster: {cluster_index} - "
#                             + f"No valid explanation found. Best IoU is 0. "
#                             )
#                 elif best_iou < 0.04:
#                     print(
#                             f"Unit: {unit} - "
#                             + f"Cluster: {cluster_index} - "
#                             + f"Explanation found but NON-INTERPETABLE because Iou ({round(best_iou,4)}). "
#                             + f"Best Label: {string_label} - "
#                             + f"Vanilla IoU: {round(vanilla_iou,4)} - "
#                             + f"Weighted IoU: {round(counter_iou,4)} - "
#                         )
#                 else:
#                     print(
#                             f"Unit: {unit} - "
#                             + f"Cluster: {cluster_index} - "
#                             + f"Best Label: {string_label} - "
#                         # + f"Is Default Explanation: {is_default_expl} - "
#                         # + f"Default Explanation String: {string_random_label} - "
#                             + f"Number Labels: {best_label} - "
#                         # + f'Default Explanation: {random_label} - '
#                         # + f"Best IoU: {round(best_iou,4)} - "
#                             + f"Vanilla IoU: {round(vanilla_iou,4)} - "
#                             + f"Weighted IoU: {round(counter_iou,4)} - "
#                             #+ f"Visited: {visited} - "
#                             #+ f"Expanded: {expanded} - "
#                             #+ f"Estimated: {estimated}"
#                             #+ f" - Time: {time_taken:.2f} seconds \n"
#                             )
            
            
#             results.append((unit, cluster_index, activation_range, best_label, string_label, best_iou, visited, expanded, estimated, time_taken))
#     exit()
#     return results

from scipy.stats import hypergeom

def p_artifact(explanation_occ, activations_occ, overlap_occ, dataset_size):
    return hypergeom.sf(overlap_occ - 1, dataset_size, explanation_occ, activations_occ)


def compute_compositional_explanations(model,masks, masks_info, disjoint_info, activations, units, config_experiment, config_compositional, *, quantile=0.005, beam_variant=None, verbose=False, constraints=None, dataset=None, first_n_interpretable_units=None, ):
    num_clusters = config_compositional['num_clusters']
    mask_shape = config_experiment['mask_shape']
    length = config_compositional['length']
    layer_name = config_compositional['layer_name']
    device = config_experiment['device']
    results_dir = config_experiment['results_dir']
    beam_limit = config_compositional['beam_limit']
    heuristic = config_compositional['heuristic']
    results = []
    counter_variant = config_compositional['counter_variant']
    diff_threshold = config_compositional['diff_threshold']

    monodimensional_activations = len(activations.shape) == 2
    if monodimensional_activations:
        metrics_names = ["activations_coverage", "detection_accuracy", "diff_adv_iou"]
    else:
        metrics_names = ["activations_coverage", "detection_accuracy", "explanation_coverage", "samples_coverage", "diff_adv_iou"]

    # How many samples to consider for the application metric
    topk= 20
    alpha = 0.01
    # Counters
    units_covered = 0
    units_analyzed = 0
    for unit in tqdm(
                units, desc="Computing Compostional explanations per unit"
            ):
        units_analyzed += 1
        is_covered = False
        if first_n_interpretable_units is not None and units_covered >= first_n_interpretable_units:
            print(f"Reached the limit of first {first_n_interpretable_units} interpretable units, stopping the computation of explanations for the remaining units.")
            break
        if len(activations.shape) == 4:
            unit_activations = activations[:,unit,:,:]
        elif len(activations.shape) == 2:
            unit_activations = activations[:,unit]
        else:
            raise ValueError(f"Activations shape {activations.shape} not supported")

        # Filter out extreme cases where there are too few activations
        nonzero_activations = torch.count_nonzero(unit_activations)
        if quantile is not None and nonzero_activations < len(unit_activations)*quantile:
            if verbose:
                print(f"Quantile is set and Unit {unit} has very few activations ({nonzero_activations}), skipping. Required {len(unit_activations)*quantile} activations. Compo Exp Limitations for being meaningful. Please consider changing the quantile or checking the activation ranges.")
            continue

        # Compute activation range to be kept in the masks
        activation_ranges = activation_utils.compute_activation_ranges(
            unit_activations, num_clusters, quantile=quantile)
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
            beam_dir = 'optimal' if heuristic == 'optimal' else beam_limit
            if num_clusters == 0:
                dir_current_results = (
                    f"{results_dir}/{beam_dir}/length_{length}/"
                    + f"{layer_name}/{unit}/{activation_range}"
                )
            else:
                dir_current_results = (
                f"{results_dir}/{beam_dir}/length_{length}/"
                + f"{layer_name}/{unit}/quant_{quantile}/{activation_range}"
            )

        
            dir_current_results = dir_current_results + f'/{beam_variant}_beam'
            if not os.path.exists(dir_current_results):
                os.makedirs(dir_current_results)
            file_algo_results = (
                f"{dir_current_results}/" + f"{length}.pickle"
            )
            # Compute binary masks
            if unit_activations.shape != masks[1].shape:
                bitmaps = activation_utils.compute_bitmaps(
                    unit_activations,
                    activation_range,
                    mask_shape=mask_shape,
                )
            else:
                #print("Activations already in the right shape, no need to upsample")
                bitmaps = torch.where(
                    (unit_activations > activation_range[0]) & (unit_activations < activation_range[1]), 
                    True, False)


            bitmaps = bitmaps.to(device)
            non_zero_bitmaps = bitmaps.sum()
            top_k_samples = min(topk, non_zero_bitmaps.item())
            print(f"Unit {unit} cluster {cluster_index} with activation range {activation_range} has {non_zero_bitmaps} activations. Quantile: {quantile}")
            # (1187 OR 1745)
            # label_1 = F.Or(F.And(F.Leaf(38), F.Leaf(2061)), F.Leaf(3519))
            # label_2 = F.Or(F.And(F.Leaf(38), F.Leaf(2061)), F.Leaf(3967))
            # mask_label_1 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_1, masks)).to(bitmaps.device)
            # mask_label_2 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_2, masks)).to(bitmaps.device)
            # print(metrics.iou(mask_label_1, bitmaps).item())
            # print(metrics.iou(mask_label_2, bitmaps).item())
            # print(metrics.iou(mask_label_1, mask_label_2).item())
            # exit()

            # label 1 (((38 AND 2061) AND (NOT 5475)) OR 3670):
            # label 2 (((38 AND 2061) AND (NOT 5470)) OR 3670)
            # label_1 = F.Or(F.And(F.And(38, 2061), F.Not(5475)), 3670)
            # mask_label_1 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_1, masks)).to(bitmaps.device)
            # label_2 = F.Or(F.And(F.And(38, 2061), F.Not(5470)), 3670)
            # label_3 = F.Or(F.And(38, 2061), 3670)
            # label_4 = F.Leaf(3670)
            # label_5 = F.Not(5475)
            # label_6 = F.Not(5470)
            # label_7 = F.And(38, 2061)
            # label_8 = F.And(F.And(38, 2061), F.Not(5475))
            # label_9 = F.And(F.And(38, 2061), F.Not(5470))
            # label_10 = F.And(F.And(38, 2061), F.And(F.Not(5470), F.Not(5475)))
            # mask_label_2 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_2, masks)).to(bitmaps.device)
            # mask_label_3 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_3, masks)).to(bitmaps.device)
            # mask_label_4 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_4, masks)).to(bitmaps.device)
            # mask_label_5 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_5, masks)).to(bitmaps.device)
            # mask_label_6 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_6, masks)).to(bitmaps.device)
            # mask_label_7 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_7, masks)).to(bitmaps.device)
            # mask_label_8 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_8, masks)).to(bitmaps.device)
            # mask_label_9 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_9, masks)).to(bitmaps.device)
            # mask_label_10 = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         label_10, masks)).to(bitmaps.device)
            # print(metrics.iou(mask_label_1, bitmaps).item())
            # print(metrics.iou(mask_label_2, bitmaps).item())
            # print(metrics.iou(mask_label_3, bitmaps).item())
            # print(metrics.iou(mask_label_4, bitmaps).item())
            # print(metrics.iou(mask_label_5, bitmaps).item())
            # print(metrics.iou(mask_label_6, bitmaps).item())
            # print(metrics.iou(mask_label_7, bitmaps).item())
            # print(metrics.iou(mask_label_8, bitmaps).item())
            # print(metrics.iou(mask_label_9, bitmaps).item())
            # print(metrics.iou(mask_label_1, mask_label_2).item())
            # print(metrics.iou(mask_label_5, mask_label_6).item())
            # print(metrics.iou(mask_label_8, mask_label_9).item())
            # print(metrics.iou(mask_label_9, mask_label_10).item())
            # exit()

            if not os.path.exists(file_algo_results):
                if quantile is None and bitmaps.sum() < 500:
                    if verbose:
                        print(f"Mu setup for NLI. Too few activations for unit {unit} cluster {cluster_index} with range {activation_range}. Only {bitmaps.sum()} activations found. Please consider changing the number of clusters or checking the activation ranges.")
                    continue
                if verbose:
                    print("Computing explanation for unit %d using beam variant %s " % (unit, beam_variant))
                if not is_covered:
                    is_covered = True
                    units_covered += 1
                start_time = timer()
                (
                    best_label,
                    best_iou,
                    visited,
                    expanded,
                    estimated
                ) = algorithms.get_heuristic_scores(
                    masks,
                    bitmaps,
                    segmentations_info=masks_info,
                    disjoint_info=disjoint_info,
                    heuristic=heuristic,
                    length=length,
                    beam_size=beam_limit,
                    max_size_mask=mask_shape[0]*mask_shape[1],
                    mask_shape=mask_shape,
                    device=device,
                    labels=config_experiment['segmentor_labels'],
                    beam_variant=beam_variant,
                    constraints=constraints,
                    counter_variant=counter_variant,
                    diff_threshold=diff_threshold
                )
                end_time = timer()
                time_taken = end_time - start_time
                string_label = F.get_formula_str(best_label, config_experiment['segmentor_labels'])
                with open(file_algo_results, "wb") as file:
                    pickle.dump((best_label, string_label, best_iou, visited, expanded, estimated, time_taken), file)
            else:
                with open(file_algo_results, "rb") as file:
                    loaded_info = pickle.load(file)
                    best_label, string_label, best_iou, visited, expanded, estimated, time_taken = loaded_info
                    if not is_covered:
                        is_covered = True
                        units_covered += 1
                if string_label != F.get_formula_str(best_label, config_experiment['segmentor_labels']):
                    raise ValueError(f"The mapping used during the computation of explanation is different from the one used during the collection. String label does not match the formula {string_label} - {F.get_formula_str(best_label, config_experiment['segmentor_labels'])}")

            if best_label is None:
                if verbose:
                    print(
                            f"Unit: {unit} - "
                            + f"Cluster: {cluster_index} - "
                            + f"No valid explanation found. Best IoU is 0. "
                            )
                continue
            

            # Metrics computation
            if best_label is not None and dataset is not None:
                formula_mask = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
                    best_label, masks)).to(bitmaps.device)
                unit_activations = unit_activations.to('cpu')
                # Extract the top k highest activating samples for the unit and cluster
                if len(bitmaps.shape) == 3:
                    only_relevant = bitmaps.sum(dim=(1,2)) > 0
                elif len(bitmaps.shape) ==2:
                    only_relevant = bitmaps.sum(dim=(1)) > 0
                elif len(bitmaps.shape) ==1:
                    only_relevant = bitmaps > 0
                else:
                    raise ValueError(f"Bitmaps shape {bitmaps.shape} not supported")
                relevant_activations = unit_activations*only_relevant.to(unit_activations.device)
                topk_values, topk_indices = torch.topk(relevant_activations, k=min(top_k_samples, relevant_activations.shape[0]), dim=0)
                if verbose:
                    print(f"Top-{top_k_samples} activating samples for unit {unit} cluster {cluster_index} ({string_label}):")
                unit_apply = 0
                for idx in topk_indices:
                    pre_sentence, hyp_sentence, target = dataset.print_sentence_at_idx(idx.item())
                    if formula_mask[idx.item()]:
                        unit_apply += 1
                    if verbose:
                        print(f"Apply:{formula_mask[idx.item()]}| Pre: {pre_sentence} | Hyp: {hyp_sentence} | Target: {target}") 
                firing_rate = nonzero_activations.item() / unit_activations.numel() 
                p_artifact_value = p_artifact(explanation_occ=formula_mask.sum().item(), activations_occ=bitmaps.sum().item(), overlap_occ=(formula_mask & bitmaps).sum().item(), dataset_size=bitmaps.numel())

            else:
                unit_apply = -1
                firing_rate = 0
                p_artifact_value = 0
            metrics_values = metrics.compute_metrics(metrics_names, best_label, masks, bitmaps)
            metrics_values['apply'] = unit_apply / top_k_samples if top_k_samples > 0 else -1
            metrics_values['iou'] = best_iou
            metrics_values['visited'] = visited
            metrics_values['expanded'] = expanded
            metrics_values['estimated'] = estimated
            metrics_values['time_taken'] = time_taken
            metrics_values['is_not_incremental'] = not is_incremental(best_label)
            metrics_values['descriptivity'] = non_zero_bitmaps.item() / nonzero_activations.item()
            metrics_values['firing_rate'] = firing_rate
            metrics_values['p_artifact'] = p_artifact_value
            #metrics_values['realiable'] = metrics_values['p_artifact'] < alpha
            #print(f"p-artifact for unit {unit} cluster {cluster_index} with label {string_label}: {metrics_values['p_artifact']}")
            #print(f"N:{bitmaps.numel()} | FE: {formula_mask.sum().item()} | K: {bitmaps.sum().item()} | C: {(formula_mask & bitmaps).sum().item()}")
            # vanilla_iou = best_iou
            # # # Compute Counter Iou
            # formula_mask = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
            #         best_label, masks)).to(bitmaps.device)
            # # vanilla_iou = metrics.iou(formula_mask, bitmaps).item()
            # adv_iou = metrics.iou(formula_mask, ~bitmaps).item()
            # diff = max(vanilla_iou - adv_iou, 0)
            # # normalize diff in a way that if vanilla_iou and adv_iou are small and their difference is small is comparable to a bigger difference but with vanilla_iou and adv_iou larger. Basically, we want to express the difference in percentage and independently from the size of iou
            # diff = diff / (vanilla_iou + 1e-5)
            # avg_diff.append(diff)
            # avg_iou.append(vanilla_iou)

            # # Other metrics
            # act_coverage = metrics.activations_coverage(formula_mask, bitmaps).item()
            # det_accuracy = metrics.detection_accuracy(formula_mask, bitmaps).item()

            # counter_iou = metrics.counter_iou(label_mask=formula_mask, bitmaps=bitmaps, counter_bitmaps=~bitmaps).item()
            #if verbose:
            if best_iou == 0:
                print(
                        f"Unit: {unit} - "
                        + f"Cluster: {cluster_index} - "
                        + f"No valid explanation found. Best IoU is 0. "
                        )
            # elif best_iou < 0.04:
            #     print(
            #             f"Unit: {unit} - "
            #             + f"Cluster: {cluster_index} - "
            #             + f"Explanation found but NON-INTERPETABLE because Iou ({round(best_iou,4)}). "
            #             + f"Best Label: {string_label} - "
            #             + f"Vanilla IoU: {round(vanilla_iou,4)} - "
            #             + f"Weighted IoU: {round(counter_iou,4)} - "
            #         )
            else:
                #if diff < 0.3 and (cluster_index == 4 or (cluster_index == 0 and num_clusters <= 1)):
                if cluster_index == 4 or (cluster_index == 0 and num_clusters <= 1):
                    print(
                            f"Unit: {unit} - "
                            + f"Cluster: {cluster_index} - "
                            + f"Best Label: {string_label} - "
                        # + f"Is Default Explanation: {is_default_expl} - "
                        # + f"Default Explanation String: {string_random_label} - "
                            + f"Number Labels: {best_label} - "
                        # + f'Default Explanation: {random_label} - '
                        + f"Best IoU: {round(best_iou,4)} - "
                        # + f"Vanilla IoU: {round(best_iou,4)} - "
                        #+ f"Weighted IoU: {round(counter_iou,4)} - "
                        + f"Diff IoU: {round(metrics_values['diff_adv_iou'],4)} - "
                        + f"Act Coverage: {round(metrics_values['activations_coverage'],4)} - "
                        + f"Det Accuracy: {round(metrics_values['detection_accuracy'],4)} - "

                            + f"Visited: {visited} - "
                            + f"Expanded: {expanded} - "
                            + f"Estimated: {estimated}"
                            + f" - Time: {time_taken:.2f} seconds \n"
                            )
                #print(f"Average Diff IoU so far: {round(sum(avg_diff)/len(avg_diff),4)} - Covered Units: {covered}/{len(units)} - Average Apply: {round(sum(avg_apply)/len(avg_apply),4)} - Average IoU: {round(sum(avg_iou)/len(avg_iou),4)}")
            
            
            results.append((unit, cluster_index, activation_range, best_label, string_label, metrics_values))
    return results



def iterative_clustering(model,masks, masks_info, disjoint_info, activations, units, config_experiment, config_compositional, *, interpolation_mode='bilinear', quantile=None, beam_variant=None, verbose=True, constraints=None, dataset=None):
    num_clusters = config_compositional['num_clusters']
    mask_shape = config_experiment['mask_shape']
    length = config_compositional['length']
    layer_name = config_compositional['layer_name']
    device = config_experiment['device']
    results_dir = config_experiment['results_dir']
    beam_limit = config_compositional['beam_limit']
    heuristic = config_compositional['heuristic']
    results = []
    counter_variant = config_compositional['counter_variant']
    avg_diff = []
    avg_apply = []
    avg_iou = []
    covered = 0
    num_clusters = 2
    length = 1
    for unit in tqdm(
                units, desc="Computing Compostional explanations per unit"
            ):
        if len(activations.shape) == 4:
            unit_activations = activations[:,unit,:,:]
        elif len(activations.shape) == 2:
            unit_activations = activations[:,unit]
        else:
            raise ValueError(f"Activations shape {activations.shape} not supported")

        # Filter out extreme cases where there are too few activations
        nonzero_activations = torch.count_nonzero(unit_activations)

        if num_clusters == 1 and nonzero_activations < len(unit_activations)*C.NETDISSECT_QUANTILE:
            print(f"Unit {unit} has very few activations ({nonzero_activations}), skipping. Compositional Limit.")
            continue
       
        # Compute activation range to be kept in the masks
        clusters_to_decompose = [(-float('inf'), float('inf'))]
        clusters_to_keep = []
        while len(clusters_to_decompose) > 0:
            print()
            print("Summary so far: ")
            print(f"Unit {unit} - Clusters to decompose: {clusters_to_decompose} ({len(clusters_to_decompose)} clusters) - Clusters to keep: {clusters_to_keep} ({len(clusters_to_keep)} clusters) - Average Diff IoU: {round(sum(avg_diff)/len(avg_diff),4) if len(avg_diff) > 0 else 'N/A'}")
            print()

            # Collect clusters to parse
            clusters_to_parse = []
            for lower_bound, upper_bound in clusters_to_decompose:
                filtered_activations = torch.where((unit_activations >= lower_bound) & (unit_activations <= upper_bound), unit_activations, torch.tensor(0.0).to(unit_activations.device))
                cluster_boundaries = activation_utils.compute_activation_ranges(
                     filtered_activations, 2, quantile=quantile)   
                print(f"Decomposing cluster ({lower_bound}, {upper_bound}) into subclusters: {cluster_boundaries}")
                clusters_to_parse.extend(cluster_boundaries)
            
            # Parse clusters
            clusters_to_decompose = []
            for cluster_index, activation_range in enumerate(
                        sorted(clusters_to_parse)
                    ):
                print(f"Unit: {unit} - Activation Range: {activation_range}")
                # Compute binary masks
                if unit_activations.shape != masks[1].shape:
                    bitmaps = activation_utils.compute_bitmaps(
                        unit_activations,
                        activation_range,
                        mask_shape=mask_shape,
                    )
                else:
                    bitmaps = torch.where(
                        (unit_activations > activation_range[0]) & (unit_activations < activation_range[1]), 
                        True, False)
                


                bitmaps = bitmaps.to(device)


                    
                start_time = timer()
                (
                    best_label,
                    best_iou,
                    visited,
                    expanded,
                    estimated
                ) = algorithms.get_heuristic_scores(
                    masks,
                    bitmaps,
                    segmentations_info=masks_info,
                    disjoint_info=disjoint_info,
                    heuristic=heuristic,
                    length=length,
                    beam_size=beam_limit,
                    max_size_mask=mask_shape[0]*mask_shape[1],
                    mask_shape=mask_shape,
                    device=device,
                    labels=config_experiment['segmentor_labels'],
                    beam_variant=beam_variant,
                    constraints=constraints,
                    counter_variant=counter_variant
                )
                end_time = timer()


                time_taken = end_time - start_time
                string_label = F.get_formula_str(best_label, config_experiment['segmentor_labels'])


                vanilla_iou = best_iou
                # # Compute Counter Iou
                formula_mask = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
                        best_label, masks)).to(bitmaps.device)
                adv_iou = metrics.iou(formula_mask, ~bitmaps).item()
                diff = max(vanilla_iou - adv_iou, 0)
                # normalize diff in a way that if vanilla_iou and adv_iou are small and their difference is small is comparable to a bigger difference but with vanilla_iou and adv_iou larger. Basically, we want to express the difference in percentage and independently from the size of iou
                diff = diff / (vanilla_iou + 1e-5)
                avg_diff.append(diff)
                avg_iou.append(vanilla_iou)


                print(
                        f"Unit: {unit} - "
                        + f"Cluster: {cluster_index} - "
                        + f"Activation Range: {activation_range} - "
                        + f"Best Label: {string_label} - "
                        + f"Number Labels: {best_label} - "
                    + f"Vanilla IoU: {round(vanilla_iou,4)} - "
                    + f"Adversarial IoU: {round(adv_iou,4)} - "
                    + f"Diff IoU: {round(diff,4)}"
                        )
                print(f"Average Diff IoU so far: {round(sum(avg_diff)/len(avg_diff),4)} - Covered Units: {covered}/{len(units)} - Average IoU: {round(sum(avg_iou)/len(avg_iou),4)}")
                
                # Decide whether to decompose further or keep the cluster as it is based on the diff between vanilla_iou and adv_iou. If the diff is small, it means that the explanation is not very specific to the cluster and we can try to decompose it further. If the diff is large, it means that the explanation is specific to the cluster and we can keep it as it is. 
                if diff < 0.9:
                    clusters_to_decompose.append(activation_range)
                    print(f"\tCluster {activation_range} is not specific enough (Diff IoU: {round(diff,4)}), decomposing further.")
                else:
                    clusters_to_keep.append(activation_range)
                    print(f"\tCluster {activation_range} is specific enough (Diff IoU: {round(diff,4)}), keeping it as it is.")
            
    exit()
    return results

def compute_random_explanations(model,masks, masks_info, disjoint_info, activations, units, config_experiment, config_compositional, *, interpolation_mode='bilinear', beam_variant=None, verbose=True, constraints=None, dataset=None):
    num_clusters = config_compositional['num_clusters']
    mask_shape = config_experiment['mask_shape']
    length = config_compositional['length']
    layer_name = config_compositional['layer_name']
    device = config_experiment['device']
    results_dir = config_experiment['results_dir']
    beam_limit = config_compositional['beam_limit']
    heuristic = config_compositional['heuristic']
    results = []
    counter_variant = config_compositional['counter_variant']
    for unit in tqdm(
                units, desc="Computing Compostional explanations per unit"
            ):
        if len(activations.shape) == 4:
            unit_activations = activations[:,unit,:,:]
        elif len(activations.shape) == 2:
            unit_activations = activations[:,unit]
        else:
            raise ValueError(f"Activations shape {activations.shape} not supported")

        # Filter out extreme cases where there are too few activations
        nonzero_activations = torch.count_nonzero(unit_activations)
        if nonzero_activations < num_clusters:
            print(f"Unit {unit} cluster {cluster_index} has very few activations ({nonzero_activations}), skipping. Please consider changing the number of clusters or checking the activation ranges.")
            continue
        # Compute activation range to be kept in the masks
        activation_ranges = activation_utils.compute_activation_ranges(
            unit_activations, num_clusters)
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
            beam_dir = 'optimal' if heuristic == 'optimal' else beam_limit
            dir_current_results = (
                f"{results_dir}/{beam_dir}/length_{length}/"
                + f"{layer_name}/{unit}/{activation_range}"
            )
            if beam_variant == 'old':
                dir_current_results = dir_current_results + '/old_beam'
            elif beam_variant == 'new':
                dir_current_results = dir_current_results + '/new_beam'
            elif beam_variant is None:
                dir_current_results = dir_current_results + '/baseline'
            if counter_variant:
                dir_current_results = dir_current_results + '/counter_variant'
            if not os.path.exists(dir_current_results):
                os.makedirs(dir_current_results)
            file_algo_results = (
                f"{dir_current_results}/random/" + f"{length}.pickle"
            )



            # Compute binary masks
            if unit_activations.shape != masks[1].shape:
                bitmaps = activation_utils.compute_bitmaps(
                    unit_activations,
                    activation_range,
                    mask_shape=mask_shape,
                )
            else:
                #print("Activations already in the right shape, no need to upsample")
                bitmaps = torch.where(
                    (unit_activations > activation_range[0]) & (unit_activations < activation_range[1]), 
                    True, False)


            bitmaps = bitmaps.to(device)
            if not os.path.exists(file_algo_results):

                bitmaps_perc = bitmaps.sum()/bitmaps.numel()
                random_bitmaps = torch.bernoulli(torch.full(bitmaps.shape, bitmaps_perc)).bool().to(device)

                if dataset is not None and bitmaps.sum() < 500:
                    #print(f"Too few activations for unit {unit} cluster {cluster_index} with range {activation_range}. Only {bitmaps.sum()} activations found. Please consider changing the number of clusters or checking the activation ranges.")
                    continue
                start_time = timer()
                (
                    best_label,
                    best_iou,
                    visited,
                    expanded,
                    estimated
                ) = algorithms.get_heuristic_scores(
                    masks,
                    random_bitmaps,
                    segmentations_info=masks_info,
                    disjoint_info=disjoint_info,
                    heuristic=heuristic,
                    length=length,
                    beam_size=beam_limit,
                    max_size_mask=mask_shape[0]*mask_shape[1],
                    mask_shape=mask_shape,
                    device=device,
                    labels=config_experiment['segmentor_labels'],
                    beam_variant=beam_variant,
                    constraints=constraints,
                    counter_variant=counter_variant
                )
                end_time = timer()



                time_taken = end_time - start_time
                string_label = F.get_formula_str(best_label, config_experiment['segmentor_labels'])
                with open(file_algo_results, "wb") as file:
                    pickle.dump((best_label, string_label, best_iou, visited, expanded, estimated, time_taken), file)
            else:
                with open(file_algo_results, "rb") as file:
                    loaded_info = pickle.load(file)
                    if len(loaded_info) == 7:
                        best_label, string_label, best_iou, visited, expanded, estimated, time_taken = loaded_info
                    elif len(loaded_info) == 6:
                        best_label, string_label, best_iou, visited, expanded, estimated = loaded_info
                        time_taken = -1
                    elif len(loaded_info) == 5:
                        best_label, string_label, best_iou, visited, expanded = loaded_info
                        estimated = -1
                        time_taken = -1

                if string_label != F.get_formula_str(best_label, config_experiment['segmentor_labels']):
                    raise ValueError(f"The mapping used during the computation of explanation is different from the one used during the collection. String label does not match the formula {string_label} - {F.get_formula_str(best_label, config_experiment['segmentor_labels'])}")

            if best_label is None:
                if verbose:
                    print(
                            f"Unit: {unit} - "
                            + f"Cluster: {cluster_index} - "
                            + f"No valid explanation found. Best IoU is 0. "
                            )
                continue
            
            if verbose:
                if best_iou == 0:
                    print(
                            f"Unit: {unit} - "
                            + f"Cluster: {cluster_index} - "
                            + f"No valid explanation found. Best IoU is 0. "
                            )
                elif best_iou < 0.04:
                    print(
                            f"Unit: {unit} - "
                            + f"Cluster: {cluster_index} - "
                            + f"Explanation found but NON-INTERPETABLE because Iou ({round(best_iou,4)}). "
                            + f"Best Label: {string_label} - "
                        )
                else:
                    print(
                            f"Unit: {unit} - "
                            + f"Cluster: {cluster_index} - "
                            + f"Best Label: {string_label} - "
                        # + f"Is Default Explanation: {is_default_expl} - "
                        # + f"Default Explanation String: {string_random_label} - "
                            + f"Number Labels: {best_label} - "
                        # + f'Default Explanation: {random_label} - '
                        # + f"Best IoU: {round(best_iou,4)} - "
                            + f"Vanilla IoU: {round(best_iou,4)} - "
                            #+ f"Visited: {visited} - "
                            #+ f"Expanded: {expanded} - "
                            #+ f"Estimated: {estimated}"
                            #+ f" - Time: {time_taken:.2f} seconds \n"
                            )
            
            
            results.append((unit, cluster_index, activation_range, best_label, string_label, best_iou, visited, expanded, estimated, time_taken))
    exit()
    return results



def load_compositional_explanations(masks, masks_info, disjoint_info, activations, units, config_experiment, config_compositional, verbose=True, beam_variant=None):
    num_clusters = config_compositional['num_clusters']
    mask_shape = config_experiment['mask_shape']
    length = config_compositional['length']
    layer_name = config_compositional['layer_name']
    device = config_experiment['device']
    results_dir = config_experiment['results_dir']
    beam_limit = config_compositional['beam_limit']
    heuristic = config_compositional['heuristic']
    counter_variant = config_compositional['counter_variant']
    results = []
    for unit in tqdm(
                units, desc="Loading Compostional explanations per unit"
            ):
        unit_activations = activations[:,unit,:,:]

        if unit_activations.sum() == 0:
            print(f"Unit {unit} has no activation, skipping")
            for cluster_index in range(num_clusters):
                results.append((unit, cluster_index, None, None, None, None, None, None, None, None))
            continue
        if torch.count_nonzero(unit_activations) < num_clusters:
            print(f"Unit {unit} has very few activations ({torch.count_nonzero(unit_activations)}), skipping")
            for cluster_index in range(num_clusters):
                results.append((unit, cluster_index, None, None, None, None, None, None, None, None))
            continue

        # Compute activation range to be kept in the masks
        activation_ranges = activation_utils.compute_activation_ranges(
            unit_activations, num_clusters)
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):



            beam_dir = 'optimal' if heuristic == 'optimal' else beam_limit
            dir_current_results = (
                f"{results_dir}/{beam_dir}/length_{length}/"
                + f"{layer_name}/{unit}/{activation_range}"
            )
            if beam_variant == 'old':
                dir_current_results = dir_current_results + '/old_beam'
            elif beam_variant == 'new':
                dir_current_results = dir_current_results + '/new_beam'
            elif beam_variant is None:
                dir_current_results = dir_current_results + '/baseline'
            if counter_variant:
                dir_current_results = dir_current_results + '/counter_variant'
            # dir_before = f"{results_dir}/{beam_dir}/length_{length}/"+ f"{layer_name}/{unit}/"
            # if os.path.exists(dir_before):
            #     all_dirs = os.listdir(dir_before)
            # else:
            #     all_dirs = []
            # if len(all_dirs) == 1:
            #     dir_current_results = f"{dir_before}/{all_dirs[0]}"
            #     file_algo_results = (
            #         f"{dir_current_results}/" + f"{length}.pickle"
            #     )
            #     print(f"Only one directory found, using {file_algo_results}")
            # else:
            file_algo_results = (
                f"{dir_current_results}/" + f"{length}.pickle"
            )
            if not os.path.exists(file_algo_results):
                print(f"File {file_algo_results} does not exist. You need to run the clustering script first.")
                best_label, string_label, best_iou, visited, expanded, estimated, time_taken = None, None, None, None, None, None, None
            else:
                with open(file_algo_results, "rb") as file:
                    loaded_info = pickle.load(file)
                    if len(loaded_info) == 7:
                        best_label, string_label, best_iou, visited, expanded, estimated, time_taken = loaded_info
                    elif len(loaded_info) == 6:
                        best_label, string_label, best_iou, visited, expanded, estimated = loaded_info
                        time_taken = -1
                    elif len(loaded_info) == 5:
                        best_label, string_label, best_iou, visited, expanded = loaded_info
                        estimated = -1
                        time_taken = -1
                    best_iou = round(best_iou, 8) # To avoid problems with float precision between torch and numpy
                if string_label != F.get_formula_str(best_label, config_experiment['segmentor_labels']):
                    raise ValueError(f"The mapping used during the computation of explanation is different from the one used during the collection. String label does not match the formula {string_label} - {F.get_formula_str(best_label, config_experiment['segmentor_labels'])}")

            if verbose:
                print(
                        f"Unit: {unit} - "
                        + f"Cluster: {cluster_index} - "
                        + f"Best Label: {string_label} - "
                        + f"Number Labels: {best_label} - "
                        + f"Best IoU: {round(best_iou,4)} - "
                        + f"Visited: {visited} - "
                        + f"Expanded: {expanded} - "
                        + f"Estimated: {estimated}"
                        + f" - Time: {time_taken:.2f} seconds"
                        )

            results.append((unit, cluster_index, activation_range, best_label, string_label, best_iou, visited, expanded, estimated, time_taken))
    return results

def collect_compositional_explanations(activations, units, config_experiment, config_compositional):
    num_clusters = config_compositional['num_clusters']
    length = config_compositional['length']
    layer_name = config_compositional['layer_name']
    results_dir = config_experiment['results_dir']
    concets_labels = config_experiment['segmentor_labels']

    results = {}
    for unit in tqdm(
                units, desc="Collect Compostional explanations per unit"
            ):
        unit_activations = activations[:,unit,:,:]

        # Compute activation range to be kept in the masks
        activation_ranges = activation_utils.compute_activation_ranges(
            unit_activations, num_clusters)
        unit_dict = {}
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
            dir_current_results = (
                f"{results_dir}/"
                + f"{layer_name}/{unit}/{activation_range}"
            )
            if not os.path.exists(dir_current_results):
                os.makedirs(dir_current_results)
            file_algo_results = (
                f"{dir_current_results}/" + f"{length}.pickle"
            )
            if not os.path.exists(file_algo_results):
                  raise ValueError(f"File {file_algo_results} does not exist. You need to run the clustering script first.")                  
            else:
                with open(file_algo_results, "rb") as file:
                    loaded_info = pickle.load(file)
                if len(loaded_info) == 3:
                    print(f"Old version of the file {file_algo_results}, we need to add the string label")
                    # Old version of the file, we need to add the string label
                    best_label, best_iou, visited = loaded_info
                    string_label = F.get_formula_str(best_label, config_experiment['segmentor_labels'])
                    with open(file_algo_results, "wb") as file:
                        pickle.dump((best_label, string_label, best_iou, visited, ), file)
                elif len(loaded_info) == 4:
                    best_label, string_label, best_iou, visited = loaded_info  
                if string_label != F.get_formula_str(best_label, concets_labels):
                    raise ValueError(f"The mapping used during the computation of explanation is different from the one used during the collection. String label does not match the formula {string_label} - {F.get_formula_str(best_label, concets_labels)}")
            unit_dict[cluster_index] = ((best_label, string_label), best_iou, visited, activation_range)
            #results.append((unit, cluster_index, activation_range, best_label, best_iou, visited))
        results[unit] = unit_dict
    return results

def get_layer_activations(cfg):
    weights = cfg.get_weights()
    dataset_name = cfg.dataset 
    layer_name= cfg.layer
    activation_dir = cfg.get_root_activations() + f"/{dataset_name}/{cfg.model}"
    if cfg.model == 'resnet_cub200':
        model_wrapper = Cub200Model(weights=weights, device=cfg.device)
    elif cfg.pretrained == 'places365':
        if cfg.model == 'densenet161':
            model_wrapper = DenseNetPlace365(model_name=cfg.model, weights=weights, device=cfg.device)
        else:
            model_wrapper = Place365Model(model_name=cfg.model, weights=weights, device=cfg.device)
    elif cfg.pretrained == 'imagenet':
        model_wrapper = ImageNetModel(model_name=cfg.model, device=cfg.device)
    elif cfg.pretrained == 'untrained' or cfg.pretrained is None:
        model_wrapper = UntrainedModel(model_name=cfg.model, device=cfg.device)
    elif cfg.pretrained == 'nli':
        print("si")
        exit()
    else:
        raise ValueError("Pretrained model not supported")

    # Get Masks Information from the concept dataset
    if dataset_name == 'broden':
        model_wrapper.set_loader(dataset_name, cfg)
    else:
        model_wrapper.set_loader(dataset_name)
    
    layer_activations = model_wrapper.get_layer_activations(layer_name, activation_dir)
    return layer_activations

def load_model(cfg):
    weights = cfg.get_weights()
    dataset_name = cfg.dataset 
    if cfg.model == 'resnet_cub200':
        model_wrapper = Cub200Model(weights=weights, device=cfg.device)
    elif cfg.pretrained == 'places365':
        if cfg.model == 'densenet161':
            model_wrapper = DenseNetPlace365(model_name=cfg.model, weights=weights, device=cfg.device)
        else:
            model_wrapper = Place365Model(model_name=cfg.model, weights=weights, device=cfg.device)
    elif cfg.pretrained == 'imagenet':
        model_wrapper = ImageNetModel(model_name=cfg.model, device=cfg.device)
    elif cfg.pretrained == 'untrained' or cfg.pretrained is None:
        model_wrapper = UntrainedModel(model_name=cfg.model, device=cfg.device)
    elif cfg.pretrained == 'nli':
        model_wrapper = NLPModelWrapper(model_name=cfg.model, weights=weights, device=cfg.device)
    else:
        raise ValueError("Pretrained model not supported")

    # Get Masks Information from the concept dataset
    if dataset_name == 'broden':
        model_wrapper.set_loader(dataset_name, cfg)
    else:
        model_wrapper.set_loader(dataset_name)
    return model_wrapper

# def get_layer_activations_and_weights(cfg, attribution_method=None, dir_attributions=None):
#     weights = cfg.get_weights()
#     dataset_name = cfg.dataset 
#     layer_name= cfg.layer
#     activation_dir = cfg.get_root_activations() + f"/{dataset_name}/{cfg.model}"
#     if cfg.model == 'resnet_cub200':
#         model_wrapper = Cub200Model(weights=weights, device=cfg.device)
#     elif cfg.pretrained == 'places365':
#         if cfg.model == 'densenet161':
#             model_wrapper = DenseNetPlace365(model_name=cfg.model, weights=weights, device=cfg.device)
#         else:
#             model_wrapper = Place365Model(model_name=cfg.model, weights=weights, device=cfg.device)
#     elif cfg.pretrained == 'imagenet':
#         model_wrapper = ImageNetModel(model_name=cfg.model, device=cfg.device)
#     else:
#         raise ValueError("Pretrained model not supported")

#     # Get Masks Information from the concept dataset
#     if dataset_name == 'broden':
#         model_wrapper.set_loader(dataset_name, cfg)
#     else:
#         model_wrapper.set_loader(dataset_name)
    
#     # layer_activations = model_wrapper.save_activations_weights(layer_name, dir_attributions=dir_attributions, attribution_method=attribution_method)
#     # exit()
#     layer_activations = model_wrapper.get_layer_activations(layer_name, activation_dir,)
    
#     return layer_activations

def get_selected_units(layer_activations, units=None, random_units=0):
    if units is not None:
        for unit in units:
            if int(unit) >= layer_activations.shape[1]:
                raise ValueError(f"Unit {unit} is out of range")
        selected_units = [int(unit) for unit in units]
    elif random_units > 0:
        selected_units = random.sample(
            range(layer_activations.shape[1]), random_units)
    else:
        selected_units = range(layer_activations.shape[1])
    return selected_units
    
def get_single_token_label(f, names_vector):
    if isinstance(f, F.And) or isinstance(f, F.Or):
        # something
        str_rep_left = get_single_token_label(f.left, names_vector)
        str_rep_right = get_single_token_label(f.right, names_vector)
        str_rep = []
        for index_left in range(len(str_rep_left)):
            for index_right in range(len(str_rep_right)):
                str_rep.append(f'{str_rep_left[index_left]} {f.op} {str_rep_right[index_right]}')

        return str_rep
    elif isinstance(f, F.Not):
        # something
        str_rep = F.get_formula_str( f.val, names_vector)
        if ', ' in str_rep:
            str_rep = str_rep.split(', ')
        else:
            str_rep = [str_rep]
        for index in range(len(str_rep)):
            str_rep[index] = f'NOT {str_rep[index]}'
        return str_rep
    else:
        str_rep = F.get_formula_str(f, names_vector)
        if ', ' in str_rep:
            return str_rep.split(', ')
        else:
            return [str_rep]

def check_label_in_other(atom_to_check,  labels_check, reference_atoms, labels_ref):

    # Transform the atom in single token
    comp2_str_atom = get_single_token_label(atom_to_check, labels_check)
    comp2_str_atom.extend(get_single_token_label(atom_to_check.flip_atoms(), labels_check))

    # Atoms are numeric, but diffferent models can use different numeric values
    # We need to check the labels
    gt_str_atoms=[]
    for f in reference_atoms:
        if isinstance(f, F.And):
            # In this case we need to include both the equivalent formulas of swaped atoms
            gt_str_atoms.extend(get_single_token_label(f, labels_ref))
            gt_str_atoms.extend(get_single_token_label(f.flip_atoms(), labels_ref))
        else:
            # in all the other cases we can just consider their string representation
            str_repr = get_single_token_label(f, labels_ref)
            gt_str_atoms.extend(str_repr)
    for atom in comp2_str_atom:
        if atom in gt_str_atoms:
            return True
    return False




def get_num_detected(atom_to_check,  labels_check, reference_atoms, labels_ref, include_specialization=True):

    # Specialization case e.g., (weel) is considered equivalent to (weel AND NOT building AND NOT sky)
    if include_specialization and isinstance(atom_to_check, F.And) and isinstance(atom_to_check.right, F.Not):
        return get_num_detected(atom_to_check.left, labels_check, reference_atoms, labels_ref, include_specialization)
    elif include_specialization and isinstance(reference_atoms,F.And) and isinstance(reference_atoms.right, F.Not):
        return get_num_detected(atom_to_check, labels_check, reference_atoms.left, labels_ref, include_specialization)
    else:
        # Removing the NOT could produce not atomic form
        new_atoms_to_check = atom_to_check.get_atoms()
        num_detected = 0
        for atom in new_atoms_to_check:
            if check_label_in_other(atom, labels_check, reference_atoms, labels_ref):
                num_detected += len(atom)
        return num_detected
    
def extract_base_atoms(label):
    label_atoms = label.get_atoms()
    base_atoms = []
    for atom in label_atoms:
        if isinstance(atom, F.And):
            base_atoms.extend(extract_base_atoms(atom.left))
        else:
            base_atoms.append(atom)
    return base_atoms

def compare_explanations_base_atoms(expl1, labels_expl1, expl2, labels_exp2, specialization):
    expl1_atoms = extract_base_atoms(expl1)
    expl2_atoms = extract_base_atoms(expl2)
    atoms_not_detected = []
    atoms_detected = []
    for atom in expl1_atoms:
        labels_in_atom = len(atom)
        num_detected_labels = get_num_detected(atom, labels_expl1, expl2_atoms, labels_exp2, specialization)
        if num_detected_labels != labels_in_atom:
            atoms_not_detected.append(atom)
        else:
            atoms_detected.append(atom)
    return atoms_not_detected, atoms_detected
