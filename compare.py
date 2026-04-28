"""Script to run the clustering algorithm for compositional explanations.
"""

import os

import torch
import absl.flags
import absl.app
import numpy as np



from compositional import mask_utils
from compositional import utils
from compositional import formula as F
from src import utils
from src import settings
from utils import common_flags # import all user flags
from utils import segmentor_utils

# Flags specific to this script
absl.flags.DEFINE_string("output_path", "output/explanations", "output path")

FLAGS = absl.flags.FLAGS




def extract_info(compo_exp):
    total_visited = []
    total_expanded = []
    total_estimated = []
    total_time = []
    total_iou = []
    total_units = []
    for index in range(len(compo_exp)):
        (unit, _, _, _, _, best_iou, visited, expanded, estimated, time_taken) = compo_exp[index]
        if visited is not None:
            total_visited.append(visited)
        if expanded is not None:
            total_expanded.append(expanded)
        if estimated is not None and estimated > -1:
            total_estimated.append(estimated)
        if time_taken is not None and time_taken > -1:
            total_time.append(time_taken)
        if best_iou is not None and best_iou > -1:
            total_units.append(unit)
            total_iou.append(best_iou)
    return total_units,total_visited, total_expanded, total_estimated, total_time, total_iou


def pairwise_compare(name_1, compo_exp_1, name_2, compo_exp_2):
    assert len(compo_exp_1) == len(compo_exp_2), "The two explanation lists must have the same length"

    # Counters
    same_explanations_counter = 0
    first_better_counter = 0
    second_better_counter = 0
    same_iou_counter = 0
    parsed_by_both_counter = 0

    # List
    first_better_ious = []
    second_better_ious = []
    first_ious_when_diff = []
    second_ious_when_diff = []
    first_better_diff_ious = []
    second_better_diff_ious = []

    for index in range(len(compo_exp_1)):
        if index not in range(len(compo_exp_2)):
            # Compare only the units parsed by both
            continue
        (unit_1, _, _, best_label_1, _, best_iou_1, _, _, _, _) = compo_exp_1[index]
        (unit_2, _, _, best_label_2_, _, best_iou_2, _, _, _, _) = compo_exp_2[index]
        if best_label_1 is None or best_label_2_ is None:
            # Compare only the units for which both methods found an explanation
            continue
        parsed_by_both_counter += 1

        assert unit_1 == unit_2, f"The units at index {index} do not match: {unit_1} vs {unit_2}"
        if best_label_1 == best_label_2_:
            same_explanations_counter += 1
        elif best_iou_1 > best_iou_2:
            print("Incremental is better for unit", unit_1)
            first_better_counter += 1
            first_better_ious.append(best_iou_1)
            first_ious_when_diff.append(best_iou_1)
            second_ious_when_diff.append(best_iou_2)
            first_better_diff_ious.append(best_iou_1 - best_iou_2)
        elif best_iou_2 > best_iou_1:
            second_better_counter += 1
            second_better_ious.append(best_iou_2)
            first_ious_when_diff.append(best_iou_1)
            second_ious_when_diff.append(best_iou_2)
            second_better_diff_ious.append(best_iou_2 - best_iou_1)
        if best_iou_1 == best_iou_2:
            same_iou_counter += 1
    print("******************************************************")
    print(f"Comparing {name_1} and {name_2} over {parsed_by_both_counter} units")
    print(f"Same explanations: {same_explanations_counter} ({same_explanations_counter/parsed_by_both_counter:.2%})")
    print(f"{name_1} better: {first_better_counter} ({first_better_counter/parsed_by_both_counter:.2%}) - Mean IoU: {np.mean(first_better_ious):.4f}")
    print(f"{name_2} better: {second_better_counter} ({second_better_counter/parsed_by_both_counter:.2%}) - Mean IoU: {np.mean(second_better_ious):.4f}")
    print(f"Same IoU: {same_iou_counter} ({same_iou_counter/parsed_by_both_counter:.2%})")
    print(f"Mean IoU for {name_1} when explanations differ: {np.mean(first_ious_when_diff):.4f}")
    print(f"Mean IoU for {name_2} when explanations differ: {np.mean(second_ious_when_diff):.4f}")
    print(f"Mean IoU difference when {name_1} is better: {np.mean(first_better_diff_ious):.4f}")
    print(f"Mean IoU difference when {name_2} is better: {np.mean(second_better_diff_ious):.4f}")
    print("******************************************************")

def extract_aggregated_statistics(info):
    total_units, total_visited, total_expanded, total_estimated, total_time, total_iou = info
    iou_mean = round(np.mean(total_iou), 4) if len(total_iou) > 0 else None
    iou_std = round(np.std(total_iou), 4) if len(total_iou) > 0 else None
    visited_mean = round(np.mean(total_visited), 4) if len(total_visited) > 0 else None
    visited_std = round(np.std(total_visited), 4) if len(total_visited) > 0 else None
    expanded_mean = round(np.mean(total_expanded), 4) if len(total_expanded) > 0 else None
    expanded_std = round(np.std(total_expanded), 4) if len(total_expanded) > 0 else None
    estimated_mean = round(np.mean(total_estimated), 4) if len(total_estimated) > 0 else None
    estimated_std = round(np.std(total_estimated), 4) if len(total_estimated) > 0 else None
    time_mean = round(np.mean(total_time), 4) if len(total_time) > 0 else None
    time_std = round(np.std(total_time), 4) if len(total_time) > 0 else None
    return {
        'iou_mean': iou_mean,
        'iou_std': iou_std,
        'visited_mean': visited_mean,
        'visited_std': visited_std,
        'expanded_mean': expanded_mean,
        'expanded_std': expanded_std,
        'estimated_mean': estimated_mean,
        'estimated_std': estimated_std,
        'time_mean': time_mean,
        'time_std': time_std,
        'total_units': len(total_units) 
    }

def main(argv):
    # Set seed
    utils.set_seed(FLAGS.seed)

    features = list(FLAGS.features)
     # Parameters
    cfg = settings.Settings(
        model= FLAGS.model, 
        pretrained= FLAGS.pretrained, 
        dataset=FLAGS.dataset,
        num_clusters=FLAGS.num_clusters,
        beam_limit=FLAGS.beam_limit,
        heuristic=FLAGS.heuristic,
        length=FLAGS.length,
        layer=FLAGS.layer,
        device=FLAGS.device,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results,
        configuration_name=FLAGS.name_configuration,
        granularity=FLAGS.granularity,
        additional_concepts=FLAGS.add_concepts,
        experiment_name=FLAGS.name_experiment,
        additional_dataset=FLAGS.additional_dataset,
        unification_files=FLAGS.unification_files,
    )
    mask_shape = cfg.get_mask_shape()
    configuration_name = cfg.configuration_name 
        
    # Load Model
    model = utils.load_model(cfg)

    # Load Activations
    print(f"Loading activations for {cfg.model} model on {cfg.dataset} dataset - Layer:{cfg.layer}")
    layer_name= cfg.layer
    activation_dir = cfg.get_model_activation_path()
    layer_activations = model.get_layer_activations(layer_name, activation_dir)
    print(f"Activation Loaded")
    
    # Select Units
    selected_units = utils.get_selected_units(layer_activations, units=FLAGS.units, random_units=FLAGS.random_units)
    selected_units = selected_units[FLAGS.starting_unit:FLAGS.ending_unit]
    for segmentor_name in list(FLAGS.segmentors):

        # Set seed
        utils.set_seed(FLAGS.seed)
        
        concepts = cfg.get_concepts(FLAGS.concepts_type, FLAGS.additional_dataset)


        def get_explanations(cfg, beam_variant, neighbors_type, features):
            experiment_config = cfg.get_experiment_config()
            segmentor_experiment_config = experiment_config.copy()
            experiment_name = segmentor_utils.get_experiment_name("human", segmentor_experiment_config, concepts)
            results_dir = utils.get_result_dir(
            cfg.get_root_results(), "beam_optimal", configuration_name, cfg.dataset, segmentor_name, mask_shape, experiment_name, cfg.model, cfg.granularity, cfg.additional_dataset)
            if cfg.dataset == 'nli':
                results_dir = results_dir + f"/{beam_variant}/{neighbors_type}/{features}/"            
            
            h_config = segmentor_experiment_config.copy()
            h_config['results_dir'] = results_dir
            h_compo_config = cfg.get_explanation_config() 
            h_compo_config['quantile'] = FLAGS.quantile
            h_compo_config['block_type_3'] = FLAGS.block_type_3 if beam_variant == 'compound' else True
            h_compo_config['num_clusters'] = cfg.num_clusters
            h_exp = utils.load_compositional_explanations(
                beam_variant, layer_activations, selected_units, 
                h_config, h_compo_config, verbose=False)

            return h_exp

        print(f"Getting explanations for Incremental beam variant")
        incremental_compo_exp = get_explanations(cfg, beam_variant='baseline', neighbors_type='baseline', features=features)
        incremental_info = extract_info(incremental_compo_exp)
        incremental_stats = extract_aggregated_statistics(incremental_info)


        print(f"Getting explanations for non-incrementtal beam variant")
        our_compo_exp = get_explanations(cfg, beam_variant='compound', neighbors_type='baseline', features=features)
        our_info = extract_info(our_compo_exp)
        our_stats = extract_aggregated_statistics(our_info)
        
        # Print comparison
        for key in incremental_stats.keys():
            print(f"{key}: Incremental: {incremental_stats[key]} - Our: {our_stats[key]}")

        pairwise_compare("Incremental", incremental_compo_exp, "Our", our_compo_exp)
        



if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
