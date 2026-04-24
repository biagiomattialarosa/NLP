"""Script to run the clustering algorithm for compositional explanations.
"""

import os

import torch
torch.set_num_threads(1)
import absl.flags
import absl.app
from torch import multiprocessing as mp


import pickle
from tqdm import tqdm
import numpy as np
from timeit import default_timer as timer

from compositional import mask_utils
from compositional import utils
from compositional import formula as F
from src import utils
from src import settings
from utils import common_flags # import all user flags
from utils import segmentor_utils
from compositional import activation_utils
from compositional import algorithms

# Flags specific to this script
absl.flags.DEFINE_string("output_path", "output/explanations", "output path")
absl.flags.DEFINE_integer("num_workers", 4, "number of workers for multiprocessing")
FLAGS = absl.flags.FLAGS
masks = None
disjoint_info = None
masks_info = None
activations = None

def mp_run_compositional_explanations(args):
    unit, dir_results, length, mask_shape, device, beam_limit, config_experiment, cluster_index, activation_range, masks_path, file_concepts_path = args
    
    if file_concepts_path is not None and os.path.exists(file_concepts_path):
        data = np.load(file_concepts_path, allow_pickle=True)
        if 'concept_quantities' not in data or 'neuron_quantities' not in data:
            data = pickle.load(open(file_concepts_path, "rb"))
            if isinstance(data, tuple):
                concept_quantities, neuron_quantities = data
            else:
                concept_quantities = data
                neuron_quantities = None
        else:
            concept_quantities = list(data['concept_quantities'])
            neuron_quantities = list(data['neuron_quantities'])
    else:
        concept_quantities = None
        neuron_quantities = None

    dir_current_results = (
        f"{dir_results}/{activation_range}"
    )
    if not os.path.exists(dir_current_results):
        os.makedirs(dir_current_results)
    file_algo_results = (
        f"{dir_current_results}/" + f"{length}.pickle"
    )
    # Compute binary masks
    bitmap = activation_utils.compute_bitmaps(
        activations[:,unit,:,:],
        activation_range,
        mask_shape=mask_shape,
    )
    if not os.path.exists(file_algo_results):
        # Compute binary masks
        #bitmaps = bitmaps.to(device)
        start_time = timer()
        results = algorithms.get_heuristic_scores(
            masks_path,
            bitmap,
            segmentations_info=masks_info,
            disjoint_info=disjoint_info,
            heuristic=FLAGS.heuristic,
            length=length,
            beam_size=beam_limit,
            max_size_mask=mask_shape[0]*mask_shape[1],
            mask_shape=mask_shape,
            device=device,
            concept_quantities=concept_quantities,
            neuron_quantities=neuron_quantities,
        )
        (
            best_label,
            best_iou,
            visited,
            expanded,
            estimated,
        ) = results
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
    #clusters_results.append((cluster_index, activation_range, (best_label, string_label, best_iou, visited, expanded)))

    return unit, (cluster_index, activation_range, (best_label, string_label, best_iou, visited, expanded, estimated, time_taken))

def mp_compute_compositional_explanations(units, config_experiment, config_compositional, masks_path):
    num_clusters = config_compositional['num_clusters']
    mask_shape = config_experiment['mask_shape']
    length = config_compositional['length']
    layer_name = config_compositional['layer_name']
    device = config_experiment['device']
    results_dir = config_experiment['results_dir']
    beam_limit = config_compositional['beam_limit']
    heuristic = config_compositional['heuristic']
    if heuristic == 'beam_optimal' or heuristic == 'optimal':
        optimal_info_dir = config_experiment['optimal_info']


    unit_info = []
    for unit in tqdm(units, desc="Computing activation ranges"):
        unit_activations = activations[:,unit,:,:]
        activation_ranges = activation_utils.compute_activation_ranges(
            unit_activations, num_clusters)
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
                # Compute binary masks
                bitmap = activation_utils.compute_bitmaps(
                    unit_activations,
                    activation_range,
                    mask_shape=mask_shape,
                )
                bitmap = bitmap.to(FLAGS.device)
                if heuristic == 'beam_optimal' or heuristic == 'optimal':
                    # Cache concept quantitis
                    # We perform this operation here to exploit GPU when available and make the whole parallization faster
                    file_concepts_dir = f"{optimal_info_dir}/{layer_name}/{unit}"
                    file_concepts_path = f"{file_concepts_dir}/concept_quantities_range_{activation_range}.npz"
                    if not os.path.exists(file_concepts_dir):
                        os.makedirs(file_concepts_dir)
                    if not os.path.exists(file_concepts_path):
                        bitmap = bitmap.to(device)
                        seg_quantities = masks_info[2]
                        _, _, concept_quantities = algorithms.get_optimal_heuristic_info(masks_path, bitmap, seg_quantities)
                        common_elements, unique_elements, uncoverable_elements = seg_quantities
                        neuron_quantities = algorithms.get_neuron_quantities(
                            bitmap, common_elements, unique_elements, uncoverable_elements
                        )
                        np.savez_compressed(file_concepts_path, concept_quantities=concept_quantities, neuron_quantities=neuron_quantities)
                        bitmap = bitmap.to('cpu')
                        del bitmap
                unit_info.append((cluster_index, activation_range, None))
    mp_args = ((unit,  f"{results_dir}/{heuristic}/length_{length}/"
                + f"{layer_name}/{unit}", length, mask_shape, device, beam_limit, config_experiment, cluster_index, activation_range, masks_path, file_concept_quantities) for unit, (cluster_index, activation_range, file_concept_quantities) in zip(units, unit_info))
    results = []
    with mp.Pool(FLAGS.num_workers) as p, tqdm(
            total=len(units), desc="Tallying units"
        ) as pbar:
        for unit, result in p.imap_unordered(
            mp_run_compositional_explanations, mp_args
        ):
            (cluster_index, activation_range, (best_label, string_label, best_iou, visited, expanded, estimated, time_taken)) = result
            print(
                f"Unit: {unit} - "
                + f"Cluster: {cluster_index} - "
                + f"Best Label: {string_label} - "
                + f"Number Labels: {best_label} - "
                + f"Best IoU: {round(best_iou,4)} - "
                + f"Visited: {visited} - "
                + f"Expanded: {expanded} - "
                + f"Estimated: {estimated} - "
                + f"Time: {time_taken:.2f} seconds"
            )
            results.append((unit, cluster_index, activation_range, best_label, string_label, best_iou, visited, expanded, estimated, time_taken))
            pbar.update(1)

    return results


def main(argv):
    global masks
    global disjoint_info
    global masks_info
    global activations

    if FLAGS.num_clusters < 1:
        raise ValueError("num_clusters must be greater than 0")
    # Set seed
    utils.set_seed(FLAGS.seed)

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
        root_optimal_info=FLAGS.root_optimal_info,
        configuration_name=FLAGS.name_configuration,
        granularity=FLAGS.granularity,
        additional_concepts=FLAGS.add_concepts,
        experiment_name=FLAGS.name_experiment,
        additional_dataset=FLAGS.additional_dataset,
        unification_files=FLAGS.unification_files,
    )
    mask_shape = cfg.get_mask_shape()
    configuration_name = cfg.configuration_name 
    experiment_config = cfg.get_experiment_config()
    
    # Load Activations
    print(f"Loading activations for {cfg.model} model on {cfg.dataset} dataset - Layer:{cfg.layer}")
    layer_activations = utils.get_layer_activations(cfg)
    activations = layer_activations
    print(f"Activation Loaded")
    
    # Select Units
    selected_units = utils.get_selected_units(layer_activations, units=FLAGS.units, random_units=FLAGS.random_units)
    selected_units = selected_units[FLAGS.starting_unit:FLAGS.ending_unit]
    #selected_units = [265]
    segmentor_name = 'human'
    # Set seed
    generator = utils.set_seed(FLAGS.seed)
    
    segmentor_experiment_config = experiment_config.copy()
    # Load Masks, Segmentor Labels and Experiment Name
    print("Loading Masks generated by ", segmentor_name)
    concepts = cfg.get_concepts(FLAGS.concepts_type, FLAGS.additional_dataset)
    disjoint_info = segmentor_utils.get_disjoint_info(
        segmentor_name, segmentor_experiment_config, concepts, segmentor_experiment_config['add_concepts'])
    print(f"Disjoint Information Loaded")
    experiment_name = segmentor_utils.get_experiment_name(segmentor_name, segmentor_experiment_config, concepts)

    segmentations_dir = segmentor_utils.get_dir_segmentations(
        segmentor_experiment_config['root_dir_segmentations'], configuration_name, segmentor_experiment_config['dataset_name'], segmentor_name, experiment_name)

    segmentor_labels = segmentor_utils.get_segmentor_concepts(segmentor_name, segmentor_experiment_config, concepts, segmentor_experiment_config['add_concepts'])
    masks = mask_utils.load_masks_path(segmentor_labels, segmentations_dir) 
    print(f"Masks Pre-Loaded for {segmentor_name} - {experiment_name}")

    # Get Info
    disjoint_info = segmentor_utils.get_disjoint_info(
        segmentor_name, segmentor_experiment_config, concepts, segmentor_experiment_config['add_concepts'])
    info_dir = segmentor_utils.get_dir_info(
            cfg.get_root_segmentations(), configuration_name, cfg.dataset, segmentor_name, experiment_name)
    if FLAGS.heuristic == 'optimal' or FLAGS.heuristic == 'beam_optimal':
        masks_info = mask_utils.get_masks_info(masks, info_directory=info_dir,
                                            mask_shape=mask_shape, device=cfg.device, quantities=True) 
    elif FLAGS.heuristic == 'mmesh':
        masks_info = mask_utils.get_masks_info(masks, info_directory=info_dir,
                                            mask_shape=mask_shape, device=cfg.device, areas=True, quantities=False, inscribed=True, bb_boxes=True) 
    elif FLAGS.heuristic == 'none':
        masks_info = None
    else:
        raise ValueError(f"Unknown heuristic {FLAGS.heuristic}")

    # Compute Compositional Explanations
    print("Computing Compositional Explanations")
    results_dir = utils.get_result_dir(
        cfg.get_root_results(), cfg.heuristic, configuration_name, cfg.dataset, segmentor_name, mask_shape, experiment_name, cfg.model, cfg.granularity, cfg.additional_dataset)
    segmentor_experiment_config['optimal_info'] = utils.get_optimal_info_dir(
        FLAGS.root_optimal_info, cfg.heuristic, configuration_name, cfg.dataset, segmentor_name, mask_shape, experiment_name, cfg.model, cfg.granularity, cfg.additional_dataset)
    
    segmentor_experiment_config['segmentor_name'] = segmentor_name
    segmentor_experiment_config['results_dir'] = results_dir
    segmentor_experiment_config['segmentor_labels'] = segmentor_labels
    config_compositional = cfg.get_explanation_config() 
    compo_exp = mp_compute_compositional_explanations(
        selected_units, 
        segmentor_experiment_config, config_compositional, masks)
    
    # # Store Results
    # print("Storing Results")
    # dir_output = FLAGS.output_path
    # if not os.path.exists(dir_output):
    #     os.makedirs(dir_output)
    # csv_file_compo = f"{dir_output}/{configuration_name}_{cfg.dataset}_{segmentor_name}_{cfg.model}_compositional_results_beam_{cfg.beam_limit}.csv"
    # with open(csv_file_compo, "w") as file:
    #     file.write("Unit,Cluster,Best Label,Best IoU,Visited\n")
    #     for result in compo_exp:
    #         (unit, cluster_index, _, _, string_label, best_iou, visited) = result
    #         file.write(f"{unit},{cluster_index},{string_label},{round(best_iou,3)},{visited}\n")
    # print("Compositional Explanations Computed and Saved in ", csv_file_compo)


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
