"""Script to run the clustering algorithm for compositional explanations.
"""

from collections import defaultdict
import csv
import os

import torch
import absl.flags
import absl.app
from tqdm import tqdm
import numpy as np

from compositional import mask_utils
from compositional import activation_utils
from compositional import formula as F
from src import utils
from src import settings
from utils import common_flags # import all user flags
from utils import metric_utils
from utils import segmentor_utils

absl.flags.DEFINE_string("output_path", "output/metrics", "output path")
FLAGS = absl.flags.FLAGS


def main(argv):
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
        additional_dataset=FLAGS.additional_dataset,
        experiment_name=FLAGS.name_experiment,
    )
    mask_shape = cfg.get_mask_shape()
    configuration_name = cfg.configuration_name 
    experiment_config = cfg.get_experiment_config()

    # Load Activations
    print(f"Loading activations for {cfg.model} model on {cfg.dataset} dataset - Layer:{cfg.layer}")
    layer_activations = utils.get_layer_activations(cfg)
    print(f"Activation Loaded")

    # Select Units
    selected_units = utils.get_selected_units(layer_activations, units=FLAGS.units, random_units=FLAGS.random_units)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # Build dictionary to store results
    for segmentor_name in list(FLAGS.segmentors):
        for index_cluster in range(FLAGS.num_clusters):
            results[segmentor_name][str(index_cluster)] =  { 
                'iou': [],
                'activation_coverage': [], 'label_coverage': [], 'samples_coverage': [],
                'explanation_coverage': [],
                
                }
    
    for i, segmentor_name in enumerate(list(FLAGS.segmentors)):    
             # Set seed
        generator = utils.set_seed(FLAGS.seed)
        segmentor_experiment_config = experiment_config.copy()
        # Load Masks, Segmentor Labels and Experiment Name
        print("Loading Masks geenerated by ", segmentor_name)
        concepts = cfg.get_concepts(FLAGS.concepts_type, FLAGS.additional_dataset)
        masks, segmentor_labels, experiment_name = segmentor_utils.get_segmentor_outputs(segmentor_name, segmentor_experiment_config, concepts)
        
        print(f"Masks Loaded for {segmentor_name} - {experiment_name}")

        # Results Info
        results_dir = utils.get_result_dir(
            cfg.get_root_results(), configuration_name, cfg.dataset, segmentor_name, mask_shape, experiment_name, cfg.model, cfg.granularity, cfg.additional_dataset)

        segmentor_experiment_config['segmentor_name'] = segmentor_name
        segmentor_experiment_config['results_dir'] = results_dir
        segmentor_experiment_config['segmentor_labels'] = segmentor_labels
        config_compositional = cfg.get_explanation_config() 

        seg_b_expl = utils.collect_compositional_explanations(
            layer_activations, selected_units, 
            segmentor_experiment_config, config_compositional)
        
        for unit in tqdm(
                selected_units, desc="Computing Scores"
            ):
                unit_activations = layer_activations[:,unit,:,:]

                # Loop over all the activation ranges
                for cluster_index in range(FLAGS.num_clusters):
                    activation_range = seg_b_expl[unit][cluster_index][3]
                    
                    # Get results for segmentor
                    seg_best_label = seg_b_expl[unit][cluster_index][0][0]
                    seg_best_iou = seg_b_expl[unit][cluster_index][1]
                    results[segmentor_name][str(cluster_index)]['iou'].append(seg_best_iou)
                    label_mask = mask_utils.get_formula_mask(seg_best_label, masks).to(
                    cfg.device)
                    neuron_mask = activation_utils.compute_bitmaps(
                            unit_activations,
                            activation_range,
                            mask_shape=mask_shape,
                        )
                    neuron_mask = neuron_mask.to(cfg.device)
                    metrics_dict = metric_utils.compute_scores(neuron_mask, label_mask)
                    for metric in metrics_dict.keys():
                        value = metrics_dict[metric]
                        results[segmentor_name][str(cluster_index)][metric].append(value)
        del masks
        # Print Results
        for index_cluster in range(FLAGS.num_clusters):
            print(f"Segmentor: {segmentor_name} Cluster: {index_cluster}")
            for metric in results[segmentor_name][str(index_cluster)].keys():
                mean_metric = np.mean(results[segmentor_name][str(index_cluster)][metric])
                std_dev_metric = np.std(results[segmentor_name][str(index_cluster)][metric])
                print(f"{metric} Mean: {mean_metric:.3f} Std Dev: {std_dev_metric:.3f}")
            print("\n")

    # Print to file
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)
    file_name_cvs = f'{FLAGS.output_path}/results_{experiment_name}_{cfg.model}_{cfg.dataset}_{list(FLAGS.segmentors)}_metrisc.csv'
    with open(file_name_cvs, 'w', newline='') as csvfile:
        fieldnames = ['Segmentor', 'Cluster',  'iou mean', 'iou std',
                        'activation_coverage mean', 'activation_coverage std dev', 'label_coverage mean', 'label_coverage std dev',
                        'samples_coverage mean', 'samples_coverage std dev', 'explanation_coverage mean', 'explanation_coverage std dev',
                       ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index_cluster in range(FLAGS.num_clusters):
            for segmentor_name in results.keys():
                data_to_write = {
                    'Segmentor': segmentor_name,
                    'Cluster': index_cluster,
                    'iou mean': round(np.mean(results[segmentor_name][str(index_cluster)]['iou']),3),
                    'iou std': round(np.std(results[segmentor_name][str(index_cluster)]['iou']),3),
                    'activation_coverage mean': round(np.mean(results[segmentor_name][str(index_cluster)]['activation_coverage']),3),
                    'activation_coverage std dev': round(np.std(results[segmentor_name][str(index_cluster)]['activation_coverage']),3),
                    'label_coverage mean': round(np.mean(results[segmentor_name][str(index_cluster)]['label_coverage']),3),
                    'label_coverage std dev': round(np.std(results[segmentor_name][str(index_cluster)]['label_coverage']),3),
                    'samples_coverage mean': round(np.mean(results[segmentor_name][str(index_cluster)]['samples_coverage']),3),
                    'samples_coverage std dev': round(np.std(results[segmentor_name][str(index_cluster)]['samples_coverage']),3),
                    'explanation_coverage mean': round(np.mean(results[segmentor_name][str(index_cluster)]['explanation_coverage']),3),
                    'explanation_coverage std dev': round(np.std(results[segmentor_name][str(index_cluster)]['explanation_coverage']),3),
       
                }
                writer.writerow(data_to_write)
        print(f"Results saved in {file_name_cvs}")

        

if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
