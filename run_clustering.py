"""Script to run the clustering algorithm for compositional explanations.
"""

import torch
import absl.flags
import absl.app


from compositional import mask_utils, metrics
from compositional import utils
from compositional import formula as F
from src import utils
from src import settings
from utils import common_flags, dataset_utils # import all user flags
from utils import segmentor_utils

# Flags specific to this script
absl.flags.DEFINE_string("output_path", "output/explanations", "output path")

absl.flags.DEFINE_string("attribution_method", "deconvolution", "attribution method to use [deconvolution, guided_backprop, integrated_gradients, gradient, gradient_shap]")
absl.flags.DEFINE_string("root_attributions", "data/cache/attributions", "root directory for attributions")
absl.flags.DEFINE_string("interpolation_mode", "bilinear", "interpolation mode to use for upsampling activations [bilinear, nearest, vanilla_influence, filtered_nearest]")

absl.flags.DEFINE_boolean("counter_variant", False, "Whether to use the counter variant of the heuristic")
absl.flags.DEFINE_float("diff_threshold", 0.1, "threshold for the admissibility constraint based on the diff between IoU and counter IoU")
absl.flags.DEFINE_string("neighbors_type", None, "Type of neighbors to use for the beam search [old, baseline, none]")
absl.flags.DEFINE_list("features", ['tokens', 'tags', 'overlap'], "Features to use for the beam search, separated by comma. Options: concepts, neighbors")
FLAGS = absl.flags.FLAGS


def main(argv):
    if FLAGS.first_n_interpretable_units is not None and (FLAGS.random_units > 0 or FLAGS.starting_unit != 0 or FLAGS.ending_unit != -1):
        raise ValueError("Cannot specify first_n_interpretable_units together with random_units, starting_unit, or ending_unit. Please choose one method for selecting units.")
    if FLAGS.random_units > 0 and (FLAGS.starting_unit != 0 or FLAGS.ending_unit != -1):
        raise ValueError("Cannot specify random_units together with starting_unit or ending_unit. Please choose one method for selecting units.")
    # Set seed
    utils.set_seed(FLAGS.seed)
    beam_variant = FLAGS.beam_variant
    quantile = FLAGS.quantile
    neighbors_type = FLAGS.neighbors_type
    if neighbors_type == 'none':
        neighbors_type = None
    if beam_variant == 'old' and neighbors_type != 'old':
        raise Warning("The old beam variant is designed to work with the old neighbors type. Consider using the new beam variant for better results with the new neighbors type.")

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
        attribution_method=FLAGS.attribution_method,
        additional_concepts=FLAGS.add_concepts,
        experiment_name=FLAGS.name_experiment,
        additional_dataset=FLAGS.additional_dataset,
        unification_files=FLAGS.unification_files,
    )
    mask_shape = cfg.get_mask_shape()
    configuration_name = cfg.configuration_name 
    experiment_config = cfg.get_experiment_config()
    
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
    selected_units = sorted(selected_units)

    #selected_units = [0, 6, 8, 15, 16, 70, 71, 89, 98]
    #selected_units = selected_units[:10]
    #selected_units = [209,244,327, 362, 366, 394, 444, 453, 483]
    #selected_units = [209, 394, 444]
    #selected_units = [0, 22]
    #selected_units = [0, 22,  151, 454] 

    #selected_units =  [15, 39, 99, 133, 473, 870, 941] # Mu

    #selected_units = [22]

    #selected_units = [8] #[6]

    #selected_units = [106]

    #selected_units = [91, 92, 394, 225, 285, 30, 39, 41, 45, 56, 70, 87, 94, 100, 109, 112, 117, 141, 153, 172, 184, 234, 239, 263, 280, 283, 323, 362]

    
    for segmentor_name in list(FLAGS.segmentors):
        # Set seed
        utils.set_seed(FLAGS.seed)
        
        segmentor_experiment_config = experiment_config.copy()
        concepts = cfg.get_concepts(FLAGS.concepts_type, FLAGS.additional_dataset)

        experiment_name = segmentor_utils.get_experiment_name(segmentor_name, segmentor_experiment_config, concepts)
        if cfg.dataset == 'nli':
            print(f"Getting features for {segmentor_name} - {experiment_name} with neighbors type: {neighbors_type}")
            masks, concept_vocab, constraints = model.get_features(neighbors_type, features=features)
            masks = torch.from_numpy(masks)
            tot_labels = len(concept_vocab['itos'].keys())
            segmentor_labels = list(range(tot_labels))
            for i, label in enumerate(concept_vocab['itos'].keys()):
                segmentor_labels[i] = concept_vocab['itos'][label]
            del concept_vocab
            info_dir = segmentor_utils.get_dir_info(
                cfg.get_root_segmentations(), configuration_name, cfg.dataset, segmentor_name, experiment_name)
            if beam_variant == 'compound':
                beam_variant_dir = 'baseline'
            else:
                beam_variant_dir = beam_variant
            info_dir = info_dir + f"/{beam_variant_dir}/{neighbors_type}/{features}/"
            disjoint_info = dataset_utils.compute_disjoint_vector_info(info_dir, masks, segmentor_labels)
            masks = masks.transpose(0,1)  # (num_concepts, num_samples)
            dataset = model.data_loader.dataset
            
        else:
            constraints = None
            dataset = None
            if FLAGS.preload_masks:
                masks, segmentor_labels = segmentor_utils.get_segmentor_outputs(segmentor_name, segmentor_experiment_config, concepts)
            else:
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
        print("Number of concepts: ", len(masks))
        print("Computing Compositional Explanations")
        results_dir = utils.get_result_dir(
            cfg.get_root_results(), cfg.heuristic, configuration_name, cfg.dataset, segmentor_name, mask_shape, experiment_name, cfg.model, cfg.granularity, cfg.additional_dataset)
        if cfg.dataset == 'nli':
            results_dir = results_dir + f"/{beam_variant}/{neighbors_type}/{features}/"
        segmentor_experiment_config['segmentor_name'] = segmentor_name
        segmentor_experiment_config['results_dir'] = results_dir
        segmentor_experiment_config['segmentor_labels'] = segmentor_labels
        config_compositional = cfg.get_explanation_config() 
        config_compositional['counter_variant'] = FLAGS.counter_variant
        config_compositional['diff_threshold'] = FLAGS.diff_threshold
        compo_exp = utils.compute_compositional_explanations(
            model, masks, masks_info, disjoint_info, layer_activations, selected_units, 
            segmentor_experiment_config, config_compositional,
            beam_variant=beam_variant,
            constraints=constraints, 
            dataset=dataset, quantile=quantile, 
            first_n_interpretable_units=FLAGS.first_n_interpretable_units,
            verbose=FLAGS.verbose)

        metrics_summary = metrics.compute_metrics_summary(compo_exp)
        print(f"Metrics Summary for {segmentor_name} - {experiment_name} with neighbors type: {neighbors_type}")
        for metric_name, metric_value in metrics_summary.items():
            print(f"{metric_name}: {round(metric_value, 3)}")
        exit()
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
        # del masks


if __name__ == "__main__":
    with torch.no_grad():
        absl.flags.mark_flags_as_required(["beam_limit", "beam_variant", "neighbors_type"])
        absl.app.run(main)
