"""Script to run the clustering algorithm for compositional explanations.
"""

from collections import Counter
import torch
import absl.flags
import absl.app
import math


from compositional import activation_utils, mask_utils, metrics
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

absl.flags.DEFINE_boolean("counter_variant", False, "Whether to use the counter variant of the heuristic")
FLAGS = absl.flags.FLAGS


def main(argv):
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
        configuration_name=FLAGS.name_configuration,
        granularity=FLAGS.granularity,
        attribution_method=FLAGS.attribution_method,
        additional_concepts=FLAGS.add_concepts,
        experiment_name=FLAGS.name_experiment,
        additional_dataset=FLAGS.additional_dataset,
        unification_files=FLAGS.unification_files,
    )

    # Untrained Config
    untrained_cfg = settings.Settings(
        model= FLAGS.model, 
        pretrained= 'untrained', 
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

    untrained_model = utils.load_model(untrained_cfg)
    untrained_activation_dir = untrained_cfg.get_model_activation_path()
    untrained_layer_activations = untrained_model.get_layer_activations(layer_name, untrained_activation_dir)
    del untrained_model 
    print(f"Activation Loaded")
    
    # Select Units
    selected_units = utils.get_selected_units(layer_activations, units=FLAGS.units, random_units=FLAGS.random_units)
    selected_units = selected_units[FLAGS.starting_unit:FLAGS.ending_unit]
    for segmentor_name in list(FLAGS.segmentors):
        # Set seed
        utils.set_seed(FLAGS.seed)
        
        segmentor_experiment_config = experiment_config.copy()
        concepts = cfg.get_concepts(FLAGS.concepts_type, FLAGS.additional_dataset)

        experiment_name = segmentor_utils.get_experiment_name(segmentor_name, segmentor_experiment_config, concepts)
        if cfg.dataset == 'nli':
            if FLAGS.beam_variant == 'old':
                neighbors_type = 'old'
            elif FLAGS.beam_variant == 'new':
                neighbors_type = 'new'
            else:
                neighbors_type = 'baseline'
            masks, concept_vocab, constraints = model.get_features(neighbors_type)
            masks = torch.from_numpy(masks).to(cfg.device)
            tot_labels = len(concept_vocab['itos'].keys())
            segmentor_labels = list(range(tot_labels))
            for i, label in enumerate(concept_vocab['itos'].keys()):
                segmentor_labels[i] = concept_vocab['itos'][label]
            del concept_vocab
            info_dir = segmentor_utils.get_dir_info(
                cfg.get_root_segmentations(), configuration_name, cfg.dataset, segmentor_name, experiment_name)
            masks = masks.transpose(0,1)  # (num_concepts, num_samples)
            
        else:
            if FLAGS.preload_masks:
                masks, segmentor_labels = segmentor_utils.get_segmentor_outputs(segmentor_name, segmentor_experiment_config, concepts)
            else:
                segmentations_dir = segmentor_utils.get_dir_segmentations(
                    segmentor_experiment_config['root_dir_segmentations'], configuration_name, segmentor_experiment_config['dataset_name'], segmentor_name, experiment_name)
            
                segmentor_labels = segmentor_utils.get_segmentor_concepts(segmentor_name, segmentor_experiment_config, concepts, segmentor_experiment_config['add_concepts'])
                masks = mask_utils.load_masks_path(segmentor_labels, segmentations_dir) 


        # Compute Compositional Explanations
        print("Computing Compositional Explanations")
        results_dir = utils.get_result_dir(
            cfg.get_root_results(), cfg.heuristic, configuration_name, cfg.dataset, segmentor_name, mask_shape, experiment_name, cfg.model, cfg.granularity, cfg.additional_dataset)
        segmentor_experiment_config['segmentor_name'] = segmentor_name
        segmentor_experiment_config['results_dir'] = results_dir
        segmentor_experiment_config['segmentor_labels'] = segmentor_labels
        config_compositional = cfg.get_explanation_config() 
        config_compositional['counter_variant'] = FLAGS.counter_variant

        compo_exp = utils.load_compositional_explanations(
                None, None, None, layer_activations, selected_units, 
                segmentor_experiment_config, config_compositional, verbose=False, beam_variant=FLAGS.beam_variant)
        
        # untrained
        results_dir_untrained = utils.get_result_dir(
            untrained_cfg.get_root_results(), untrained_cfg.heuristic, configuration_name, untrained_cfg.dataset, segmentor_name, mask_shape, experiment_name, untrained_cfg.model, untrained_cfg.granularity, untrained_cfg.additional_dataset)
        untrained_experiment_config = untrained_cfg.get_experiment_config()
        untrained_experiment_config['segmentor_name'] = segmentor_name
        untrained_experiment_config['results_dir'] = results_dir_untrained
        untrained_experiment_config['segmentor_labels'] = segmentor_labels
        untrained_config_compositional = untrained_cfg.get_explanation_config()
        untrained_config_compositional['counter_variant'] = FLAGS.counter_variant
        untrained_compo_exp = utils.load_compositional_explanations(
                None, None, None, untrained_layer_activations, selected_units, 
                untrained_experiment_config, untrained_config_compositional, verbose=False, beam_variant=FLAGS.beam_variant)

        no_expl = [ 0 for _ in range(FLAGS.num_clusters)]
        non_interp = [ 0 for _ in range(FLAGS.num_clusters)]
        pass_sanity_0 = [ 0 for _ in range(FLAGS.num_clusters)]
        pass_sanity_0_cluster = [ 0 for _ in range(FLAGS.num_clusters)]
        pass_sanity_1 = [ 0 for _ in range(FLAGS.num_clusters)]
        sanity_1_not_applicable = [ 0 for _ in range(FLAGS.num_clusters)]
        pass_sanity_2 = [ 0 for _ in range(FLAGS.num_clusters)]
        pass_sanity_3 = [ 0 for _ in range(FLAGS.num_clusters)]
        fails_sanity_3_and_is_interpretable = [ 0 for _ in range(FLAGS.num_clusters)]
        avg_iou = [ [] for _ in range(FLAGS.num_clusters)]
        avg_counter_iou = [ [] for _ in range(FLAGS.num_clusters)]

        # Sanity Check 0: Clustered Default explanations

        # We did not distinguish per cluster
        all_labels = []
        labels_per_cluster = [[] for _ in range(FLAGS.num_clusters)]
        for  unit, cluster_index, activation_range, best_label, string_label, best_iou, _, _, _, _  in untrained_compo_exp:
            all_labels.append(best_label)
            labels_per_cluster[cluster_index].append(best_label)
        # This is just for printing
        summary = [
            (F.get_formula_str(k, segmentor_labels), v)
            for k, v in Counter(all_labels).most_common()
            if v > 1
            ]
        print(f"Summary (filtering out labels with count=1): {summary}")
        summary = [
            (label, count) for label, count
            in Counter(all_labels).most_common()
            if count > 1
        ]

        summary_per_cluster = []
        for i in range(FLAGS.num_clusters):
            summary_cluster = [
                (F.get_formula_str(k, segmentor_labels), v)
                for k, v in Counter(labels_per_cluster[i]).most_common()
                if v > 1
            ]
            summary_per_cluster.append(summary_cluster)
            print(f"Summary Cluster {i} (filtering out labels with count=1): {summary_cluster}")



        # Compute default formulas
        top_k = math.ceil((len(selected_units)/100)*20)
        default_formulas = [
            label for label, count
            in Counter(all_labels).most_common(top_k)]
        
        print(f"Default formulas (top {top_k}): {default_formulas}")

        default_formulas_per_cluster = []
        for i in range(FLAGS.num_clusters):
            top_k_cluster = math.ceil((len(labels_per_cluster[i])/100)*20)
            default_formulas_cluster = [
                label for label, count
                in Counter(labels_per_cluster[i]).most_common(top_k_cluster)]
            default_formulas_per_cluster.append(default_formulas_cluster)

        for (expl, untrained_expl) in zip(compo_exp, untrained_compo_exp):
            unit, cluster_index, activation_range, best_label, string_label, best_iou, _, _, _, _ = expl
            untrained_unit, untrained_cluster_index, _, untrained_best_label, string_untrained_best_label, _, _, _, _, _ = untrained_expl
            assert unit == untrained_unit, f"Unit mismatch between trained and untrained explanations: {unit} vs {untrained_unit}"
            assert cluster_index == untrained_cluster_index, f"Cluster index mismatch between trained and untrained explanations: {cluster_index} vs {untrained_cluster_index}"
            if best_label is None:
                # Explanations not computed 
                continue
            if best_iou == 0:
                no_expl[cluster_index] += 1
            elif best_iou < 0.04:
                non_interp[cluster_index] += 1

            # Compute binary masks
            unit_activations = layer_activations[:, unit]
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
                
            # Compute Formula Mask
            formula_mask = mask_utils.parse_mask_by_type(mask_utils.get_formula_mask(
                best_label, masks)).to(bitmaps.device)
            
            # Additional metrics
            counter_iou = metrics.counter_iou(label_mask=formula_mask, bitmaps=bitmaps, counter_bitmaps=~bitmaps).item()
            avg_iou[cluster_index].append(best_iou)
            avg_counter_iou[cluster_index].append(counter_iou)
            

            # Sanity Check 0: Clustered Default explanations
            if best_label not in default_formulas:
                pass_sanity_0[cluster_index] += 1
            if best_label not in default_formulas_per_cluster[cluster_index]:
                pass_sanity_0_cluster[cluster_index] += 1

            # Sanity Check 1: Random Network
            # Problem: activation ranges are different
            #TODO Compute Explanation for random network
            if untrained_best_label != best_label:
                if untrained_best_label is None:
                    sanity_1_not_applicable[cluster_index] += 1
                else:
                    pass_sanity_1[cluster_index] += 1

            # Sanity Check 2: Random Activation
            bitmaps_perc = bitmaps.sum()/bitmaps.numel()
            random_bitmaps = torch.bernoulli(torch.full(bitmaps.shape, bitmaps_perc)).bool().to(cfg.device)
            #TODO Compute Explanation for random bitmaps

            # Sanity Check 3: Adversarial Bitmap
            adv_bitmaps = ~bitmaps

            adv_iou = metrics.iou(formula_mask, adv_bitmaps).item()
            if best_iou > adv_iou:
                pass_sanity_3[cluster_index] += 1
            else:
                if best_iou > 0.04:  # Only consider it a failure if the explanation is somewhat interpretable
                    fails_sanity_3_and_is_interpretable[cluster_index] += 1
                if cluster_index == 4:
                    print(f"Unit {unit} in cluster {cluster_index} failed sanity check 3: expl={string_label} best_iou={best_iou:.3f}, adv_iou={adv_iou:.3f}")

        # Summary
        print("Summary of Sanity Checks:")
        for i in range(FLAGS.num_clusters):
            denominator = len(avg_iou[i]) 
            if denominator == 0:
                raise ValueError(f"No explanations found for cluster {i}. Please consider changing the number of clusters or checking the activation ranges.")
            print(f"Cluster {i}: Total:{denominator} No Explanation: {no_expl[i]}, Non-Interpretable: {non_interp[i]}")
            print(f"Pass Sanity 0: {pass_sanity_0[i]}/{len(avg_iou[i])}, Pass Sanity 0 (cluster): {pass_sanity_0_cluster[i]}/{len(avg_iou[i])}, Pass Sanity 1: {pass_sanity_1[i]}/{len(avg_iou[i])}, Pass Sanity 3: {pass_sanity_3[i]}/{len(avg_iou[i])}, Sanity 1 Not Applicable: {sanity_1_not_applicable[i]}/{len(avg_iou[i])}, Fails Sanity 3 (interpretable): {fails_sanity_3_and_is_interpretable[i]}/{len(avg_iou[i])}")
            print(f"Avg IoU: {sum(avg_iou[i])/len(avg_iou[i]) if len(avg_iou[i]) > 0 else 0:.3f}, Avg Counter IoU: {sum(avg_counter_iou[i])/len(avg_counter_iou[i]) if len(avg_counter_iou[i]) > 0 else 0:.3f}")
            
            print("------------------------")
            


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
        absl.app.run(main)
