import absl.flags
import absl.app

from datasets import datasets_register  # register all new datasets

# user flags

absl.flags.DEFINE_string(
    "dataset", "nli", "subset to use. Values:[ade20k, pascal]"
)
absl.flags.DEFINE_string(
    "model",
    "bowman",
    "model to use. Values:[resnet18, resnet_cub200, alexnet, resnet50, densenet161]",
)
absl.flags.DEFINE_string(
    "pretrained",
    "nli",
    "whether to use pretrained weights. Values [cub, imagenet, places365, None]",
)
absl.flags.DEFINE_string(
    "layer",
    "mlp",
    "Layer to analyze.",
)
absl.flags.DEFINE_string(
    "name_experiment",
    "",
    "Custom name for the experiment",
)
absl.flags.DEFINE_string(
    "name_configuration",
    "",
    "Configuration to use. Values:[normal, wordnet_step1, wordnet_step2, wordnet_step3]",
)
absl.flags.DEFINE_string("device", "cuda", "device to use to store the model")

absl.flags.DEFINE_list("segmentors", ["human"], "segmentors to use")
absl.flags.DEFINE_string("concepts_type", "dataset", "type of concepts to use")
absl.flags.DEFINE_list("add_concepts", [], "additional concepts to use")
absl.flags.DEFINE_list("granularity", None, "granularity of the concepts")
absl.flags.DEFINE_string("additional_dataset", None, "additional dataset to use")

absl.flags.DEFINE_integer("length", 5, "length of explanations")
absl.flags.DEFINE_integer("num_clusters", 0, "number of clusters")
absl.flags.DEFINE_integer("beam_limit", None, "beam limit")
absl.flags.DEFINE_string("heuristic", "beam_optimal", "heuristic to use")
absl.flags.DEFINE_integer("random_units", 0, "number of units")
absl.flags.DEFINE_list("units", None, "Specific units to investigate")
absl.flags.DEFINE_list('unification_files', None, 'Files storing the unification')
absl.flags.DEFINE_integer("step_size", 80, "step size")
absl.flags.DEFINE_string(
    "root_models", "data/model", "root directory for models"
)
absl.flags.DEFINE_string(
    "root_datasets", "data/dataset", "root directory for datasets"
)
absl.flags.DEFINE_string(
    "root_segmentations",
    "data/cache/segmentations",
    "root directory for segmentations",
)
absl.flags.DEFINE_string(
    "root_activations",
    "data/cache/activations",
    "root directory for activations",
)
absl.flags.DEFINE_string(
    "root_results", "data/results", "root directory for results"
)
absl.flags.DEFINE_string(
    "root_optimal_info",
    "data/cache/optimal_info",
    "directory to store optimal info",
)
absl.flags.DEFINE_integer("seed", 0, "seed to use to set reproducibility")
absl.flags.DEFINE_boolean("preload_masks", True, "whether to load segmentation masks")
absl.flags.DEFINE_integer("starting_unit", 0, "unit to start from")
absl.flags.DEFINE_integer("ending_unit", -1, "unit to end at")
absl.flags.DEFINE_integer("first_n_interpretable_units", None, "number of first interpretable units to consider")

absl.flags.DEFINE_string("beam_variant", None, "beam variant to use [old, new]")
absl.flags.DEFINE_float("quantile", None, "quantile to use for activation range")
absl.flags.DEFINE_string("neighbors_type", None, "Type of neighbors to use for the beam search [old, baseline, none]")
absl.flags.DEFINE_list("features", ['tokens', 'tags', 'overlap'], "Features to use for the beam search, separated by comma. Options: concepts, neighbors")
absl.flags.DEFINE_boolean("block_type_3", True, "Whether to block type 3 explanations in the beam search")

absl.flags.DEFINE_boolean("verbose", False, "whether to print the top activated samples for each unit")

# Disable overly verbose logging of detectron2
import logging
logger = logging.getLogger("detectron2")
logger.setLevel(logging.WARNING)