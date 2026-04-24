from src.segmentor_wrapper import Detectron2GroundTruth, BrodenGroundTruth, Mask2Former, Masqclip, CATSeg, OpenSeeD, SCAN, SED
from utils import wordnet_utils

def get_base_dir_segmentations(root, configuration, dataset_name, segmentor_name, used_classes):
    if 'wordnet_' in configuration:
        base = f"{root}/{used_classes}/{dataset_name}/{segmentor_name}/{configuration}"
    elif 'categorical' in configuration and segmentor_name not in ['human', 'mask2former']:
        base = f"{root}/{configuration}/{dataset_name}/{segmentor_name}"
    else:
        base = f"{root}/{used_classes}/{dataset_name}/{segmentor_name}"
    return base

def get_dir_info(root, configuration, dataset_name, segmentor_name, used_classes):
    base = get_base_dir_segmentations(root, configuration, dataset_name, segmentor_name, used_classes)
    info_dir = f"{base}/info"
    return info_dir


def get_dir_segmentations(root, configuration, dataset_name, segmentor_name, used_classes):
    if segmentor_name == 'human' or segmentor_name == 'mask2former':
        # masks do not change based on the configuration
        base = get_base_dir_segmentations(root, "", dataset_name, segmentor_name, used_classes)
    else:
        base = get_base_dir_segmentations(root, configuration, dataset_name, segmentor_name, used_classes)
    segmentations_dir = f"{base}/masks"
    return segmentations_dir

def get_segmentor(segmentor_name, config, use_classes):
    dataset_name = config['dataset_name']
    mask_shape = config['mask_shape']
    cfg = config['broden_cfg']
    if segmentor_name == 'human':
        if 'broden' in dataset_name:
            segmentor = BrodenGroundTruth(dataset_name, config=cfg)
        else:
            segmentor = Detectron2GroundTruth(dataset_name, mask_shape=mask_shape)
    elif segmentor_name == 'mask2former':
        segmentor = Mask2Former(dataset_name, mask_shape=mask_shape, batch_size=1, use_classes=use_classes)
    elif segmentor_name == 'masqclip':
        segmentor = Masqclip(dataset_name, mask_shape=mask_shape, batch_size=1, use_classes=use_classes)
    elif segmentor_name == 'cat_seg':
        # Supported only batch size 1
        segmentor = CATSeg(dataset_name, batch_size=1, use_classes=use_classes)
    elif segmentor_name == 'openseed':
        segmentor = OpenSeeD(dataset_name, mask_shape=mask_shape, batch_size=1, use_classes=use_classes)
    elif segmentor_name == 'scan':
        segmentor = SCAN(dataset_name, mask_shape=mask_shape, batch_size=1, use_classes=use_classes)
    elif segmentor_name == 'sed':
        segmentor = SED(dataset_name, mask_shape=mask_shape, batch_size=1, use_classes=use_classes)
    else:
        raise ValueError(f"Segmentor {segmentor_name} not supported")
    return segmentor

def get_experiment_name(segmentor_name, config, concepts):
    custom_name_experiment = config['custom_name']
    if segmentor_name == 'human' or segmentor_name == 'mask2former':
        experiment_name = 'dataset'
    elif custom_name_experiment != '':
        experiment_name = custom_name_experiment
    elif isinstance(concepts, list):
        experiment_name = 'list'
    else:
        experiment_name = concepts
    return experiment_name

def get_segmentor_concepts(segmentor_name, config, concepts, concepts_to_add):
    segmentor = get_segmentor(segmentor_name, config, concepts)
    configuration_name = config['configuration_name']
    dataset_name = config['dataset_name']

    unification_files = config['unification_files']
    # Add custom concepts
    if len(concepts_to_add) > 0 and segmentor_name not in ['human', 'mask2former']:
        segmentor.set_concept_labels(dataset_name, segmentor.concept_labels + concepts_to_add)
    if segmentor_name != 'human':
        unified_labels = wordnet_utils.get_multistep_wordnet_labels(configuration_name, dataset_name, unification_files, segmentor.concept_labels)
        segmentor.set_concept_labels(dataset_name, unified_labels) 
    labels = segmentor.concept_labels 
    return labels

def get_concepts_type(segmentor_name, configuration_name, concepts_type):
    if segmentor_name == 'human':
        use_concepts = "own"
    elif segmentor_name == 'mask2former':
        # Mask2Former is a closed vocabulary model
        use_concepts = "dataset"
    elif 'wordnet' in configuration_name:
        use_concepts = 'wordnet'
    else:
        use_concepts = concepts_type
    return use_concepts

def get_segmentor_masks(segmentor, config):
    segmentations_dir = config['segmentation_dir']
    mask_shape = config['mask_shape']
    dataset_name = config['dataset_name']
    if 'broden' in dataset_name:
        ignore = ['']
    elif 'cub200' in dataset_name:
        ignore = ['other', 'background']
    else:
        ignore = []
    step_size = config['step_size']
    masks = segmentor.get_masks(
                segmentations_dir, mask_shape=mask_shape, ignore=ignore, step_size=step_size)    
    return masks


def get_disjoint_info(segmentor_name, config, concepts, concepts_to_add):
    segmentor = get_segmentor(segmentor_name, config, concepts)
    configuration_name = config['configuration_name']
    dataset_name = config['dataset_name']
    root_dir_segmentations = config['root_dir_segmentations']
    unification_files = config['unification_files']
    # Add custom concepts
    experiment_name = get_experiment_name(segmentor_name, config, concepts)
    # Get masks and Unify masks if needed
    segmentations_dir = get_dir_segmentations(
            root_dir_segmentations, configuration_name, dataset_name, segmentor_name, experiment_name)
    config_seg = config.copy()
    config_seg['segmentation_dir'] = segmentations_dir

    dir_info = get_dir_info(
               root_dir_segmentations, configuration_name, dataset_name, segmentor_name, experiment_name)
    if segmentor_name != 'human':
        unified_labels = wordnet_utils.get_multistep_wordnet_labels(configuration_name, dataset_name, unification_files, segmentor.concept_labels)
        segmentor.set_concept_labels(dataset_name, unified_labels) 

    disjoint_info = segmentor.compute_disjoint_info(dir_info)

    del segmentor

    return disjoint_info


def get_config_masks(segmentor_name, config, concepts, concepts_to_add):
    segmentor = get_segmentor(segmentor_name, config, concepts)
    configuration_name = config['configuration_name']
    dataset_name = config['dataset_name']
    root_dir_segmentations = config['root_dir_segmentations']
    unification_files = config['unification_files']
    # Add custom concepts
    if len(concepts_to_add) > 0 and segmentor_name not in ['human', 'mask2former']:
        segmentor.set_concept_labels(dataset_name, segmentor.concept_labels + concepts_to_add)

    experiment_name = get_experiment_name(segmentor_name, config, concepts)
    # Get masks and Unify masks if needed
    segmentations_dir = get_dir_segmentations(
            root_dir_segmentations, configuration_name, dataset_name, segmentor_name, experiment_name)
    config_seg = config.copy()
    config_seg['segmentation_dir'] = segmentations_dir

    if segmentor_name != 'human':
        unified_labels = wordnet_utils.get_multistep_wordnet_labels(configuration_name, dataset_name, unification_files, segmentor.concept_labels)
        segmentor.set_concept_labels(dataset_name, unified_labels) 
    
    masks = get_segmentor_masks(segmentor, config_seg)
    if segmentor_name == 'human':
        masks = wordnet_utils.merge_multistep_wordnet_masks(configuration_name, dataset_name, unification_files, masks, segmentor) 

    labels = segmentor.concept_labels

    del segmentor

    return masks, labels

def get_categorical_masks(concepts, segmentor_name, experiment_config, concepts_to_add):
    masks = []
    segmentor_labels = []
    for category_labels in concepts:
        masks_category, segmentor_labels_category = get_config_masks(
            segmentor_name, experiment_config, category_labels, concepts_to_add)
        masks += masks_category
        segmentor_labels += segmentor_labels_category
    return masks, segmentor_labels

def get_segmentor_outputs(segmentor_name, config, concepts_to_use):
    # Add custom concepts
    if 'add_concepts' in config.keys():
        concepts_to_add = config['add_concepts']
    else:
        concepts_to_add = []
    
    configuration_name = config['configuration_name']
    concepts_type = get_concepts_type(segmentor_name, configuration_name, concepts_to_use)

    if 'categorical' in configuration_name and segmentor_name not in ['human', 'mask2former']:
        return get_categorical_masks(concepts_type, segmentor_name, config, concepts_to_add)
    else:
        return get_config_masks(segmentor_name, config, concepts_type, concepts_to_add)