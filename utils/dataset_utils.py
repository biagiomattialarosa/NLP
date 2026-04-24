import os
import json

import torch
import torchvision
import numpy as np
import pickle
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader
import detectron2.data.transforms as T

from compositional import segmentations
from compositional import utils
from src.mapper import DatasetMapper
from datasets import snli

PAD_IDX = 1

def get_cub_concepts_by(granularity=1, exclusive=False):
    bird_classes = ['other', 'background', "black footed albatross", "laysan albatross", "sooty albatross", "groove billed ani", "crested auklet", "least auklet", "parakeet auklet", "rhinoceros auklet", "brewer blackbird", "red winged blackbird", "rusty blackbird", "yellow headed blackbird", "bobolink", "indigo bunting", "lazuli bunting", "painted bunting", "cardinal", "spotted catbird", "gray catbird", "yellow breasted chat", "eastern towhee", "chuck will widow", "brandt cormorant", "red faced cormorant", "pelagic cormorant", "bronzed cowbird", "shiny cowbird", "brown creeper", "american crow", "fish crow", "black billed cuckoo", "mangrove cuckoo", "yellow billed cuckoo", "gray crowned rosy finch", "purple finch", "northern flicker", "acadian flycatcher", "great crested flycatcher", "least flycatcher", "olive sided flycatcher", "scissor tailed flycatcher", "vermilion flycatcher", "yellow bellied flycatcher", "frigatebird", "northern fulmar", "gadwall", "american goldfinch", "european goldfinch", "boat tailed grackle", "eared grebe", "horned grebe", "pied billed grebe", "western grebe", "blue grosbeak", "evening grosbeak", "pine grosbeak", "rose breasted grosbeak", "pigeon guillemot", "california gull", "glaucous winged gull", "heermann gull", "herring gull", "ivory gull", "ring billed gull", "slaty backed gull", "western gull", "anna hummingbird", "ruby throated hummingbird", "rufous hummingbird", "green violetear", "long tailed jaeger", "pomarine jaeger", "blue jay", "florida jay", "green jay", "dark eyed junco", "tropical kingbird", "gray kingbird", "belted kingfisher", "green kingfisher", "pied kingfisher", "ringed kingfisher", "white breasted kingfisher", "red legged kittiwake", "horned lark", "pacific loon", "mallard", "western meadowlark", "hooded merganser", "red breasted merganser", "mockingbird", "nighthawk", "clark nutcracker", "white breasted nuthatch", "baltimore oriole", "hooded oriole", "orchard oriole", "scott oriole", "ovenbird", "brown pelican", "white pelican", "western wood pewee", "sayornis", "american pipit", "whip poor will", "horned puffin", "common raven", "white necked raven", "american redstart", "geococcyx", "loggerhead shrike", "great grey shrike", "baird sparrow", "black throated sparrow", "brewer sparrow", "chipping sparrow", "clay colored sparrow", "house sparrow", "field sparrow", "fox sparrow", "grasshopper sparrow", "harris sparrow", "henslow sparrow", "le conte sparrow", "lincoln sparrow", "nelson sharp tailed sparrow", "savannah sparrow", "seaside sparrow", "song sparrow", "tree sparrow", "vesper sparrow", "white crowned sparrow", "white throated sparrow", "cape glossy starling", "bank swallow", "barn swallow", "cliff swallow", "tree swallow", "scarlet tanager", "summer tanager", "artic tern", "black tern", "caspian tern", "common tern", "elegant tern", "forsters tern", "least tern", "green tailed towhee", "brown thrasher", "sage thrasher", "black capped vireo", "blue headed vireo", "philadelphia vireo", "red eyed vireo", "warbling vireo", "white eyed vireo", "yellow throated vireo", "bay breasted warbler", "black and white warbler", "black throated blue warbler", "blue winged warbler", "canada warbler", "cape may warbler", "cerulean warbler", "chestnut sided warbler", "golden winged warbler", "hooded warbler", "kentucky warbler", "magnolia warbler", "mourning warbler", "myrtle warbler", "nashville warbler", "orange crowned warbler", "palm warbler", "pine warbler", "prairie warbler", "prothonotary warbler", "swainson warbler", "tennessee warbler", "wilson warbler", "worm eating warbler", "yellow warbler", "northern waterthrush", "louisiana waterthrush", "bohemian waxwing", "cedar waxwing", "american three toed woodpecker", "pileated woodpecker", "red bellied woodpecker", "red cockaded woodpecker", "red headed woodpecker", "downy woodpecker", "bewick wren", "cactus wren", "carolina wren", "house wren", "marsh wren", "rock wren", "winter wren", "common yellowthroat"]
    bird_parts = ['other', 'background', 'bird\'s wing', 'bird\'s upperparts', 'bird\'s underparts', 'bird\'s back', 'bird\'s tail', 'bird\'s head', 'bird\'s breast', 'bird\'s throat', 'bird\'s eye', 'bird\'s nape', 'bird\'s under tail', 'bird\'s forehead', 'bird\'s belly', 'bird\'s leg', 'bird\'s crown', 'bird\'s bill']
    bird_colors = ['other', 'background', 'blue bird', 'brown bird', 'iridescent bird', 'purple bird', 'rufous bird', 'gray bird', 'yellow bird', 'olive bird', 'green bird', 'pink bird', 'orange bird', 'red bird', 'black bird', 'white bird', 'buff bird', 'multi-colored bird']
    bird_shape = ['other', 'background', 'upright-perching_water bird', 'chicken-like-marsh bird', 'long-legged bird', 'duck', 'owl', 'gull', 'hummingbird', 'pigeon', 'tree-clinging bird', 'hawk bird', 'sandpiper bird', 'upland-ground bird', 'swallow bird', 'perching bird', 'generic bird']
    background_colors = ['other', 'background', 'blue background', 'brown background', 'iridescent background', 'purple background', 'rufous background', 'gray background', 'yellow background', 'olive background', 'green background', 'pink background', 'orange background', 'red background', 'black background', 'white background', 'buff background', 'multi-colored background']
    part_colors = ['other', 'background', 'blue bird\'s wing', 'brown bird\'s wing', 'iridescent bird\'s wing', 'purple bird\'s wing', 'rufous bird\'s wing', 'gray bird\'s wing', 'yellow bird\'s wing', 'olive bird\'s wing', 'green bird\'s wing', 'pink bird\'s wing', 'orange bird\'s wing', 'red bird\'s wing', 'black bird\'s wing', 'white bird\'s wing', 'buff bird\'s wing', 'multi-colored bird\'s wing',
                     'blue bird\'s upperparts', 'brown bird\'s upperparts', 'iridescent bird\'s upperparts', 'purple bird\'s upperparts', 'rufous bird\'s upperparts', 'gray bird\'s upperparts', 'yellow bird\'s upperparts', 'olive bird\'s upperparts', 'green bird\'s upperparts', 'pink bird\'s upperparts', 'orange bird\'s upperparts', 'red bird\'s upperparts', 'black bird\'s upperparts', 'white bird\'s upperparts', 'buff bird\'s upperparts', 'multi-colored bird\'s upperparts',
                     'blue bird\'s back', 'brown bird\'s back', 'iridescent bird\'s back', 'purple bird\'s back', 'rufous bird\'s back', 'gray bird\'s back', 'yellow bird\'s back', 'olive bird\'s back', 'green bird\'s back', 'pink bird\'s back', 'orange bird\'s back', 'red bird\'s back', 'black bird\'s back', 'white bird\'s back', 'buff bird\'s back', 'multi-colored bird\'s back',
                     'blue bird\'s upper tail', 'brown bird\'s upper tail', 'iridescent bird\'s upper tail', 'purple bird\'s upper tail', 'rufous bird\'s upper tail', 'gray bird\'s upper tail', 'yellow bird\'s upper tail', 'olive bird\'s upper tail', 'green bird\'s upper tail', 'pink bird\'s upper tail', 'orange bird\'s upper tail', 'red bird\'s upper tail', 'black bird\'s upper tail', 'white bird\'s upper tail', 'buff bird\'s upper tail', 'multi-colored bird\'s upper tail',
                     'blue bird\'s breast', 'brown bird\'s breast', 'iridescent bird\'s breast', 'purple bird\'s breast', 'rufous bird\'s breast', 'gray bird\'s breast', 'yellow bird\'s breast', 'olive bird\'s breast', 'green bird\'s breast', 'pink bird\'s breast', 'orange bird\'s breast', 'red bird\'s breast', 'black bird\'s breast', 'white bird\'s breast', 'buff bird\'s breast', 'multi-colored bird\'s breast',
                     'blue bird\'s throat', 'brown bird\'s throat', 'iridescent bird\'s throat', 'purple bird\'s throat', 'rufous bird\'s throat', 'gray bird\'s throat', 'yellow bird\'s throat', 'olive bird\'s throat', 'green bird\'s throat', 'pink bird\'s throat', 'orange bird\'s throat', 'red bird\'s throat', 'black bird\'s throat', 'white bird\'s throat', 'buff bird\'s throat', 'multi-colored bird\'s throat',
                     'blue bird\'s eye', 'brown bird\'s eye', 'iridescent bird\'s eye', 'purple bird\'s eye', 'rufous bird\'s eye', 'gray bird\'s eye', 'yellow bird\'s eye', 'olive bird\'s eye', 'green bird\'s eye', 'pink bird\'s eye', 'orange bird\'s eye', 'red bird\'s eye', 'black bird\'s eye', 'white bird\'s eye', 'buff bird\'s eye', 'multi-colored bird\'s eye',
                     'blue bird\'s forehead', 'brown bird\'s forehead', 'iridescent bird\'s forehead', 'purple bird\'s forehead', 'rufous bird\'s forehead', 'gray bird\'s forehead', 'yellow bird\'s forehead', 'olive bird\'s forehead', 'green bird\'s forehead', 'pink bird\'s forehead', 'orange bird\'s forehead', 'red bird\'s forehead', 'black bird\'s forehead', 'white bird\'s forehead', 'buff bird\'s forehead', 'multi-colored bird\'s forehead',
                     'blue bird\'s under tail', 'brown bird\'s under tail', 'iridescent bird\'s under tail', 'purple bird\'s under tail', 'rufous bird\'s under tail', 'gray bird\'s under tail', 'yellow bird\'s under tail', 'olive bird\'s under tail', 'green bird\'s under tail', 'pink bird\'s under tail', 'orange bird\'s under tail', 'red bird\'s under tail', 'black bird\'s under tail', 'white bird\'s under tail', 'buff bird\'s under tail', 'multi-colored bird\'s under tail',
                     'blue bird\'s nape', 'brown bird\'s nape', 'iridescent bird\'s nape', 'purple bird\'s nape', 'rufous bird\'s nape', 'gray bird\'s nape', 'yellow bird\'s nape', 'olive bird\'s nape', 'green bird\'s nape', 'pink bird\'s nape', 'orange bird\'s nape', 'red bird\'s nape', 'black bird\'s nape', 'white bird\'s nape', 'buff bird\'s nape', 'multi-colored bird\'s nape',
                     'blue bird\'s belly', 'brown bird\'s belly', 'iridescent bird\'s belly', 'purple bird\'s belly', 'rufous bird\'s belly', 'gray bird\'s belly', 'yellow bird\'s belly', 'olive bird\'s belly', 'green bird\'s belly', 'pink bird\'s belly', 'orange bird\'s belly', 'red bird\'s belly', 'black bird\'s belly', 'white bird\'s belly', 'buff bird\'s belly', 'multi-colored bird\'s belly',
                     'blue bird\'s leg', 'brown bird\'s leg', 'iridescent bird\'s leg', 'purple bird\'s leg', 'rufous bird\'s leg', 'gray bird\'s leg', 'yellow bird\'s leg', 'olive bird\'s leg', 'green bird\'s leg', 'pink bird\'s leg', 'orange bird\'s leg', 'red bird\'s leg', 'black bird\'s leg', 'white bird\'s leg', 'buff bird\'s leg', 'multi-colored bird\'s leg',
                     'blue bird\'s crown', 'brown bird\'s crown', 'iridescent bird\'s crown', 'purple bird\'s crown', 'rufous bird\'s crown', 'gray bird\'s crown', 'yellow bird\'s crown', 'olive bird\'s crown', 'green bird\'s crown', 'pink bird\'s crown', 'orange bird\'s crown', 'red bird\'s crown', 'black bird\'s crown', 'white bird\'s crown', 'buff bird\'s crown',  'multi-colored bird\'s crown',
                     'blue bird\'s bill', 'brown bird\'s bill', 'iridescent bird\'s bill', 'purple bird\'s bill', 'rufous bird\'s bill', 'gray bird\'s bill', 'yellow bird\'s bill', 'olive bird\'s bill', 'green bird\'s bill', 'pink bird\'s bill', 'orange bird\'s bill', 'red bird\'s bill', 'black bird\'s bill', 'white bird\'s bill', 'buff bird\'s bill', 'multi-colored bird\'s bill']
    part_shapes = ['other', 'background', 'curved bird\'s bill', 'dagger bird\'s bill', 'hooked bird\'s bill', 'needle bird\'s bill', 'hooked seabird\'s bill', 'spatulate bird\'s bill', 'cone bird\'s bill', 'specialized bird\'s bill', 'all-purpose bird\'s bill', 'forked bird\'s tail', 'rounded bird\'s tail', 'notched bird\'s tail', 'fan-shaped bird\'s tail', 'pointed bird\'s tail', 'squared bird\'s tail', 'rounded bird\'s wing', 'pointed bird\'s wing', 'broad bird\'s wing', 'tapered bird\'s wing', 'long bird\'s wing']
    part_patterns = ['other', 'background', 'solid bird\'s breast', 'spotted bird\'s breast', 'striped bird\'s breast', 'spotted bird\'s head', 'malar bird\'s head', 'crested bird\'s head', 'masked bird\'s head', 'unique bird\'s head', 'eyebrow bird\'s head', 'eyering bird\'s head', 'plain bird\'s head', 'eyeline bird\'s head', 'striped bird\'s head', 'capped bird\'s head', 'solid bird\'s back', 'spotted bird\'s back', 'striped bird\'s back', 'solid bird\'s wing', 'spotted bird\'s wing', 'striped bird\'s wing']

    if granularity == 0:
        concepts_to_use = [bird_shape, bird_classes, bird_colors+background_colors]
    elif granularity == 1:
        concepts_to_use = [bird_parts]
    elif granularity == 2:
        concepts_to_use = [part_colors, part_shapes, part_patterns]
    else:
        raise ValueError(f"Granularity {granularity} not supported")
    return concepts_to_use

def get_cub_concepts(granularity, additional_dataset=None):
    if not isinstance(granularity, list) or len(granularity) == 0:
        raise ValueError("Granularity must be a list with at least one element")
    else:
        concepts = []
        granularity = [int(g) for g in granularity]
        sorted_granularity = sorted(granularity)
        for index_gran, gran in enumerate(sorted_granularity):
            # We load the full base for the lowest granularity and then add concepts for the rest of the granularities
            exclusive = True if index_gran > 0 else False
            gran_concepts = get_cub_concepts_by(gran, exclusive)
            concepts.extend(gran_concepts)

        if additional_dataset:
            dataset_classes, _ = get_class_names('human', additional_dataset, 'dataset')
            concepts.append(dataset_classes)
    
    return concepts, None

def get_dataset_concepts(dataset_name, dataset):
    if dataset_name in DatasetCatalog:
        return MetadataCatalog.get(dataset_name).stuff_classes.copy()
    elif dataset_name == 'broden':
        return dataset.labels

def get_dataset(dataset_name):
    if dataset_name in DatasetCatalog:
        return get_detectron_dataset(dataset_name)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not suuported yet")

def get_detectron_dataset(dataset_name):
    dataset = DatasetCatalog.get(dataset_name)
    return dataset

def get_json_labels(dataset_name, model_name):
    dir_json = f"datasets/json/{model_name}/"
    if dataset_name == "ade20k_sem_seg_val":
        file_name = "ade150.json"
    elif dataset_name == "ade20k_full_sem_seg_freq_val_all":
        file_name = "ade847.json"
    elif dataset_name == "context_59_test_sem_seg":
        file_name = "pc59.json"
    elif dataset_name == "context_459_test_sem_seg":
        file_name = "pc459.json"
    elif dataset_name == "voc_2012_test_sem_seg":
        file_name = "voc20.json"
    elif dataset_name == "cityscapes_fine_sem_seg_val" and model_name == "openseed":
        file_name = "cityscapes.json"
    elif dataset_name == "coco_2017_test_stuff_all_sem_seg":
        file_name = "coco.json"
    else:
        file_name = None
    if file_name is None or not os.path.exists(f"{dir_json}{file_name}"):
        print(f"Json  for Dataset {dataset_name} not found. Loading labels from MetadataCatalog")
        labels = MetadataCatalog.get(dataset_name).stuff_classes.copy()
        used_classes = "dataset"
    else:
        labels = json.load(open(f"{dir_json}{file_name}"))
        used_classes = "own"

    if model_name == "masqclip":
        labels = labels + ["background"]
        used_classes = "own"
        print("Adding background class to labels")
    return labels, used_classes
    


def get_probing_transformations(dataset_name, img_size, mean, std):
    if dataset_name in DatasetCatalog.keys() or dataset_name == 'broden' or dataset_name == 'cub200':
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((img_size, img_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ]
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not suuported yet")

def get_class_names(model_name, dataset_name, use_classes):
        if isinstance(use_classes, list):
            class_names = use_classes
            used_classes = "custom"
        elif use_classes == "own":
            class_names, used_classes = get_json_labels(dataset_name, model_name)
        else:
            catalog = MetadataCatalog.get(dataset_name)
            if "stuff_classes" in catalog.__dict__:  
                class_names = catalog.stuff_classes.copy()
            else:
                class_names = catalog.thing_classes.copy()
            used_classes = "dataset"
        if use_classes == "mono":
            split_class_names = []
            for class_name in class_names:
                if ', ' in class_name:
                    split_class_names.append(class_name.split(', ')[0])
                else:
                    split_class_names.append(class_name)
            class_names = split_class_names
            used_classes = "mono"
        return class_names, used_classes

def pad_collate(batch, sort=True):
    src, src_feats, src_multifeats, src_len, idx = zip(*batch)
    idx = torch.tensor(idx)
    src_len = torch.tensor(src_len)
    src_pad = torch.nn.utils.rnn.pad_sequence(src, padding_value=PAD_IDX)
    # NOTE: part of speeches are padded with 0 - we don't actually care here
    src_feats_pad = torch.nn.utils.rnn.pad_sequence(src_feats, padding_value=-1)
    src_multifeats_pad = torch.nn.utils.rnn.pad_sequence(src_multifeats, padding_value=-1)

    if sort:
        src_len_srt, srt_idx = torch.sort(src_len, descending=True)
        src_pad_srt = src_pad[:, srt_idx]
        src_feats_pad_srt = src_feats_pad[:, srt_idx]
        src_multifeats_pad_srt = src_multifeats_pad[:, srt_idx]
        idx_srt = idx[srt_idx]
        return (
            src_pad_srt,
            src_feats_pad_srt,
            src_multifeats_pad_srt,
            src_len_srt,
            idx_srt,
        )
    return src_pad, src_feats_pad, src_multifeats_pad, src_len, idx

def get_data_loader(dataset_name, dataset, batch_size=1, transforms=[], seed=0):
    if dataset_name in DatasetCatalog:
        import logging
        logger = logging.getLogger("detectron2")
        logger.setLevel(logging.WARNING)
        data_loader = build_detection_test_loader(
            dataset, mapper=DatasetMapper(
                is_train=False,
                # We need to resize the image to the same size used for the probed model
                augmentations=transforms,
                image_format="RGB",
                
                ), batch_size=batch_size)
    elif dataset_name == 'broden':
        generator = utils.set_seed(seed)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            worker_init_fn=seed,
            generator=generator,
        )
    elif dataset_name == 'nli':
        generator = utils.set_seed(seed)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=32,
            collate_fn=lambda batch: pad_collate(batch, sort=False),
             worker_init_fn=seed,
            generator=generator,
        )
    return data_loader

def get_broden_data(dataset_name, cfg):
    # Load data
    dataset = segmentations.BrodenDataset(
        cfg.dir_datasets,
        resolution=cfg.get_img_size(),
        broden_version=1,
        transform_image=torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize(cfg.get_mask_shape()),
                 torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(
                #     cfg.get_image_mean(), cfg.get_image_stdev()
          #      ),
            ]
        ),
    )
    generator = utils.set_seed(0)
    segmentation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        worker_init_fn=utils.seed_worker,
        generator=generator,
    )
    return segmentation_loader, dataset.labels

def get_data_iter_fn(dataset_name):
    if dataset_name in DatasetCatalog.keys():
        return lambda x: x[0]['image']
    elif dataset_name == 'broden':
        return lambda x: x[0]
    elif dataset_name == 'nli':
        return lambda x: decompose_batch_nli(x)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported yet")

def decompose_batch_nli(batch):
    src, _, _, src_lengths, idx = batch
    with torch.no_grad():
        # Combine q/h pairs
        src_one = src.squeeze(2)
        src_one_comb = snli.pairs(src_one)
        src_lengths_comb = snli.pairs(src_lengths)

        s1 = src_one_comb[:, :, 0]
        s1len = src_lengths_comb[:, 0]

        s2 = src_one_comb[:, :, 1]
        s2len = src_lengths_comb[:, 1]

    return s1, s1len, s2, s2len, idx

def compute_disjoint_vector_info(info_dir, masks, concept_labels):
    # Specific for NLI
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    path_matrix = f"{info_dir}/disjoint_matrix.pt"
    if os.path.exists(path_matrix):
        disjoint_matrix = pickle.load(open(path_matrix, "rb"))
        assert disjoint_matrix.shape == (len(concept_labels), len(concept_labels)), f"Disjoint matrix shape {disjoint_matrix.shape} does not match expected shape {(len(concept_labels), len(concept_labels))}"
        return disjoint_matrix
    
    disjoint_matrix = np.zeros((len(concept_labels), len( concept_labels)), dtype=bool)
    for concept in range(len(concept_labels)):
        disjoint_matrix[concept, concept] = False
    for concept_1 in tqdm(range(len(concept_labels)), desc="Computing Disjoint Info"):
        for concept_2 in range(concept_1 + 1, len(concept_labels)):
            vec_1 = masks[:, concept_1]
            vec_2 = masks[:, concept_2]
            # Check if the two concepts are disjoint
            if torch.sum(vec_1 & vec_2) == 0:   
                disjoint_matrix[concept_1, concept_2] = True
                disjoint_matrix[concept_2, concept_1] = True
    with open(path_matrix, "wb") as f:
        pickle.dump(disjoint_matrix, f)
    return disjoint_matrix