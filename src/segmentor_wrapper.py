
from collections import defaultdict
import os
from tqdm import tqdm
import warnings
import pickle

import torch
import numpy as np
import scipy.sparse as sparse
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer

from src import config as config_collection
from utils import dataset_utils
from compositional import mask_utils

# # Register models
# from segmentors.openseed.BaseModel import BaseModel as OpenSeeDBaseModel
# import segmentors.masqclip.masq_tuning
# import segmentors.openseed
# import segmentors.cat_seg
# import segmentors.scan
# import segmentors.sed


class SegmentorWrapper:
    def __init__(self, dataset_name, *additional_args) -> None:
        self.data_loader = None
        self.concept_labels = None
        raise NotImplementedError

    def load_masks(self, segmentations_directory):
        """
        Loads the sparse masks from the given directory.
        Args:
            concept_names (list): list of concept names.
            segmentations_directory (str): directory where the masks are stored.
        Returns:
            List of sparse masks.
        """
        return mask_utils.load_sparse_masks(
            self.concept_labels, segmentations_directory
        )

    def extract_segmentations(self, data):
        raise NotImplementedError
    
    def parse_concept(self, segm_data, concept_value, mask_shape):
        num_categories_segmentations = segm_data.shape[1]
        concept_mask = torch.zeros(
            (
                segm_data.shape[0],
                segm_data.shape[2],
                segm_data.shape[3],
            ),
            dtype=bool,
        ).cuda()
        for index_segmentation in range(num_categories_segmentations):
            concept_mask = concept_mask | (
                segm_data[:, index_segmentation] == concept_value
            )
        concept_mask  = torch.nn.functional.interpolate(
                concept_mask.float().unsqueeze(0),
                size=mask_shape,
                mode="nearest",
            ).bool().squeeze(0)
        return concept_mask
    
    def fast_save_segmentation_masks(self, segmentation_dir, mask_shape, step_size=20, missing=None, regenerate=True):
        print("Using the fast implementation to compute the masks. It requires more RAM but it is faster. If you have memory issues, set the flag fast_impl of the save_segmentation_masks() function to False")
        tot_concepts = len(self.concept_labels)
        ranges = range(0, tot_concepts, step_size)
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)
        from utils import visual_utils

        # Collect segmentation masks for the whole dataset
        dataset_segmentations = []
        for data in tqdm(self.data_loader, desc="Computing Segmentations"):
            segmentations = self.extract_segmentations(data)
            dataset_segmentations.append(segmentations)
        
        # Compute the masks for the selected concepts
        for starting_index in tqdm(ranges):
            concepts = range(starting_index, starting_index + step_size)
            masks = defaultdict(lambda: [])
            # Remove missing from concepts if regenerate is False
            if not regenerate:
                concepts = [concept for concept in concepts if concept in missing]
            # Remove concepts that are out of range
            concepts = [concept for concept in concepts if concept < tot_concepts]
            # Compute the masks for the selected concepts
            for concept_index in concepts:
                concept_mask = []
                for segmentations in dataset_segmentations:
                    segmentations = segmentations.cuda()
                    concept_mask.append(self.parse_concept(segmentations, concept_index, mask_shape))
                concept_mask = torch.cat(concept_mask, 0)
                masks[concept_index].append(concept_mask.cpu())
        
            for concept_index in sorted(masks.keys()):
                if concept_index < len(self.concept_labels):
                    # Prepare for scipy sparse matrix format
                    masks[concept_index] = torch.cat(masks[concept_index], 0)
                    masks[concept_index] = torch.reshape(
                        masks[concept_index], (masks[concept_index].shape[0], -1)
                    )
                    masks[concept_index] = masks[concept_index].numpy()
                    with open(
                        f"{segmentation_dir}/"
                        + f"{self.concept_labels[concept_index]}.npz",
                        "wb",
                    ) as file:
                        sparse.save_npz(
                            file, sparse.csr_matrix(masks[concept_index])
                        )
                    del masks[concept_index]
            del masks
        return 

    def compute_disjoint_info(self, info_dir):
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)
        path_matrix = f"{info_dir}/disjoint_matrix.pt"
        if os.path.exists(path_matrix):
            disjoint_matrix = pickle.load(open(path_matrix, "rb"))
            return disjoint_matrix
        path_dict = f"{info_dir}/disjoint_info.pt"
        if os.path.exists(path_dict):
            disjoint_dict = pickle.load(open(path_dict, "rb"))
            print("Disjoint matrix not computed, but disjoint_dict already exists, converting it to matrix")
            disjoint_matrix = np.zeros((len(self.concept_labels), len(self.concept_labels)), dtype=bool)
            for label_i in disjoint_dict.keys():
                for label_j in disjoint_dict[label_i]:
                    if label_i != label_j:
                        disjoint_matrix[label_i, label_j] = True
                        disjoint_matrix[label_j, label_i] = True
            return disjoint_matrix
        print("Disjoint matrix not computed, computing it")
        # Collect segmentation masks for the whole dataset
        dataset_segmentations = []
        for data in tqdm(self.data_loader, desc="Computing Segmentations"):
            segmentations = self.extract_segmentations(data)
            dataset_segmentations.append(segmentations)

        # Create a dictionary to store the disjoint information
        disjoint_matrix = np.ones((len(self.concept_labels), len(self.concept_labels)), dtype=bool)
        for concept in range(len(self.concept_labels)):
            disjoint_matrix[concept, concept] = False
        for segmentations in tqdm(dataset_segmentations, desc="Computing Disjoint Info"):
            multiple_segmentations = (segmentations > 0).sum(axis=(1)) > 1
            index_multiple_segmentations = multiple_segmentations.nonzero()
            for (b, h, w) in index_multiple_segmentations:
                overlapping_concepts = segmentations[b, :,  h, w].unique()
                # Remove the concepts overlapping from the disjoint_dict
                for index_over, concept_in_overlap in enumerate(overlapping_concepts):
                    concept_in_overlap = concept_in_overlap.item()
                    for other_concept in overlapping_concepts[index_over + 1:]:
                        other_concept = other_concept.item()
                        disjoint_matrix[concept_in_overlap, other_concept] = False
                        disjoint_matrix[other_concept, concept_in_overlap] = False
        with open(path_matrix, "wb") as f:
            pickle.dump(disjoint_matrix, f)
        return disjoint_matrix

    def save_segmentation_masks(self, segmentation_dir, mask_shape, step_size=20, missing=None, regenerate=True, fast_impl= True):
        if fast_impl:
            self.fast_save_segmentation_masks(segmentation_dir, mask_shape, step_size, missing, regenerate)
            return
        tot_concepts = len(self.concept_labels)
        ranges = range(0, tot_concepts, step_size)
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)
        for starting_index in tqdm(ranges):
            concepts = range(starting_index, starting_index + step_size)
            masks = defaultdict(lambda: [])
            # Remove missing from concepts if regenerate is False
            if not regenerate:
                concepts = [concept for concept in concepts if concept in missing]
            # Remove concepts that are out of range
            concepts = [concept for concept in concepts if concept < tot_concepts]
            # Compute the masks for the selected concepts
            if len(concepts) > 0:
                for index_batch, data in enumerate(self.data_loader):
                    segmentations = self.extract_segmentations(data)
                    segmentations = segmentations.cuda()
  
                    for concept_index in concepts:
                        concept_mask = self.parse_concept(segmentations, concept_index, mask_shape)
                        masks[concept_index].append(concept_mask.cpu())
                for concept_index in sorted(masks.keys()):
                    if concept_index < len(self.concept_labels):
                        # Prepare for scipy sparse matrix format
                        masks[concept_index] = torch.cat(masks[concept_index], 0)
                        masks[concept_index] = torch.reshape(
                            masks[concept_index], (masks[concept_index].shape[0], -1)
                        )
                        masks[concept_index] = masks[concept_index].numpy()

                        with open(
                            f"{segmentation_dir}/"
                            + f"{self.concept_labels[concept_index]}.npz",
                            "wb",
                        ) as file:
                            sparse.save_npz(
                                file, sparse.csr_matrix(masks[concept_index])
                            )
                        del masks[concept_index]
            del masks

    def get_masks(self, masks_directory, mask_shape, ignore=None, step_size=20):
        """
        Returns the masks for the given dataloader and labels.
        Args:
            masks_directory (str): directory where the sparse masks are stored.
            dataloader (torch.utils.data.DataLoader): dataloader for the images.
            labels (list): list of labels.
        Returns:
            List of masks.
        """
        if not os.path.exists(masks_directory):
            os.makedirs(masks_directory)
        # If some file is missing, generate again the sparse masks
        missing = []
        for index, concept in enumerate(self.concept_labels):
            if os.path.exists(f"{masks_directory}/{concept}.npz"):
                continue
            else:
                missing.append(index)
        if len(missing) == 0 and len(os.listdir(masks_directory)) != len(self.concept_labels):
            warnings.warn(f"All masks are have been already generated in {masks_directory} but there is a mismatch in the number of masks in the directory and concept labels." +\
            f'If you are using categorical masks, this behavior is expected. Otherwise, please check if something changed in the experimental configuration')
        elif len(missing) > 0:
            print(f"Missing {len(missing)} masks in {masks_directory}")
            self.save_segmentation_masks(masks_directory, mask_shape, step_size, missing=missing, regenerate=False, fast_impl=False)
        masks = self.load_masks(masks_directory)
        for i in range(len(masks)):
            # Zero out the masks that need to be ignored
            if ignore is not None and self.concept_labels[i] in ignore:
                masks[i] = sparse.csr_matrix(torch.zeros_like(torch.from_numpy(masks[i].toarray())).numpy())
                #masks[i] = torch.zeros_like(torch.from_numpy(masks[i].toarray()))
            #masks[i] = torch.from_numpy(masks[i].toarray())
            
        return masks
    
class BrodenGroundTruth(SegmentorWrapper):
    def __init__(self, dataset_name, *, config) -> None:
        self.data_loader, self.concept_labels = dataset_utils.get_broden_data(
            dataset_name, config
        )

    def extract_segmentations(self, data):
        return data[1]
    
class Detectron2Segmentor(SegmentorWrapper):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1) -> None:
        # Supported Datasets: ade20k_150, ade20k_full, pascal20, cityscapes, mapillary, coco-stuff, pascal_context_59, pascal_context_459
        dataset = dataset_utils.get_dataset(dataset_name)
        if mask_shape is not None:
            augmentation = [T.ResizeShortestEdge(512, 2048,sample_style='choice')] #[T.Resize(mask_shape)]
        else:
            augmentation = []
        

        self.data_loader = dataset_utils.get_data_loader(
            dataset_name, dataset, transforms=augmentation, batch_size=batch_size
        )
       
        self.concept_labels = dataset_utils.get_dataset_concepts(
            dataset_name, dataset)
        self.mask_shape = mask_shape
    
    def extract_segmentations(self, data):
        raise NotImplementedError
    
    def adjust_data_info(self, data):
        if self.mask_shape is None:
            return data
        # Reshape the data to the mask shape so that the model process it in the same way of the probing model
        for i in range(len(data)):
            data[i]['width'] = self.mask_shape[0]
            data[i]['height'] = self.mask_shape[1]
        return data

class Detectron2GroundTruth(Detectron2Segmentor):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1) -> None:
        super().__init__(dataset_name, mask_shape=mask_shape, batch_size=batch_size)
        self.concept_category = "own"
    def extract_segmentations(self, data):
        return data[0]["sem_seg"].unsqueeze(0).unsqueeze(0)

class Detectron2Model(Detectron2Segmentor):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1) -> None:
        super().__init__(dataset_name, mask_shape=mask_shape, batch_size=batch_size)
        self.mask_shape = mask_shape

    def extract_segmentations(self, data):
        
        #data = self.adjust_data_info(data)  
        # Get the segmentations
        # print(data)
        # print(data[0]['image'].shape)
        # print(data[0]['image'])
        output = self.model(data)
        segmentations = torch.stack([output[i]["sem_seg"] for i in range(len(output)) ])
        segmentations = torch.argmax(segmentations, dim=1)

        del output
        return segmentations.unsqueeze(1)

    def eval(self):
        self.model.eval()

    def set_data_loader(self, dataset_name, min_size, max_size, batch_size):
        
        dataset = dataset_utils.get_dataset(dataset_name)
        augmentation = [T.ResizeShortestEdge(min_size, max_size,sample_style='choice')] 

        self.data_loader = dataset_utils.get_data_loader(
            dataset_name, dataset, transforms=augmentation, batch_size=batch_size
        )

    def set_model(self, cfg, weights_file):
        segmentors_dir = "data/model/segmentors"
        model_weights = f"{segmentors_dir}/{weights_file}"
        self.model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(self.model, save_dir=segmentors_dir).resume_or_load(
            model_weights, resume=False
        )
        self.model.eval()
                  
class Mask2Former(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1, use_classes=None) -> None:
        self.set_data_loader(dataset_name, 1024, 2048, batch_size)
        cfg = config_collection.mask2former_config()

        self.set_model(cfg, "mask2former.pkl")
        self.mask_shape = mask_shape
        self.set_concept_labels('coco_2017_train_panoptic', 'dataset')
        # Mask2Former does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)
    def set_concept_labels(self, dataset_name, use_classes):
        self.concept_labels, self.concept_category = dataset_utils.get_class_names("mask2former", dataset_name, use_classes)       


class Masqclip(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1, use_classes=None) -> None:        
        self.set_data_loader(dataset_name, 512, 2048, batch_size)
        config = config_collection.masqclip_config()
        self.set_model(config, "masqclip_cross_dataset.pth")

        self.mask_shape = mask_shape

        self.set_concept_labels(dataset_name, use_classes)
        # Masqclip does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)
        
    def set_concept_labels(self, dataset_name, use_classes):
        class_names, self.concept_category = dataset_utils.get_class_names("masqclip", dataset_name, use_classes)
        self.concept_labels = self.model.masqclip.set_text_embedding(class_names)
        self.model.masqclip = self.model.masqclip.to("cuda")

class CATSeg(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1, use_classes=None) -> None:
        torch.set_float32_matmul_precision("high")
        self.set_data_loader(dataset_name, 640, 2560, batch_size)
        cfg = config_collection.cat_seg_config()
        self.set_model(cfg, "cat_seg_large.pth")

        self.mask_shape = mask_shape
        self.set_concept_labels(dataset_name, use_classes)
    
    def set_concept_labels(self, dataset_name, use_classes):
        class_names, self.concept_category = dataset_utils.get_class_names("cat_seg", dataset_name, use_classes)
        self.concept_labels = self.model.sem_seg_head.predictor.set_text_embedding(class_names)

class OpenSeeD(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1, use_classes=None) -> None:
        
        #TODO to change based on other datasets
        min_size = 640
        max_size = 2560
        if 'city' in dataset_name:
            min_size = 1024
            max_size = 2048
        elif 'coco' in dataset_name:
            min_size = 800
            max_size = 1333
        self.mask_shape = mask_shape
        self.set_data_loader(dataset_name, min_size, max_size, batch_size)
        cfg = config_collection.openseed_config()
         
        self.model = OpenSeeDBaseModel(cfg, segmentors.openseed.build_model(cfg)).cuda()
        segmentors_dir = "data/model/segmentors"
        model_weights = f"{segmentors_dir}/open_seed_swint_51.2ap.pt"
        DetectionCheckpointer(self.model, save_dir=segmentors_dir).resume_or_load(
                model_weights, resume=False
            )
        self.model = self.model.from_pretrained(model_weights)
        self.model = self.model.eval().cuda()

        self.set_concept_labels(dataset_name, use_classes)
        # OpenSeed does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)
    def set_concept_labels(self, dataset_name, use_classes):
        class_names, self.concept_category = dataset_utils.get_class_names("openseed", dataset_name, use_classes)
        self.concept_labels =  self.model.model.sem_seg_head.predictor.lang_encoder.set_text_embeddings(class_names)


    def extract_segmentations(self, data):
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(data, inference_task="sem_seg")
        segmentations = torch.stack([outputs[i]["sem_seg"] for i in range(len(outputs)) ])
        segmentations = torch.argmax(segmentations, dim=1)
        return segmentations.unsqueeze(1)

class SCAN(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1, use_classes=None) -> None:

        self.set_data_loader(dataset_name, 640, 2560, batch_size)
        cfg = config_collection.scan_config()
        self.set_model(cfg, "SCAN.pth")

        self.mask_shape = mask_shape
        self.set_concept_labels(dataset_name, use_classes)
        # SCAN does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)
    def set_concept_labels(self, dataset_name, use_classes):
        class_names, self.concept_category = dataset_utils.get_class_names("scan", dataset_name, use_classes)
        self.concept_labels = self.model.set_text_embedding(class_names)

class SED(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, batch_size=1, use_classes=None) -> None:
        torch.set_float32_matmul_precision("high")
        self.set_data_loader(dataset_name, 640, 2560, batch_size)
        cfg = config_collection.sed_config()
        self.set_model(cfg, "sed_model_large.pth")
        
        self.mask_shape = mask_shape
        self.set_concept_labels(dataset_name, use_classes)
        # SED does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)
    def set_concept_labels(self, dataset_name, use_classes):
        # this model uses the same class names as CATSeg
        class_names, self.concept_category = dataset_utils.get_class_names("cat_seg", dataset_name, use_classes)
        self.concept_labels = self.model.sem_seg_head.predictor.set_text_embedding(class_names)


