"""Module containing the class for the settings.
It is an adaptation of the code referenced in:
https://github.com/jayelm/compexp/blob/master/vision/settings.py"""


from utils import dataset_utils


class Settings:
    """
    Class that stores all the settings used in each run.
    """

    def __init__(
        self,
        *,
        model,
        pretrained,
        num_clusters,
        beam_limit,
        heuristic,
        length,
        configuration_name,
        experiment_name,
        layer,
        device,
        step_size=80,
        dataset="places365",
        additional_concepts=[],
        additional_dataset=None,
        attribution_method=None,
        granularity=None,
        unification_files=None,
        root_models="data/model/",
        root_datasets="data/dataset/",
        root_segmentations="data/cache/segmentations",
        root_activations="data/cache/activations",
        root_results="data/results",
        root_optimal_info="data/cache/optimal_info",

    ):
        self.model = model
        self.dataset = dataset
        if pretrained == "places365" or pretrained == "imagenet" or pretrained == "cub200" or pretrained == "nli":
            self.pretrained = pretrained
        else:
            self.pretrained = None
        self.num_clusters = num_clusters
        self.beam_limit = beam_limit
        self.heuristic = heuristic
        self.dir_datasets = root_datasets
        self.__root_segmentations = root_segmentations
        self.__root_activations = root_activations
        self.__root_results = root_results
        self.__root_models = root_models
        self.__root_optimal_info = root_optimal_info
        self.device = device
        self.step_size = step_size
        self.granularity = list(granularity) if granularity is not None else None
        self.configuration_name = self.set_configuration_name(configuration_name)
        self.name_experiment = experiment_name
        self.additional_concepts = additional_concepts
        self.layer = layer
        self.length = length
        self.additional_dataset = additional_dataset
        self.unification_files = unification_files
        self.attribution_method = attribution_method

    def set_configuration_name(self, configuration_name):
        if 'combined_categorical' in configuration_name:
            if self.granularity is None:
                raise ValueError("Granularity must be set for combined_categorical")
            else:
                return f'combined_categorical_multiple_granularity'
        else:
            return configuration_name

    def get_root_results(self):
        return self.__root_results + f"/{self.pretrained}"
    
    def get_root_activations(self):
        return self.__root_activations + f"/{self.pretrained}"
    
    def get_root_segmentations(self):
        return self.__root_segmentations
    
    def get_root_optimal_info(self):
        return self.__root_optimal_info

    def get_model_activation_path(self):
        activation_dir = self.get_root_activations() + f"/{self.dataset}/{self.model}"
        return activation_dir
    
    def get_experiment_config(self):
        experiment_config = {
            'dataset_name': self.dataset,
            'configuration_name': self.configuration_name,
            'mask_shape': self.get_mask_shape(),
            'step_size': self.step_size,
            'broden_cfg': self,
            'root_dir_segmentations': self.get_root_segmentations(),
            'device': self.device, 
            'custom_name':self.name_experiment,
            'add_concepts': self.additional_concepts,
            'unification_files': self.unification_files,
        }
        return experiment_config
    
    def get_explanation_config(self):
        config_compositional = {
            'length': self.length,
            'num_clusters': self.num_clusters,
            'layer_name': self.layer,
            'beam_limit': self.beam_limit,
            'heuristic': self.heuristic,
            'attribution_method': self.attribution_method,
            'pretrained': self.pretrained,
            }
        return config_compositional

    def get_concepts(self, concept_type, additional_dataset=None):
        if concept_type == 'user':
            if self.additional_concepts is not None:
                concepts = list(self.additional_concepts)
            else:
                raise ValueError("Additional concepts must be provided for user concept type")
        elif 'cub200' in self.dataset:
            if 'combined_categorical' in self.configuration_name:
                concepts, _ = dataset_utils.get_cub_concepts(
                    additional_dataset=additional_dataset, granularity=self.granularity)
            else:
                #concepts, _ = dataset_utils.get_cub_concepts()
                concepts = concept_type
        else:
            concepts = concept_type
        # if 'cityperson' in self.dataset:
        #     cityperson_labels = ['background', 'pedestrian', 'rider', 'sitting person', 'person with unusual postures', 'group of people']
        #     concepts = cityperson_labels
        return concepts
    def get_image_mean(self):
        """
        Returns the mean of the dataset.
        """
        if self.pretrained == "imagenet":
            return [0.485, 0.456, 0.406]
        elif self.pretrained == "places365":
            return [0.485, 0.456, 0.406]
        else:
            return [0.5, 0.5, 0.5]

    def get_image_stdev(self):
        """
        Returns the standard deviation of the dataset.
        """
        if self.pretrained == "imagenet":
            return [0.229, 0.224, 0.225]
        elif self.pretrained == "places365":
            return [0.229, 0.224, 0.225]
        else:
            return [0.5, 0.5, 0.5]

    def get_num_classes(self):
        """
        Returns the number of classes of the dataset.
        """
        if self.pretrained == "places365":
            return 365
        elif self.pretrained == "imagenet":
            return 1000

    def get_model_file_path(self):
        """
        Returns the path to the pretrained weights of the model.
        """
        if self.pretrained == "places365":
            if self.model == "densenet161":
                model_file_name = (
                    "whole_densenet161_places365_python36.pth.tar"
                )
            else:
                model_file_name = f"{self.model}_places365.pth.tar"
            return self.__root_models + "/zoo/" + model_file_name
        elif self.pretrained == "imagenet":
            return None
        elif self.pretrained == "cub200":
            return self.__root_models + "/other/" + "bird_res50.tar"
        elif self.pretrained == "nli":
            if self.model == "bert" or self.model == "llama":
                # TEMPORARY SOLUTION, CHANGE WITH THE RIGHT PATH TO THE MODEL
                return self.__root_models + "/nlp/" + "bowman.pth"
            return self.__root_models + "/nlp/" + f"{self.model}.pth"
        else:
            return "<UNTRAINED>"

    def get_weights(self):
        """
        Returns the pretrained weights of the model.
        """
        if self.pretrained == "imagenet":
            return "IMAGENET1K_V1"
        elif self.pretrained == "places365" or self.pretrained == "cub200":
            return self.get_model_file_path()
        elif self.pretrained == "nli":
            return self.get_model_file_path()
        else:
            return None

    def get_data_directory(self):
        """
        Returns the directory where the data is stored.
        """
        if self.model != "alexnet":
            return f"{self.dir_datasets}broden1_224"
        else:
            return f"{self.dir_datasets}broden1_227"

    def get_model_root(self):
        """
        Returns the root directory where the models are stored.
        """
        return self.__root_models

    def get_img_size(self):
        """
        Returns the size of the images.
        """
        if self.model != "alexnet":
            return 224
        else:
            return 227

    def get_mask_shape(self):
        """
        Returns the shape of the mask.
        """
        return (112, 112)

    def get_max_mask_size(self):
        """
        Returns the maximum size of the mask.
        """
        return self.get_mask_shape()[0] * self.get_mask_shape()[1]

    def get_feature_names(self):
        """
        Returns the names of the layers that will be used to
        extract the features.
        """
        if self.model == "resnet18":
            return [
                #  ['layer2', '0', 'conv1'], ['layer2', '0', 'conv2'],
                #  ['layer2', '1', 'conv1'], ['layer2', '1', 'conv2'],
                #  ['layer3', '0', 'conv1'], ['layer3', '0', 'conv2'],
                #  ['layer3', '1', 'conv1'], ['layer3', '1', 'conv2'],
                #  ['layer4', '0', 'conv1'], ['layer4', '0', 'conv2'],
                # ["layer4", "1", "conv1"],
                #  ["layer4", "1", "conv2"],
                "layer4"
            ]
        elif self.model == "resnet50":
            return ["layer4"]
        elif self.model == "resnet101":
            return ["layer4"]
        elif self.model == "densenet161":
            return ["features"]
        elif self.model == "alexnet":
            return ["features"]
        elif self.model == "vgg16":
            return ["features"]

    def get_parallel(self):
        """
        Returns True if the model is parallelized.
        """
        if self.pretrained == "places365":
            return True
        else:
            return False
