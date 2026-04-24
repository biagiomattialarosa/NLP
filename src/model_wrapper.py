from typing import List 
import os 

import torchvision
import torch
import numpy as np

from utils import dataset_utils
import src.models.nlp_models as nlp_models
from datasets.snli import AnalysisDataset
#from src.models.DifferentiableSVD.model_init import Newmodel as CUBmodel 
#from src.models.DifferentiableSVD.src.representation import SEB

class ModelWrapper:
    
    
    def __init__(self, *args, **kwargs):
        self.model = None
        self.data_loader = None
        self.transformation = None
        self.iter_data = None

    def load_model_from_settings(
        self,
        *args, **kwargs
    ):
        raise NotImplementedError

    def set_loader(self, *args, **kwargs):
        raise NotImplementedError

    # Reference: https://github.com/jayelm/compexp/blob/master/vision/loader/model_loader.py
    def hook(self, hook_fn, feature_names):
        """
        Register a hook to a model.

        Args:
            model (torch.nn.Module): model
            hook_fn (function): hook function
            feature_names (list): list of feature names

        Returns:
            list: list of handles
        """
        handles = []
        for name in feature_names:
            if isinstance(name, list):
                # Iteratively retrive the module
                hook_model = self.model
                for n in name:
                    hook_model = hook_model._modules.get(n)
            else:
                hook_model = self.model._modules.get(name)
            if hook_model is None:
                raise ValueError(f"Couldn't find feature {name}")
            handles.append(hook_model.register_forward_hook(hook_fn))
        return handles
    


   
    @torch.no_grad()
    def compute_activations( self,
            layers: List):
        """Retrieve the activations of a given layer feeding the model with the
        images in the loader.

        Args:
            layers (list): list of layers

        Returns:
            torch.Tensor: activations
        """
        device = next(self.model.parameters()).device
        temp_activations = []

        def hook_feature(module, inp, output):
            temp_activations.append(output.data.cpu())

        handles = self.hook(hook_feature, layers)

        activations = [[] for _ in range(len(layers))]
        for data in self.data_loader:
            # Transformations
            images = self.iter_data(data)

            if len(images.shape) == 4:
                images = images.squeeze(0)

            if self.transformation is not None:
                images = self.transformation(images)
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            
            # Move to GPU
            images = images.to(device)

            # Forward pass
            _ = self.model(images)

            # collect data
            for index_layer in range(len(layers)):
                activations[index_layer].append(temp_activations[index_layer])

            # Empty the temp list
            del temp_activations[:]
            temp_activations = []

        for handle in handles:
            handle.remove()
        for layer in range(len(layers)):
            activations[layer] = torch.unsqueeze(
                torch.cat(activations[layer]), dim=0
            )
        activations = torch.cat(activations, dim=0)
        return activations

    def get_layer_activations(self,  layer, dir_activations):
        """Checks if the activations are already computed and saved, otherwise
        computes them and saves them.

        Args:
            model (torch.nn.Module): model
            layer (str): layer name
            units (list): list of units whose activations are to be computed

        Returns:
            activations (list): list of activations for each unit
        """
        if self.data_loader is None:
            raise ValueError("Data loader not set. Please set the data loader first (run wrapper_item.set_loader(args)).")
        layer_file = f"{dir_activations}/{layer}.npy"
        total_activations = []
        if os.path.exists(layer_file):
            total_activations = np.load(layer_file)
        else:
            print(f"Computing activations for layer {layer}")

            activations = self.compute_activations(
               [layer]
            )
            # since the function checks one layer at a time
            activations = activations[0].numpy()
            if not os.path.exists(dir_activations):
                os.makedirs(dir_activations)
            np.save(f"{layer_file}", activations)
            total_activations = activations
        total_activations = torch.from_numpy(total_activations)
        return total_activations

class NLPModelWrapper(ModelWrapper):
    def __init__(self, model_name, weights, device):
        super().__init__()
        self.vocab = None
        self.dataset = None
        self.model = self.load_checkpoint(model_name, weights, device)

    def load_checkpoint(
        self,
        model_name,
        weights,
        device,
    ):
        """
        Load the model from the settings.
        Args:
            config (settings.Settings): current config of the settings
            device (torch.device): device to use

        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model:{model_name}")
        ckpt = torch.load(weights, map_location="cpu")

        # Load vocab from trained model
        self.vocab = {"itos": ckpt["itos"], "stoi": ckpt["stoi"]}
        
        # Load model
        model_enc = nlp_models.TextEncoder(len(ckpt["stoi"]))
        model_clf = nlp_models.BowmanEntailmentClassifier
        model = model_clf(model_enc)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model  

    def set_loader(self, dataset_name, cfg=None):
        precomputed_features = "data/features/snli_1.0_dev.feats"

        with open(precomputed_features, "r") as f:
            lines = f.readlines()
        self.dataset = AnalysisDataset(lines, self.vocab, "data/datasets/snli_1.0/", 'dev')
        self.data_loader = dataset_utils.get_data_loader(
                dataset_name, self.dataset)
        self.iter_data = dataset_utils.get_data_iter_fn(dataset_name)

    def get_dataset(self):
        return self.dataset

    def get_features(self, neighbors=None, features=None):
        if self.data_loader is not None and self.dataset is not None:
            toks, feats, idxs = self.dataset.get_features(self.data_loader)
            if features is not None:
                tokens_feat = "tokens" in features
                tags_feat = "tags" in features
                overlap_feat = "overlap" in features
                class_feat = "class" in features
            else:               
                tokens_feat = tags_feat = overlap_feat = True
                class_feat = False
            
            print(f"Using features: Tokens {tokens_feat}, Tags {tags_feat}, Overlap {overlap_feat}, Class {class_feat}")
            tok_feats, tok_feats_vocab =  self.dataset.to_sentence(toks, feats, tokens_feats=tokens_feat, tags_feats=tags_feat, overlap_feats=overlap_feat, cls_feats=class_feat)
            if neighbors is None:
                return tok_feats, tok_feats_vocab, None
            elif neighbors == 'old':
                tok_tot_feats, tok_tot_feats_vocab = self.dataset.add_old_neighbors_feat(tok_feats_vocab, tok_feats)
                constraints = None
            elif neighbors == 'baseline':
                vecpath = 'data/datasets/snli_1.0/snli_1.0_dev.vec'
                tok_tot_feats, tok_tot_feats_vocab, _ = self.dataset.add_neighbors_feat(tok_feats_vocab, tok_feats, vecpath)
                constraints = None
            elif neighbors == 'new':
                #vecpath = 'data/datasets/snli_1.0/snli_1.0_devMIO.vec'
                vecpath = 'data/datasets/snli_1.0/snli_1.0_dev.vec'
                tok_tot_feats, tok_tot_feats_vocab, constraints = self.dataset.add_neighbors_feat(tok_feats_vocab, tok_feats, vecpath)
                constraints = None
            elif neighbors == 'tokens':
                exit()
                vecpath = 'data/datasets/snli_1.0/snli_1.0_dev.vec'
                tok_tot_feats, tok_tot_feats_vocab, constraints = self.dataset.keep_tokens_feat(tok_feats_vocab, tok_feats, vecpath)
            return tok_tot_feats, tok_tot_feats_vocab, constraints
        else:
            raise ValueError("Data loader or dataset not set. Please set the data loader first (run wrapper_item.set_loader(args)).")

    @torch.no_grad()
    def compute_activations( self,
            layers: List):
        """Retrieve the activations of a given layer feeding the model with the
        images in the loader.

        Args:
            layers (list): list of layers

        Returns:
            torch.Tensor: activations
        """
        device = next(self.model.parameters()).device

        activations = [[] for _ in range(len(layers))]
        for index_layer in range(len(layers)):
            for data in self.data_loader:
                # Transformations
                batch = self.iter_data(data)


                # Move to GPU
                s1, s1len, s2, s2len, _ = batch
                s1 = s1.to(device)
                s1len = s1len.to(device)
                s2 = s2.to(device)
                s2len = s2len.to(device)

                # Forward pass
                states = self.model.get_final_reprs(s1, s1len, s2, s2len)
                activations[index_layer].append(states.cpu())

        for layer in range(len(layers)):
            activations[layer] = torch.unsqueeze(
                torch.cat(activations[layer]), dim=0
            )
        activations = torch.cat(activations, dim=0)
        return activations  
      
class ImageNetModel(ModelWrapper):
    def __init__(self, model_name, device):
        super().__init__()
        self.model = self.load_checkpoint(model_name, device)
    
    def load_checkpoint(
        self,
        model_name,
        device,
    ):
        """
        Load the model from the settings.
        Args:
            config (settings.Settings): current config of the settings
            device (torch.device): device to use

        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model:{model_name}")

        model_fn = torchvision.models.__dict__[model_name]

        model = model_fn(pretrained=True)
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model

    def set_loader(self, dataset_name, cfg=None):
        if dataset_name == 'broden':
            self.data_loader, _ = dataset_utils.get_broden_data(dataset_name, cfg)

        else:
            dataset = dataset_utils.get_dataset(dataset_name)
            self.data_loader = dataset_utils.get_data_loader(
                dataset_name, dataset)

        self.transformation = dataset_utils.get_probing_transformations(
            dataset_name, 32, [0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225]
        )

        self.iter_data = dataset_utils.get_data_iter_fn(dataset_name)


class UntrainedModel(ModelWrapper):
    def __init__(self, model_name, device):
        super().__init__()
        self.model = self.load_checkpoint(model_name, device)
    
    def load_checkpoint(
        self,
        model_name,
        device,
    ):
        """
        Load the model from the settings.
        Args:
            config (settings.Settings): current config of the settings
            device (torch.device): device to use

        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model:{model_name}")

        model_fn = torchvision.models.__dict__[model_name]

        model = model_fn(pretrained=False)
        model = model.to(device)
        model.eval()
        print(f"Untrained {model_name} loaded in {device}. Modality: Evaluation")
        return model

    def set_loader(self, dataset_name, cfg=None):
        if dataset_name == 'broden':
            self.data_loader, _ = dataset_utils.get_broden_data(dataset_name, cfg)

        else:
            dataset = dataset_utils.get_dataset(dataset_name)
            self.data_loader = dataset_utils.get_data_loader(
                dataset_name, dataset)

        self.transformation = dataset_utils.get_probing_transformations(
            dataset_name, 32, [0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225]
        )

        self.iter_data = dataset_utils.get_data_iter_fn(dataset_name)

class Place365Model(ModelWrapper):
    def __init__(self, model_name, weights, device):
        if "densenet" in model_name:
            raise NameError("Please use DenseNetPlace365 class for DenseNet models")
        super().__init__()
        self.input_size = 227 if "alexnet" in model_name else 224
        self.model = self.load_checkpoint(model_name, weights, device)
    
    def set_loader(self, dataset_name, cfg=None):
        if dataset_name == 'broden':
            self.data_loader, _ = dataset_utils.get_broden_data(dataset_name, cfg)

        else:
            dataset = dataset_utils.get_dataset(dataset_name)
            self.data_loader = dataset_utils.get_data_loader(
                dataset_name, dataset)

        self.transformation = dataset_utils.get_probing_transformations(
            dataset_name, self.input_size, [0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225]
        )

        self.iter_data = dataset_utils.get_data_iter_fn(dataset_name)
        
    def load_checkpoint(
        self,
        model_name,
        weights,
        device,
    ):
        """
        Load the model from the settings.
        Args:
            config (settings.Settings): current config of the settings
            device (torch.device): device to use

        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model:{model_name}\n\tfrom {weights}")

        model_fn = torchvision.models.__dict__[model_name]

        checkpoint = torch.load(weights, map_location=device)
        model = model_fn(num_classes=365)
        # the data parallel layer will add 'module' before each
        # layer name
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model
    
class DenseNetPlace365(Place365Model):
    def __init__(self, model_name, weights, device):
        self.model = self.load_checkpoint(model_name, weights, device)
        self.input_size = 224
    def load_checkpoint(
        self,
        model_name,
        weights,
        device,
    ):
        """
        Load the model from the settings.
        Args:
            config (settings.Settings): current config of the settings
            device (torch.device): device to use

        Returns:
            torch.nn.Module: model
        """
        def rep(k):
            for i in range(6):
                k = k.replace(f"norm.{i}", f"norm{i}")
                k = k.replace(f"relu.{i}", f"relu{i}")
                k = k.replace(f"conv.{i}", f"conv{i}")
            return k
        print(f"Loading model:{model_name}\n\tfrom {weights}")

        model_fn = torchvision.models.__dict__[model_name]

        checkpoint = torch.load(weights, map_location=device)
        model = model_fn(num_classes=365)
        # Fix old densenet pytorch names.        
        state_dict = checkpoint.state_dict()
        state_dict = {rep(k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model
    
class Cub200Model(ModelWrapper):
    def __init__(self, weights, device):
        super().__init__()
        self.model = self.load_checkpoint('resnet50', weights, device)
    
    def load_checkpoint(
        self,
        model_name,
        weights,
        device,
    ):

        print(f"Loading model:{model_name}")

        model_fn = CUBmodel
        representation = {'function':SEB,
                          'is_vec':True,
                          'input_dim':2048,
                          'dimension_reduction':256}
        model = model_fn(model_name, representation, 200, 0, pretrained=True)

        checkpoint = torch.load(weights)
          # the data parallel layer will add 'module' before each
        # layer name
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model

    def set_loader(self, dataset_name, cfg=None):
        if dataset_name == 'broden':
            self.data_loader, _ = dataset_utils.get_broden_data(dataset_name, cfg)

        else:
            dataset = dataset_utils.get_dataset(dataset_name)
            self.data_loader = dataset_utils.get_data_loader(
                dataset_name, dataset)
        self.transformation = dataset_utils.get_probing_transformations(
            dataset_name, 448, [0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225]
        )
        self.iter_data = dataset_utils.get_data_iter_fn(dataset_name)