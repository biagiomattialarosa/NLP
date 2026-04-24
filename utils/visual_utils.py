import os

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from compositional import mask_utils

def get_image(path_image, shape):
    image = torchvision.io.decode_image(torchvision.io.read_file(path_image), mode=torchvision.io.image.ImageReadMode.RGB)
    image = torchvision.transforms.functional.resize(image, shape)
    return image

def add_segmentation(image, mask, color='black', alpha=1):
    segmented_image = torchvision.utils.draw_segmentation_masks(
            image, mask, alpha=alpha, colors=color)
    return segmented_image

def visualize_segmentation(path_image, mask):
    mask_shape = (mask.shape[0], mask.shape[1])
    image = get_image(path_image, mask_shape)
    return add_segmentation(image, ~mask)

def get_figure(imgs, labels=None, width=8, height=4.8, hspace=0.5):
    """Show images in a grid"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(labels, list):
        labels = [labels]
    fig, axs = plt.subplots(
        nrows=len(imgs), squeeze=False,
        gridspec_kw={'wspace': 0, 'hspace': hspace})
    fig.set_figwidth(width)
    fig.set_figheight(height)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if labels is not None:
            axs[i, 0].set_title(labels[i])
    return fig

def save_fig(figure, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    figure.savefig(
        f'{dir}/' +
        f'{name}.png')

def get_grid_neuron(dataset, label, masks, bitmaps, number_samples, mask_shape, device, starting_percentage=0, ending_percentage=None, sorted=True):
    
    label_mask = mask_utils.get_formula_mask(label, masks).to(device)
    fire_and_label = bitmaps & label_mask

    sorted_indices, sorted_values = sort_indices_by_coverage(fire_and_label)
    selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    # If the selected indices are not enough, try to select the samples with at least 0.01 coverage
    while len(selected_indices) < number_samples and starting_percentage > 0.01:
        starting_percentage -= 0.01
        selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    if len(selected_indices) > 0:
        segmented_images = []
        for index in selected_indices:
            path_image = dataset[index]['file_name']
            mask_shape = (mask_shape[0], mask_shape[1])
            bitmaps_image = bitmaps[index].reshape(mask_shape[0], mask_shape[1])
            image = get_image(path_image, mask_shape)
            image_plus_neuron = add_segmentation(image, bitmaps_image, color='blue', alpha=0.5)
            segmented_images.append(image_plus_neuron)
        grid = torchvision.utils.make_grid(
            segmented_images, padding=2, pad_value=255)  
    else:
        grid = None
    return grid

# def get_grid_samples(dataset, label_inside, label_whole, masks, neuron_fires, number_samples, mask_shape, device):
    
#     label_mask = mask_utils.get_formula_mask(label_inside, masks)
#     samples_seg_atom = get_active_samples(neuron_fires, label_inside, label_whole, masks, device)
#     positive_indices = torch.nonzero(samples_seg_atom).flatten()
#     selected_indicees = positive_indices[:number_samples]
#     selected_images = []
#     selected_masks = []

#     for index in selected_indicees:
#         path_image = dataset[index]['file_name']
#         image =  torchvision.io.decode_image(torchvision.io.read_file(path_image), mode=torchvision.io.image.ImageReadMode.RGB)
#         image = torchvision.transforms.functional.resize(image, mask_shape)
#         selected_images.append(image)
#         mask = ~label_mask[index]
#         selected_masks.append(mask)
#         images = []
#     for index_sample, sample_image in enumerate(selected_images):
#         #image = sample_image.permute(2, 0, 1)
#         mask_concept = selected_masks[index_sample].reshape(
#             mask_shape[0], mask_shape[1])
#         segmented_image = torchvision.utils.draw_segmentation_masks(
#             sample_image, mask_concept, alpha=1, colors='black')
#         segmented_image = torchvision.transforms.functional.resize(segmented_image, (512,512))
#         images.append(segmented_image)
#     grid = torchvision.utils.make_grid(
#             images, padding=2, pad_value=255)
#     return grid

def get_active_samples(sample_bitmaps, atom, formula, masks, device):
    mask_atom = mask_utils.get_formula_mask(atom, masks).to(
        device)
    samples_atom = torch.any(mask_atom, dim=1)
    mask_formula = mask_utils.get_formula_mask(formula, masks).to(
        device)
    samples_formula = torch.any(mask_formula, dim=1)
    active_samples = (samples_atom & samples_formula) & sample_bitmaps
    return active_samples

def get_active_samples_neuron(sample_bitmaps, atom, formula, masks, device):
    mask_atom = mask_utils.get_formula_mask(atom, masks).to(
        device)
    samples_atom = torch.any(mask_atom, dim=1)
    mask_formula = mask_utils.get_formula_mask(formula, masks).to(
        device)
    samples_formula = torch.any(mask_formula, dim=1)
    active_samples = (samples_atom & samples_formula) & sample_bitmaps
    return active_samples


def make_grid(selected_images, selected_masks, mask_shape):
    images = []
    for index_sample, sample_image in enumerate(selected_images):
        #image = sample_image.permute(2, 0, 1)
        mask_concept = selected_masks[index_sample].reshape(
            mask_shape[0], mask_shape[1])
        segmented_image = torchvision.utils.draw_segmentation_masks(
            sample_image, mask_concept, alpha=1, colors='black')
        segmented_image = torchvision.transforms.functional.resize(segmented_image, (512,512))
        images.append(segmented_image)
    grid = torchvision.utils.make_grid(
        images, padding=2, pad_value=255)  
    return grid

def get_grid(label_mask, dataset, indices, mask_shape):
    selected_images = []
    selected_masks = []
    for index in indices:
        path_image = dataset[index]['file_name']
        image =  torchvision.io.decode_image(torchvision.io.read_file(path_image), mode=torchvision.io.image.ImageReadMode.RGB)
        image = torchvision.transforms.functional.resize(image, mask_shape)
        selected_images.append(image)
        mask = ~label_mask[index]
        selected_masks.append(mask)
    
    grid = make_grid(selected_images, selected_masks, mask_shape)
    return grid

def get_grid_overlap(dataset, label_inside, label_whole, masks, bitmaps, number_samples, mask_shape, device, starting_percentage=0, ending_percentage=None, sorted=True):
    
    label_mask = mask_utils.get_formula_mask(label_inside, masks).to(device)
    formula_mask = mask_utils.get_formula_mask(label_whole, masks).to(device)
    fire_and_label = bitmaps & label_mask & formula_mask
    sorted_indices, sorted_values = sort_indices_by_coverage(fire_and_label)
    selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    # If the selected indices are not enough, try to select the samples with at least 0.01 coverage
    while len(selected_indices) < number_samples and starting_percentage > 0.0:
        starting_percentage -= 0.01
        starting_percentage = max(starting_percentage, 0)
        selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    if len(selected_indices) > 0:
        grid = get_grid(label_mask, dataset, selected_indices, mask_shape)
    else:
        grid = None
    return grid

def get_grid_by_samples(dataset, label_inside, label_whole, masks, bitmaps, number_samples, mask_shape, device, starting_percentage=0, ending_percentage=None, sorted=True):
    
    label_mask = mask_utils.get_formula_mask(label_inside, masks).to(device)
    formula_mask = mask_utils.get_formula_mask(label_whole, masks).to(device)
    bitmaps_samples = bitmaps.any(dim=1)
    label_samples = label_mask.any(dim=1)
    formula_samples = formula_mask.any(dim=1)
    fire_and_label = bitmaps_samples & label_samples & formula_samples
    positive_indices = torch.nonzero(fire_and_label).flatten()
    selected_indices = positive_indices[:number_samples]
    if len(selected_indices) > 0:
        grid = get_grid(label_mask, dataset, selected_indices, mask_shape)
    else:
        grid = None
    return grid

def get_grid_explanation(dataset, label, masks, bitmaps, number_samples, mask_shape, device, starting_percentage=0, ending_percentage=None, sorted=True):
    formula_mask = mask_utils.get_formula_mask(label, masks).to(device)
    fire_and_label = bitmaps  & formula_mask
    sorted_indices, sorted_values = sort_indices_by_coverage(fire_and_label)
    selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    # If the selected indices are not enough, try to select the samples with at least 0.01 coverage
    while len(selected_indices) < number_samples and starting_percentage > 0.01:
        starting_percentage -= 0.01
        selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    if len(selected_indices) > 0:
        grid = get_grid(fire_and_label, dataset, selected_indices, mask_shape)
    else:
        grid = None
    return grid



def get_grid_intersection(dataset, label, masks, bitmaps, number_samples, mask_shape, device, starting_percentage=0, ending_percentage=None, sorted=True):
    
    label_mask = mask_utils.get_formula_mask(label, masks).to(device)
    fire_and_label = bitmaps & label_mask

    sorted_indices, sorted_values = sort_indices_by_coverage(fire_and_label)
    selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    # If the selected indices are not enough, try to select the samples with at least 0.01 coverage
    while len(selected_indices) < number_samples and starting_percentage > 0.01:
        starting_percentage -= 0.01
        selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    if len(selected_indices) > 0:
        segmented_images = []
        for index in selected_indices:
            path_image = dataset[index]['file_name']
            mask_shape = (mask_shape[0], mask_shape[1])
            image = get_image(path_image, mask_shape)
            fire_image = fire_and_label[index].reshape(mask_shape[0], mask_shape[1])
            image_plus_mask = add_segmentation(image, fire_image, color='blue', alpha=0.6)
            segmented_images.append(image_plus_mask)
        grid = torchvision.utils.make_grid(
            segmented_images, padding=2, pad_value=255)  
    else:
        grid = None
    return grid

def get_grid_masked_positions(dataset, label, masks, bitmaps, number_samples, mask_shape, device, starting_percentage=0, ending_percentage=None, sorted=True):
    
    label_mask = mask_utils.get_formula_mask(label, masks).to(device)
    fire_and_label = bitmaps & label_mask

    sorted_indices, sorted_values = sort_indices_by_coverage(fire_and_label)
    selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    # If the selected indices are not enough, try to select the samples with at least 0.01 coverage
    while len(selected_indices) < number_samples and starting_percentage > 0.01:
        starting_percentage -= 0.01
        selected_indices = extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=sorted, starting_percentage=starting_percentage, ending_percentage=ending_percentage)
    if len(selected_indices) > 0:
        segmented_images = []
        for index in selected_indices:
            path_image = dataset[index]['file_name']
            mask_shape = (mask_shape[0], mask_shape[1])
            image = get_image(path_image, mask_shape)
            label_mask_image = label_mask[index].reshape(mask_shape[0], mask_shape[1])
            bitmaps_image = bitmaps[index].reshape(mask_shape[0], mask_shape[1])
            image_plus_label = add_segmentation(image, ~label_mask_image, color='black', alpha=0.9)
            image_plus_neuron = add_segmentation(image_plus_label, bitmaps_image, color='blue', alpha=0.5)
            segmented_images.append(image_plus_neuron)
        grid = torchvision.utils.make_grid(
            segmented_images, padding=2, pad_value=255)  
    else:
        grid = None
    return grid

def sort_indices_by_coverage(bitmaps):
    fire_cov_per_sample = bitmaps.sum(dim=1) / bitmaps.shape[1]
    sorted_values, sorted_indices = torch.sort(fire_cov_per_sample, descending=True)
    return sorted_indices, sorted_values

def extract_candidate_indices(sorted_indices, sorted_values, number_samples, sorted=False, starting_percentage=None, ending_percentage=None):
    if starting_percentage is not None:
        above_starting = sorted_values > starting_percentage
    else:
        above_starting = torch.ones_like(sorted_values).bool()
    if ending_percentage is not None:
        below_ending = sorted_values < ending_percentage
    else:
        below_ending = torch.ones_like(sorted_values).bool()
    candidate_indices = sorted_indices[above_starting & below_ending]    
    if len(candidate_indices) > number_samples:
        if sorted:
            selected_indices = candidate_indices[:number_samples]
        else:
            # Randomly select the samples among the candidates
            selected_indices = torch.randperm(len(candidate_indices))[:number_samples]
    else:
        selected_indices = candidate_indices
    return selected_indices

def get_grid_neuron_samples(dataset, atoms_to_exclude, masks, bitmaps, neuron_fires, number_samples, mask_shape, starting_percentage=0.4):
    fires_to_consider = neuron_fires.clone()
    for atom in atoms_to_exclude:
        atom_mask = mask_utils.get_formula_mask(atom, masks).cuda()
        samples_atom_mask = torch.any(atom_mask, dim=1)
        fires_to_consider = ~samples_atom_mask & fires_to_consider
    positive_indices = torch.nonzero(fires_to_consider).flatten()
    fire_cov_per_sample = bitmaps.sum(dim=1) / bitmaps.shape[1]
    # Select the positive indices with at least 30% of coverage if possible
    selected_indices = positive_indices[fire_cov_per_sample[positive_indices] > starting_percentage]
    while len(selected_indices) < number_samples and starting_percentage > 0:
        starting_percentage -= 0.05
        selected_indices = positive_indices[fire_cov_per_sample[positive_indices] > starting_percentage]

    if len(selected_indices) < number_samples:
        remaining_samples = number_samples - len(selected_indices)
        selected_indices = torch.cat((selected_indices, positive_indices[:remaining_samples]))
    selected_indices = selected_indices[:number_samples]

    #selected_indicees = positive_indices[:number_samples]
    grid = get_grid(bitmaps, dataset, selected_indices, mask_shape)

    return grid