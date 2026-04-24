from torcheval.metrics.functional import binary_f1_score, binary_recall, binary_precision

def sample_similarity(atom_a, label_a, masks_a,  atom_b, label_b, masks_b, device):
    samples_label_a = get_active_samples(atom_a, label_a, masks_a, device)
    samples_label_b = get_active_samples(atom_b, label_b, masks_b, device)
    is_eq = is_equivalent(samples_label_a, samples_label_b)
    is_spec = is_specialization(samples_label_a, samples_label_b)
    is_gen = is_generalization(samples_label_a, samples_label_b)
    rate_spec = rate_specialization(samples_label_a, samples_label_b).item()
    rate_gen = rate_generalization(samples_label_a, samples_label_b).item()
    return is_eq, is_spec, is_gen, rate_spec, rate_gen

def get_single_token_label(f, names_vector):
    if isinstance(f, F.And) or isinstance(f, F.Or):
        # something
        str_rep_left = get_single_token_label(f.left, names_vector)
        str_rep_right = get_single_token_label(f.right, names_vector)
        str_rep = []
        for index_left in range(len(str_rep_left)):
            for index_right in range(len(str_rep_right)):
                str_rep.append(f'{str_rep_left[index_left]} {f.op} {str_rep_right[index_right]}')

        return str_rep
    elif isinstance(f, F.Not):
        # something
        str_rep = F.get_formula_str( f.val, names_vector)
        if ', ' in str_rep:
            str_rep = str_rep.split(', ')
        else:
            str_rep = [str_rep]
        for index in range(len(str_rep)):
            str_rep[index] = f'NOT {str_rep[index]}'
        return str_rep
    else:
        str_rep = F.get_formula_str(f, names_vector)
        if ', ' in str_rep:
            return str_rep.split(', ')
        else:
            return [str_rep]

def get_active_samples(atom, formula, masks, device):
    mask_atom = mask_utils.get_formula_mask(atom, masks).to(
        device)
    samples_atom = torch.any(mask_atom, dim=1)
    mask_formula = mask_utils.get_formula_mask(formula, masks).to(
        device)
    samples_formula = torch.any(mask_formula, dim=1)
    active_samples = samples_atom & samples_formula
    return active_samples

def is_specialization(samples_formula_a, samples_formula_b):
    if is_equivalent(samples_formula_a, samples_formula_b):
        return False
    TP = torch.sum(samples_formula_a & samples_formula_b)
    FP = torch.sum(samples_formula_a & ~samples_formula_b)
    return TP > 0 and FP == 0

def rate_specialization(samples_formula_a, samples_formula_b):
    # if is_equivalent(samples_formula_a, samples_formula_b):
    #     return False
    TP = torch.sum(samples_formula_a & samples_formula_b)
    FP = torch.sum(samples_formula_a & ~samples_formula_b)
    return 1 - (FP / (TP + FP))

def is_generalization(samples_formula_a, samples_formula_b):
    if is_equivalent(samples_formula_a, samples_formula_b):
        return False
    TP = torch.sum(samples_formula_a & samples_formula_b)
    FN = torch.sum(~samples_formula_a & samples_formula_b)
    return TP > 0 and FN == 0

def rate_generalization(samples_formula_a, samples_formula_b):
    # if is_equivalent(samples_formula_a, samples_formula_b):
    #     return False
    TP = torch.sum(samples_formula_a & samples_formula_b)
    FN = torch.sum(~samples_formula_a & samples_formula_b)
    return 1 - (FN / (TP + FN))

def is_equivalent(samples_formula_a, samples_formula_b):
    return torch.all(samples_formula_a == samples_formula_b)

def check_label_in_gt(atom_to_check,  labels_check, reference_atoms, labels_ref):

    # Transform the atom in single token
    seg_str_atom = get_single_token_label(atom_to_check, labels_check)
    seg_str_atom.extend(get_single_token_label(atom_to_check.flip_atoms(), labels_check))

    # Atoms are numeric, but diffferent models can use different numeric values
    # We need to check the labels
    human_str_atoms=[]
    for f in reference_atoms:
        if isinstance(f, F.And):
            # In this case we need to include both the equivalent formulas of swaped atoms
            human_str_atoms.extend(get_single_token_label(f, labels_ref))
            human_str_atoms.extend(get_single_token_label(f.flip_atoms(), labels_ref))
        else:
            # in all the other cases we can just consider their string representation
            str_repr = get_single_token_label(f, labels_ref)
            human_str_atoms.extend(str_repr)
    for atom in seg_str_atom:
        if atom in human_str_atoms:
            return True
    return False

def get_active_samples_neuron(sample_bitmaps, atom, formula, masks, device):
    mask_atom = mask_utils.get_formula_mask(atom, masks).to(
        device)
    samples_atom = torch.any(mask_atom, dim=1)
    mask_formula = mask_utils.get_formula_mask(formula, masks).to(
        device)
    samples_formula = torch.any(mask_formula, dim=1)
    active_samples = (samples_atom & samples_formula) & sample_bitmaps
    return active_samples



def get_index_most_similar(sample_bitmaps, atomA, formulaA, masksA, list_atomsB, formulaB, masksB, device):
    active_samples_A = get_active_samples_neuron(sample_bitmaps, atomA, formulaA, masksA, device)
    # Get samples that are activated in the formula
    f1_similarities = []
    recall_similarities = []
    precision_similarities = []
    for atomB in list_atomsB:
        active_samples_B = get_active_samples_neuron(sample_bitmaps, atomB, formulaB, masksB, device)
        # Compute iou between the two masks
        f1 = binary_f1_score(active_samples_B, active_samples_A)
        f1_similarities.append(f1.item())
        recall = binary_recall(active_samples_B, active_samples_A)
        recall_similarities.append(recall.item())
        precision = binary_precision(active_samples_B, active_samples_A)
        precision_similarities.append(precision.item())
    
    scores = [f1_similarities, recall_similarities, precision_similarities]
    score_id = np.argmax([max(f1_similarities), max(recall_similarities), max(precision_similarities)])
    index_atom = np.argmax(scores[score_id])
    return index_atom


def get_num_detected(atom_to_check,  labels_check, reference_atoms, labels_ref, include_specialization=True):

    # Specialization case e.g., (weel) is considered equivalent to (weel AND NOT building AND NOT sky)
    if include_specialization and isinstance(atom_to_check, F.And) and isinstance(atom_to_check.right, F.Not):
        return get_num_detected(atom_to_check.left, labels_check, reference_atoms, labels_ref, include_specialization)
    elif include_specialization and isinstance(reference_atoms,F.And) and isinstance(reference_atoms.right, F.Not):
        return get_num_detected(atom_to_check, labels_check, reference_atoms.left, labels_ref, include_specialization)
    else:
        # Removing the NOT could produce not atomic form
        new_atoms_to_check = atom_to_check.get_atoms()
        num_detected = 0
        for atom in new_atoms_to_check:
            if check_label_in_gt(atom, labels_check, reference_atoms, labels_ref):
                num_detected += len(atom)
        if isinstance(atom_to_check,F.And):
            print(atom_to_check)
            exit()
        return num_detected   

def compute_atom_contribution(atom_mask, formula_mask, neuron_bitmap):
    dict_formula = compute_scores(formula_mask, neuron_bitmap)
    dict_atom = compute_scores(atom_mask, neuron_bitmap)

    # Individual contributions
    iou_contribution = dict_atom['iou'] / dict_formula['iou']
    activation_coverage_contribution = dict_atom['activation_coverage'] / dict_formula['activation_coverage']
    explanation_coverage_contribution = dict_atom['explanation_coverage'] / dict_formula['explanation_coverage']
    detection_accuracy_contribution = dict_atom['label_coverage'] / dict_formula['label_coverage']
    samples_coverage_contribution = dict_atom['samples_coverage'] / dict_formula['samples_coverage']
    return iou_contribution, activation_coverage_contribution, explanation_coverage_contribution, detection_accuracy_contribution, samples_coverage_contribution

def compute_scores(neuron_mask, label_mask):
    iou = metrics.iou(neuron_mask, label_mask)
    activation_coverage = metrics.activations_coverage(
        neuron_mask, label_mask
    )
    detection_accuracy = metrics.detection_accuracy(
        neuron_mask, label_mask
    )
    samples_coverage = metrics.samples_coverage(
        neuron_mask, label_mask
    )


    explanation_coverage = metrics.explanation_coverage(
        neuron_mask, label_mask)

    dict_results = {
        "iou": iou.item(),
        "activation_coverage": activation_coverage.item(),
        "label_coverage": detection_accuracy.item(),
        "samples_coverage": samples_coverage.item(),
        "explanation_coverage": explanation_coverage.item(),
    }
    return dict_results

def get_segmentor(segmentor_name, dataset_name, mask_shape, cfg, use_classes):
    if segmentor_name == 'human':
        if 'broden' in dataset_name:
            segmentor = BrodenGroundTruth(dataset_name, config=cfg)
        else:
            segmentor = Detectron2GroundTruth(dataset_name, mask_shape=mask_shape)
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

def centroid_similarity(formulaA, masksA, list_atomsB, masksB, device):
    print(formulaA)
    mask_formulaA = mask_utils.get_formula_mask(formulaA, masksA).to(
        device)
    for formulaB in list_atomsB:
        print(formulaB)
        mask_formulaB = mask_utils.get_formula_mask(formulaB, masksB).to(
            device)
    exit()
    return

def get_active_samples(atom, formula, masks, device):
    mask_atom = mask_utils.get_formula_mask(atom, masks).to(
        device)
    samples_atom = torch.any(mask_atom, dim=1)
    mask_formula = mask_utils.get_formula_mask(formula, masks).to(
        device)
    samples_formula = torch.any(mask_formula, dim=1)
    active_samples = samples_atom & samples_formula
    return active_samples

def iou_explanation_overlap(bitmaps, atomA, formulaA, masksA, list_atomsB, formulaB, masksB, device):
    active_samples_A = get_active_samples_neuron(sample_bitmaps, atomA, formulaA, masksA, device)
    # Get samples that are activated in the formula
    f1_similarities = []
    recall_similarities = []
    precision_similarities = []
    for atomB in list_atomsB:
        active_samples_B = get_active_samples_neuron(sample_bitmaps, atomB, formulaB, masksB, device)
        # Compute iou between the two masks
        f1 = binary_f1_score(active_samples_B, active_samples_A)
        f1_similarities.append(f1.item())
        recall = binary_recall(active_samples_B, active_samples_A)
        recall_similarities.append(recall.item())
        precision = binary_precision(active_samples_B, active_samples_A)
        precision_similarities.append(precision.item())
    
    scores = [f1_similarities, recall_similarities, precision_similarities]
    score_id = np.argmax([max(f1_similarities), max(recall_similarities), max(precision_similarities)])
    index_atom = np.argmax(scores[score_id])
    return f1_similarities[index_atom], recall_similarities[index_atom], precision_similarities[index_atom], index_atom

def get_grid_samples(dataset, label_inside, label_whole, masks, bitmaps, neuron_fires, number_samples, mask_shape, device, starting_percentage):
    
    label_mask = mask_utils.get_formula_mask(label_inside, masks)
    samples_comp2_atom = get_active_samples_neuron(neuron_fires, label_inside, label_whole, masks, device)
    positive_indices = torch.nonzero(samples_comp2_atom).flatten()
    #selected_indicees = positive_indices[:number_samples]
    fire_cov_per_sample = bitmaps.sum(dim=1) / bitmaps.shape[1]
    # Select the positive indices with at least 30% of coverage if possible
    selected_indices = positive_indices[fire_cov_per_sample[positive_indices] > starting_percentage]
    while len(selected_indices) < number_samples and starting_percentage > 0:
        starting_percentage -= 0.05
        selected_indices = positive_indices[fire_cov_per_sample[positive_indices] > starting_percentage]

    if len(selected_indices) < number_samples:
        remaining_samples = number_samples - len(selected_indices)
        sorted_indices = torch.argsort(fire_cov_per_sample[positive_indices], descending=True)
        selected_indices = torch.cat((selected_indices, sorted_indices[:remaining_samples]))
    selected_indices = selected_indices[:number_samples]

    selected_images = []
    selected_masks = []

    for index in selected_indices:
        path_image = dataset[index]['file_name']
        image =  torchvision.io.decode_image(torchvision.io.read_file(path_image), mode=torchvision.io.image.ImageReadMode.RGB)
        image = torchvision.transforms.functional.resize(image, mask_shape)
        selected_images.append(image)
        mask = ~label_mask[index]
        selected_masks.append(mask)
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
    selected_images = []
    selected_masks = []

    for index in selected_indices:
        path_image = dataset[index]['file_name']
        image =  torchvision.io.decode_image(torchvision.io.read_file(path_image), mode=torchvision.io.image.ImageReadMode.RGB)
        image = torchvision.transforms.functional.resize(image, mask_shape)
        selected_images.append(image)
        mask = ~bitmaps[index]
        selected_masks.append(mask)
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

def get_exclusive_mask_atom(atom, formula, masks, neuron_bitmap, device):
    mask_atom = mask_utils.get_formula_mask(atom, masks).to(
        device)
    mask_formula = mask_utils.get_formula_mask(formula, masks).to(
        device)
    exclusive_mask = mask_atom & ~mask_formula & neuron_bitmap
    return exclusive_mask



def compute_extra_thresholds(label_quantities, num_hits):
    """Compute the maximum extra space that can be used for the label."""
    unique_intersection = heuristic_utils.get_quantity(
        label=None, concepts_quantities=label_quantities,
        quantity_name='unique_intersection', quantity_type='max',
        quantity_scope='sum'
    )
    unique_extras = heuristic_utils.get_quantity(
        label=None, concepts_quantities=label_quantities,
        quantity_name='unique_extras', quantity_type='max',
        quantity_scope='sum'
    )

    general_threshold = (num_hits*(num_hits + unique_extras - unique_intersection))/ unique_intersection
    specific_threshold = (unique_extras*(num_hits-unique_intersection)+num_hits*(num_hits-2*unique_intersection))/ unique_intersection

    return general_threshold, specific_threshold


def reduce_by_extra_threshold(frontier, threshold, best_label, label_mapping, heuristic_info, max_size_mask, disjoint_info):
    new_frontier = []
    for node in frontier:
        label_quantities = optimal_heuristic.get_esti_quantities(optimal_heuristic.sum_heuristic,
            node, label_mapping=label_mapping, heuristic_info=heuristic_info, max_size_mask=max_size_mask, disjoint_info=disjoint_info, debug=False)
        unique_extras = heuristic_utils.get_quantity(
            label=None, concepts_quantities=label_quantities,
            quantity_name='unique_extras', quantity_type='max',
            quantity_scope='sum'
        )
        if unique_extras <= threshold:
            new_frontier.append(node)
    return new_frontier


