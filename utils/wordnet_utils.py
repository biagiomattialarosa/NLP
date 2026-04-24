
import json

import torch


WORDNET_CONFIGURATIONS = ['wordnet_step1', 'wordnet_step2', 'wordnet_step3']
IGNORE_NODES = ['equipment','substance','tracheophyte, vascular plant','piece of furniture, article of furniture, furniture','furnishing','barrier','art, fine art', 'surface','vessel','container', 'covering', 'device', 'way', 'path', 'craft','transport, conveyance',   'natural object', 'object, physical object', ]
ABSTRACT = ['attribute','form, shape','relation', 'impediment, impedimenta, obstructer, obstruction, obstructor','structure, construction', 'entity', 'matter', 'creation','grouping, group', 'artefact, artifact', 'physical entity', 'unit, whole', 'instrumentality, instrumentation, means', 'abstraction, abstract entity', 'measure, amount, quantity', 'being, organism','language unit, linguistic unit', 'consumer goods','durable goods, durables, consumer durables','animate thing, living thing','good, trade good, commodity','cause, causal agency, causal agent','part, portion, component, constituent, component part',]
IGNORE_NODES = IGNORE_NODES +  ABSTRACT

def update_synsets_mapping(wordnet, labels):
    wordnet_mapping, _ = build_synset_mapping_best_synset(wordnet, labels)
    for concept, synset in wordnet_mapping.items():
        wordnet_mapping[concept] = synset.id

def get_words_best_synset(wordnet, words):
    best_synset = (None, 0, 0) # This ensure that the default synset is the first one (most common)    
    manual_mapping = {'water': 'oewn-07951744-n', 'cushion':'oewn-03156166-n',
                      'van': 'oewn-04527465-n', 'plate': 'oewn-03965779-n',
                      'radiator': 'oewn-04047545-n'}
    if len(words) == 1 and words[0] in manual_mapping:
        return wordnet.synset(manual_mapping[words[0]]), 1, 1
    for word in words:
        candidate_synsets = wordnet.synsets(word, pos='n')        
        for synset in candidate_synsets:
            lemmas_synset = synset.lemmas()
            lemmas_synset = [lemma.lower() for lemma in lemmas_synset]
            common = list(set(lemmas_synset).intersection(words))
            percentage_words = len(common)/len(words)
            percentage_lemmas = len(common)/len(lemmas_synset)
            if percentage_words > best_synset[1]:
                best_synset = (synset, percentage_words, percentage_lemmas)
            elif percentage_words == best_synset[1] and percentage_lemmas > best_synset[2]:
                best_synset = (synset, percentage_words, percentage_lemmas)

    return best_synset

def load_synset_mapping(wordnet, mapping_file, configuration_name, labels):
    if configuration_name not in WORDNET_CONFIGURATIONS:
        # Build the mapping since wordnet has not been used before
        wordnet_mapping, _ = build_synset_mapping_best_synset(wordnet, labels)
        for concept, synset in wordnet_mapping.items():
            wordnet_mapping[concept] = synset.id
    else:
        with open(mapping_file, 'r') as f:
            wordnet_mapping = json.load(f)
    return wordnet_mapping

def get_synsets_mapping(wordnet, wordnet_mapping, labels):
    synsets_mapping = {}
    not_found = []
    for concept in labels:
        if concept in wordnet_mapping:
            synsets_mapping[concept] = wordnet.synset(wordnet_mapping[concept])
            continue
        elif ',' in concept:
            words = concept.split(',')
            words = [word.strip() for word in words] # Necessary to fix some typos
        else:
            words = [concept]

        mapping_available = False
        for w in words:
            if w in wordnet_mapping:
                synsets_mapping[concept] = wordnet.synset(wordnet_mapping[w])
                mapping_available = True
                break
        if not mapping_available:
            print(f"Mapping not available for {concept}")
            best_synset = get_words_best_synset(wordnet, words)
            if best_synset[0] is None:
                not_found.append(concept)
                continue    
            else:
                synsets_mapping[concept] = best_synset[0]

    return synsets_mapping, not_found 

def get_updated_configuration_name(configuration_name):
    if configuration_name in WORDNET_CONFIGURATIONS:
        step_number = int(configuration_name.split('_step')[-1])
        updated_configuration = f'wordnet_step{step_number+1}'
    else:
        updated_configuration = 'wordnet_step1'
    return updated_configuration

def build_synset_mapping_best_synset(wordnet, labels):
    mapping = {}
    not_found = []
    for concept in labels:
        if ',' in concept:
            words = concept.split(',')
            words = [word.strip() for word in words] # Necessary to fix some typos
        else:
            words = [concept]
        
        best_synset = get_words_best_synset(wordnet, words)
        if best_synset[0] is None:
            not_found.append(concept)
            continue    
        else:
            mapping[concept] = best_synset[0]
    return mapping, not_found


def build_synset_mapping(wordnet, labels):
    mapping = {}
    not_found = []
    for concept in labels:
        if ',' in concept:
            words = concept.split(',')
            words = [word.strip() for word in words] # Necessary to fix some typos
            candidate_synsets = []
            for word in words:
                candidate_synsets += wordnet.synsets(word, pos='n')
        else:
            candidate_synsets = wordnet.synsets(concept, pos='n')

        if len(candidate_synsets) == 0:
            not_found.append(concept)
            continue
        else:
            # Consider the first synset as the best one
            mapping[concept] = candidate_synsets
    return mapping, not_found


def unify(wordnet, synset1, synset2):
    mapping_unmeaningfull_concepts, _ = build_synset_mapping(wordnet, IGNORE_NODES)
    unmeaningfull_synsets = []
    for concept in mapping_unmeaningfull_concepts.keys():
        unmeaningfull_synsets.extend(mapping_unmeaningfull_concepts[concept])
    unmeaningfull_synsets = list(set(unmeaningfull_synsets))

    mapping_not_generalize, _ = build_synset_mapping(wordnet, DO_NOT_GENERALIZE)
    not_generalize_synsets = []
    for concept in mapping_not_generalize.keys():
        not_generalize_synsets.extend(mapping_not_generalize[concept])
    not_generalize_synsets = list(set(not_generalize_synsets))
    
    common_ancestors = []
    common_hypernyms = synset1.lowest_common_hypernyms(synset2)
    # Dummy case where one of the synsets is a hypernym of the other or one them cannot be generalized
    if synset1 in not_generalize_synsets:
        if synset1 in common_hypernyms:
            return [synset1]
        else:
            return []
    elif synset2 in not_generalize_synsets:
        if synset2 in common_hypernyms:
            return [synset2]
        else:
            return []
    
    # In all the other cases
    flag_removed = False
    for ancestor in common_hypernyms:
        # Remove common hypernyms that are unmeaningfull
        if ancestor not in unmeaningfull_synsets:
            common_ancestors.append(ancestor)
        elif 'vascular plant' in ancestor.lemmas():
            flag_removed = True
    # If the lowest common hypernyms are unmeaningfull, we need to find the common hypernyms at higher levels
    if len(common_ancestors) == 0:
        common_hypernyms = synset1.common_hypernyms(synset2)
        for ancestor in common_hypernyms:
            if ancestor not in unmeaningfull_synsets and ancestor not in common_ancestors:
                common_ancestors.append(ancestor)
    # Extract the lowest common ancestors in case there are multiple
    min_distance = torch.inf
    lowest_common_ancestor = []
    for ancestor in common_ancestors:
        distance = len(ancestor.shortest_path(synset1)) + len(ancestor.shortest_path(synset2))
        if distance < min_distance:
            min_distance = distance
            lowest_common_ancestor = [ancestor]
        elif distance == min_distance:
            lowest_common_ancestor.append(ancestor)
    common_ancestors = lowest_common_ancestor
    return common_ancestors

def get_mapping(unification_file, mapping_dir='data/cache/mapping'):
    mapping_file = f'{mapping_dir}/{unification_file}'
    with open(mapping_file, 'r') as f:
        merge_mapping = json.load(f)

    return merge_mapping

def unify_labels(unification_file, labels):
    mapping = get_mapping(unification_file)
    new_labels = []
    for label in labels:
        label = label.lower()
        label = label.strip()
        if label in mapping.keys():
            if mapping[label] not in new_labels:
                new_labels.append(mapping[label])
        else:
            new_labels.append(label)
    return new_labels

def merge_masks(unification_file, masks, segmentor):
    merge_mapping = get_mapping(unification_file)
    for label in merge_mapping.keys():
        merged_into = merge_mapping[label]
        # Check if the new labels is an abstraction not present in the segmentor concept labels
        if merged_into not in segmentor.concept_labels:
            masks.append(torch.zeros_like(masks[0]))
            segmentor.concept_labels.append(merged_into)
        # Get the index of the labels
        index_label = segmentor.concept_labels.index(label)
        index_merged_into = segmentor.concept_labels.index(merged_into)
        # Merge the masks
        masks[index_merged_into] = torch.logical_or(masks[index_label], masks[index_merged_into])
        # Zero out the mask to be merged
        masks[index_label] = torch.zeros_like(masks[index_label])
    return masks

def parse_unifition_files(configuration_name, unification_files):
    if configuration_name in WORDNET_CONFIGURATIONS:
        unification_files = list(unification_files)

        step_number = int(configuration_name.split('_step')[-1])
        if unification_files is None:
            raise ValueError("Unification file must be provided when using wordnet configurations")
        elif len(unification_files) != step_number:
            raise ValueError("Unification files must contain the files of the previous steps")
    return unification_files
def merge_multistep_wordnet_masks(configuration_name, dataset_name, unification_file, masks, segmentor):
    parsed_files = parse_unifition_files(configuration_name, unification_file)
    if configuration_name in WORDNET_CONFIGURATIONS:  
        for i in range(len(parsed_files)):
            masks = merge_masks(parsed_files[i], masks, segmentor)      
    return masks

def get_multistep_wordnet_labels(configuration_name, dataset_name, unification_file, labels):
    parsed_files = parse_unifition_files(configuration_name, unification_file)
    if configuration_name in WORDNET_CONFIGURATIONS:
        updated_labels = labels
        for i in range(len(parsed_files)):
            updated_labels = unify_labels(parsed_files[i], updated_labels)
        return updated_labels
    else:
        return labels