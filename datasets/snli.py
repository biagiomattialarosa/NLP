"""
SNLI dataset
"""


import os
import copy

import numpy as np
import spacy
from tqdm import tqdm
import torch
from scipy.spatial.distance import cdist


LABEL_STOI = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_ITOS = {v: k for k, v in LABEL_STOI.items()}
N_SENTENCE_FEATS = 2000  # how many of the most common sentence lemmas to keep

def pairs(x):
    """
    (max_len, batch_size, *feats)
    -> (max_len, batch_size / 2, 2, *feats)
    """
    if x.ndim == 1:
        return x.unsqueeze(1).view(-1, 2)
    else:
        return x.unsqueeze(2).view(x.shape[0], -1, 2, *x.shape[2:])


class SNLI:
    def __init__(self, path, split, vocab=None, max_data=None, unknowns=False):
        self.path = path
        self.unknowns = unknowns

        # Counterfactual SNLI
        self.c = "counterfactual" in self.path

        self.split = split
        if not self.c:
            assert self.split in {"train", "dev", "test"}
            self.text_path = os.path.join(self.path, f"snli_1.0_{self.split}.txt")
        else:
            self.text_path = os.path.join(self.path, "csnli.tsv")
        self.max_data = max_data
        self.label_stoi = LABEL_STOI
        self.label_itos = LABEL_ITOS
        self.spacy = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

        if vocab is None:
            self.stoi = {
                "UNK": 0,
                "PAD": 1,
            }
        else:
            self.stoi, self.itos = vocab

        self.labels = []
        self.raw_s1s = []
        self.raw_s2s = []
        self.s1s = []
        self.s2s = []
        self.s1lens = []
        self.s2lens = []
        n_skipped = 0
        with open(self.text_path, "r") as f:
            for i, line in enumerate(f):

                if i == 0:  # Header
                    continue

                if self.c:
                    s1, s2, label = line.strip().split("\t")
                else:
                    label, _, _, _, _, s1, s2, *_ = line.strip().split("\t")

                if label not in self.label_stoi:  # Hard example
                    assert label == "-"
                    if self.unknowns:
                        label_i = -1
                    else:
                        n_skipped += 1
                        continue
                else:
                    label_i = self.label_stoi[label]
                self.labels.append(label_i)

                if self.max_data is not None and i >= self.max_data:
                    break

                s1_doc = self.spacy(s1)
                s1_tok = [t.lower_ for t in s1_doc]
                s2_doc = self.spacy(s2)
                s2_tok = [t.lower_ for t in s2_doc]

                self.raw_s1s.append(s1_tok)
                self.raw_s2s.append(s2_tok)

                # Add to vocab
                s1_ns = []
                for tok in s1_tok:
                    if vocab is None and tok not in self.stoi:
                        # Build the vocab
                        self.stoi[tok] = len(self.stoi)
                    s1_ns.append(self.stoi.get(tok, 0))

                # Add to vocab
                s2_ns = []
                for tok in s2_tok:
                    if vocab is None and tok not in self.stoi:
                        # Build the vocab
                        self.stoi[tok] = len(self.stoi)
                    s2_ns.append(self.stoi.get(tok, 0))

                self.s1s.append(np.array(s1_ns))
                self.s1lens.append(len(s1_ns))
                self.s2s.append(np.array(s2_ns))
                self.s2lens.append(len(s2_ns))

        if vocab is None:
            self.itos = {v: k for k, v in self.stoi.items()}

    def __getitem__(self, i):
        s1 = torch.as_tensor(self.s1s[i])
        s1len = self.s1lens[i]

        s2 = torch.as_tensor(self.s2s[i])
        s2len = self.s2lens[i]

        label = self.labels[i]

        return s1, s1len, s2, s2len, label

    def __len__(self):
        return len(self.s1s)
    

"""
Dataset for analysis of features
"""


import torch
import numpy as np


PAD_IDX = 1
UNK_TOKEN = "UNK"


class AnalysisDataset:
    def __init__(self, text_raw, fields, path, split):
        """
        Initialize an analysis dataset from the given text file and vocab
        (`fields`) file which gives word-level annotations.

        Closed categories are categories whose # feats is constant w.r.t. dataset size.

        Open categories are those whose # feats grows roughly linear in the dataset
        size (e.g. lemmas). We keep these as separate categories and do not
        automatically search for these.
        """
        self.words = []
        self.fs = []
        self.mfs = []
        self.lengths = []
        self.fields = fields

        if "src" in fields:  # onmt-style fields
            self.ignore_tokens = {
                self.fields["src"].base_field.pad_token,
                self.fields["src"].base_field.eos_token,
            }
            self.stoi = self.fields["src"].base_field.vocab.stoi
            self.itos = self.fields["src"].base_field.vocab.itos
        else:
            self.ignore_tokens = {
                "PAD",
            }
            self.stoi = self.fields["stoi"]
            self.itos = self.fields["itos"]

        self.pos_stoi = {}

        first_line = text_raw[0]
        self.byte_fmt = isinstance(first_line, bytes)
        if self.byte_fmt:
            first_line = first_line.decode("utf-8")
        # FIXME: closed feats should be part of the shared vocab (pos:)...etc.
        (
            self.cs,
            self.cnames,
            self.ctypes,
            self.ocnames,
            self.ccnames,
            self.mcnames,
        ) = self.compute_cat_names(first_line)

        self.text = text_raw[1:]
        self.targets = self.load_labels(path, split)
        self.target_itos = LABEL_ITOS

        # Mapping legend:
        # feats (short: f) are the individual values
        # cats (short: c) are the categories (e.g. POS, Lemma, Synset)
        # Mappings - feat names to individual features
        self.cnames2fis = {cn: set() for cn in self.cnames}
        self.parse_docs()
        self.fis2cnames = {}
        for cname, fis in self.cnames2fis.items():
            for fi in fis:
                self.fis2cnames[fi] = cname

        self.citos = dict(zip(self.cs, self.cnames))
        self.cstoi = dict(zip(self.cnames, self.cs))

        self.cis2fis = {
            self.cstoi[cname]: fis for cname, fis in self.cnames2fis.items()
        }
        self.fis2cis = {}
        for ci, fis in self.cis2fis.items():
            for fi in fis:
                self.fis2cis[fi] = ci

        self.n_cs = len(self.cnames)
        self.n_fs = len(self.fstoi)
        self.cfis = []
        self.ofis = []
        self.mfis = []
        for cname, fis in self.cnames2fis.items():
            ctype = self.ctypes[cname]
            if ctype == "open":
                self.ofis.extend(fis)
            elif ctype == "multi":
                self.mfis.extend(fis)
            else:
                self.cfis.extend(fis)

        for ws, fs, mfs in self.docs:
            self.words.append(np.array(ws))
            self.fs.append(np.array(fs))
            self.mfs.append(mfs)
            self.lengths.append(len(ws))

    def load_labels(self, path, split):
        self.path = path

        # Counterfactual SNLI
        self.c = "counterfactual" in self.path

        self.split = split
        if not self.c:
            assert self.split in {"train", "dev", "test"}
            self.text_path = os.path.join(self.path, f"snli_1.0_{self.split}.txt")
        else:
            self.text_path = os.path.join(self.path, "csnli.tsv")

        labels = []
        with open(self.text_path, "r") as f:
            for i, line in enumerate(f):

                if i == 0:  # Header
                    continue

                if self.c:
                    s1, s2, label = line.strip().split("\t")
                else:
                    label, _, _, _, _, s1, s2, *_ = line.strip().split("\t")
                # Assert there is correspondence with the loaded text
                if label not in LABEL_STOI:  # Hard example
                    label = str(len(LABEL_STOI))
                else:
                    label_i = LABEL_STOI[label]
                labels.append(label_i)
        return labels

                
    def parse_docs(self):
        """
        Generator to parse documents, tokens, and feats
        """
        self.docs = []
        self.fstoi = {UNK_TOKEN: 0} # feature string to id
        self.fitos = {0: UNK_TOKEN} # feature id to string
        self.idx2multi = {}
        self.multi2idx = {}
        for line in self.text:
            line = line.strip()
            if self.byte_fmt:
                line = line.decode("utf-8")
            doc_words = []
            doc_feats = []
            doc_multifeats = []
            for tok in line.split(" "):
                word, *feats = tok.split("|")
                word_n = self.stoi.get(word.lower(), self.stoi["UNK"])
                feats = dict(zip(self.cnames, feats))
                feats_p = []
                multifeats_p = []
                for fn, f in feats.items():
                    if self.is_multi(fn):
                        fs = f.split(";")
                        fs_n = []
                        for f in fs:
                            # First assign global feature id
                            f = f"{fn}:{f}"
                            if f not in self.fstoi:
                                new_n = len(self.fstoi)
                                self.fstoi[f] = new_n
                                self.fitos[new_n] = f
                            f_n = self.fstoi[f]
                            # Next map it to a one hot index
                            if f_n not in self.multi2idx:
                                new_n = len(self.multi2idx)
                                self.multi2idx[f_n] = new_n
                                self.idx2multi[new_n] = f

                            fs_n.append(f_n)
                            self.cnames2fis[fn].add(f_n)
                        multifeats_p.append(fs_n)
                    else:
                        if fn == "lemma":
                            # Lowercase lemmas
                            f = f.lower()
                        if not f:
                            f = UNK_TOKEN
                        else:
                            f = f"{fn}:{f}"
                        if f not in self.fstoi:
                            new_n = len(self.fstoi)
                            self.fstoi[f] = new_n
                            self.fitos[new_n] = f
                        f_n = self.fstoi[f]
                        feats_p.append(f_n)
                        # Update feature name
                        self.cnames2fis[fn].add(f_n)
                doc_words.append(word_n)
                doc_feats.append(feats_p)
                doc_multifeats.append(multifeats_p)
            self.docs.append((doc_words, doc_feats, doc_multifeats))

    def is_open(self, feat):
        return self.ctypes[feat] == "open"

    def is_closed(self, feat):
        return self.ctypes[feat] == "closed"

    def is_multi(self, feat):
        return self.ctypes[feat] == "multi"

    def __getitem__(self, i):
        ws = torch.as_tensor(self.words[i]).unsqueeze(1)
        fs = torch.as_tensor(self.fs[i])
        # These are multi-hot vectors with width equal to the number of
        # multi-hot features.
        mfs = torch.zeros((len(self.words[i]), len(self.mfis)), dtype=torch.bool)
        for w_i, word_mfeatures in enumerate(self.mfs[i]):
            for cat_f in word_mfeatures:
                for mfi in cat_f:
                    # Map to index
                    mfi_index = self.multi2idx[mfi]
                    mfs[w_i, mfi_index] = 1
        ls = torch.tensor(self.lengths[i])
        return ws, fs, mfs, ls, i

    def compute_cat_names(self, feat_str):
        cat_names = feat_str.strip().split("|")
        cat_names, ocs = zip(*[fn.split(";") for fn in cat_names])
        for oc in ocs:
            if oc not in {"open", "closed", "multi"}:
                raise RuntimeError(f"Invalid tag {oc}")
        ocn = [cn for cn, oc in zip(cat_names, ocs) if oc == "open"]
        ccn = [cn for cn, oc in zip(cat_names, ocs) if oc == "closed"]
        mcn = [cn for cn, oc in zip(cat_names, ocs) if oc == "multi"]
        cat_types = dict(zip(cat_names, ocs))
        cat_is = list(range(len(cat_names)))
        return cat_is, cat_names, cat_types, ocn, ccn, mcn

    def __len__(self):
        return len(self.words)

    def to_text(self, sentence):
        words = []
        for tok in sentence:
            word = self.itos[tok]
            if word == "<unk>":
                word = "UNK"
            if word not in self.ignore_tokens:
                words.append(word)
        return words

    def to_text_batch(self, src):
        batch_size = src.shape[1]
        batch = []
        for i in range(batch_size):
            sentence = src[:, i, :].squeeze(1)
            words = []
            for tok in sentence.cpu().numpy():
                word = self.itos[tok]
                if word == "<unk>":
                    word = UNK_TOKEN
                if word not in self.ignore_tokens:
                    words.append(word)
            batch.append(words)
        return batch

    def feats_batch_to_text(self, feats_batch):
        output = []
        for sentence in feats_batch:
            for word in sentence:
                word_output = []
                for cat_f in word:
                    if self.name_feat(cat_f) != UNK_TOKEN:
                        word_output.append(self.name_feat(cat_f))
                    else:
                        word_output.append(UNK_TOKEN)
                output.append(word_output)
        return output
    
    def multi_feat_batch_to_text(self, feats_batch):
        output = []
        for sentence in feats_batch:
            for word in sentence:
                word_output = []
                for cat_index, is_true in enumerate(word):
                    if is_true:
                        word_output.append(self.multi_name_feat(cat_index))
                output.append(word_output)
        return output
    
    def name_feat(self, i):
        return self.fitos.get(i, UNK_TOKEN)
    
    def multi_name_feat(self, i):
        return self.idx2multi.get(i, UNK_TOKEN)
    
    def get_features(self, data_loader):  
        all_srcs = []
        all_feats = []
        all_multifeats = []
        all_idxs = []
        for src, src_feats, src_multifeats, src_lengths, idx in data_loader:
            # Memory bank - hidden states for each step
            with torch.no_grad():
                # Combine q/h pairs
                src_one = src.squeeze(2)
                src_one_comb = pairs(src_one)

            # Pack the sequence
            all_srcs.extend(list(np.transpose(src_one_comb.cpu().numpy(), (1, 2, 0))))
            all_feats.extend(
                list(np.transpose(pairs(src_feats).cpu().numpy(), (1, 2, 0, 3)))
            )
            all_multifeats.extend(
                list(np.transpose(pairs(src_multifeats).cpu().numpy(), (1, 2, 0, 3)))
            )
            all_idxs.extend(list(pairs(idx).cpu().numpy()))

        all_feats = {"onehot": all_feats, "multi": all_multifeats}

        return all_srcs,  all_feats, all_idxs
    
    def print_sentence_at_idx(self, idx):
        real_idx = idx*2 # This is because in the bitmaps/dataloader premise and hypothesis are concatenated
        pre_sentence = ''
        hyp_sentence = ''

        for tok in self.text[real_idx].split(" "):
            word, *feats = tok.split("|")
            pre_sentence += word + ' '
        for tok in self.text[real_idx+1].split(" "):
            word, *feats = tok.split("|")
            hyp_sentence += word + ' '
        return pre_sentence.strip(), hyp_sentence.strip(), self.target_itos[self.targets[idx]]

    def to_sentence(self, toks, feats, tok_feats_vocab=None, tokens_feats=True, tags_feats=True, overlap_feats=True, cls_feats=False):
        """
        Convert token-level feats to sentence feats

        # FORMAT token:
        HYP/PRE/OTH:feat_cat:feat_value
            HYP/PRE/OTH: this info cannot be encoded as separate concepts since the activation is one per sentence
                HYP/PRE - token appear in hypothesis/premise
                OTH - other feature (currently only overlap)
            feat_cat: tok or tag
                tok - word token (lemma)
                tag - part of speech tag
            feat_value - actual value of the feature


        """
        tokens = np.zeros(len(self.stoi), dtype=np.int64)
        encoder_uniques = []
        decoder_uniques = []
        #  both_uniques = []

        encoder_tag_uniques = []
        decoder_tag_uniques = []
        #  both_tag_uniques = []

        tag_i = self.cstoi["tag"]

        other_features = []
        oth_names = [
            ("overlap25", "overlap"),
            ("overlap50", "overlap"),
            ("overlap75", "overlap"),
        ]

        class_features = []
        class_names = [
            ("entailment", "class"),
            ("neutral", "class"),
            ("contradiction", "class"),
        ]

        for idx, (pair, featpair) in enumerate(zip(toks, feats["onehot"])):
            pair_counts = np.bincount(pair.ravel())
            tokens[: len(pair_counts)] += pair_counts

            enct = np.unique(pair[0])
            dect = np.unique(pair[1])

            encu = np.setdiff1d(enct, dect)
            decu = np.setdiff1d(dect, enct)
            both = np.intersect1d(enct, dect)
            encoder_uniques.append(enct)
            decoder_uniques.append(dect)
            #  both_uniques.append(both)

            # PoS
            enctag = np.unique(featpair[0, :, tag_i])
            dectag = np.unique(featpair[1, :, tag_i])

            enctag = enctag[enctag != -1]
            dectag = dectag[dectag != -1]

            #  enctagu = np.setdiff1d(enctag, dectag)
            #  dectagu = np.setdiff1d(dectag, enctag)
            #  bothtagu = np.intersect1d(enctag, dectag)

            encoder_tag_uniques.append(enctag)
            decoder_tag_uniques.append(dectag)
            #  both_tag_uniques.append(bothtagu)

            # Compute degree of overlap in tokens (gt 50%)
            overlap = len(both) / (len(encu) + len(decu) + 1e-5)
            # TODO: Do overlap at various degrees
            other_features.append(
                (
                    overlap > 0.25,
                    overlap > 0.5,
                    overlap > 0.75,
                )
            )

            class_features.append(
                (
                    self.targets[idx] == LABEL_STOI["entailment"],
                    self.targets[idx] == LABEL_STOI["neutral"],
                    self.targets[idx] == LABEL_STOI["contradiction"],
                )
            )

        SKIP = {"a", "an", "the", "of", ".", ",", "UNK", "PAD"}
        if tok_feats_vocab is None:
            for s in SKIP:
                if s in self.stoi:
                    tokens[self.stoi[s]] = 0

            # Keep top tokens, use as features
            tokens_by_count = np.argsort(tokens)[::-1]
            tokens_by_count = tokens_by_count[: N_SENTENCE_FEATS]

            # Create feature dict
            # Token features
            tokens_stoi = {}
            for prefix in ["pre", "hyp"]:
                if tokens_feats:
                    for t in tokens_by_count:
                        ts = self.itos[t]
                        t_prefixed = f"{prefix}:tok:{ts}"
                        tokens_stoi[t_prefixed] = len(tokens_stoi)
                if tags_feats:
                    # PoS
                    for pos_i in self.cnames2fis["tag"]:
                        pos = self.fitos[pos_i].lower()
                        assert pos.startswith("tag:")
                        pos_prefixed = f"{prefix}:{pos}"
                        tokens_stoi[pos_prefixed] = len(tokens_stoi)

            # Other features
            if overlap_feats:
                for oth, oth_type in oth_names:
                    oth_prefixed = f"oth:{oth_type}:{oth}"
                    tokens_stoi[oth_prefixed] = len(tokens_stoi)

            tokens_itos = {v: k for k, v in tokens_stoi.items()}

            tok_feats_vocab = {
                "itos": tokens_itos,
                "stoi": tokens_stoi,
            }


        # ADD cls vocab if enabled
        if cls_feats:
            for cls, cls_type in class_names:
                cls_prefixed = f"cls:{cls_type}:{cls}"
                if cls_prefixed not in tok_feats_vocab["stoi"]:
                    tok_feats_vocab["stoi"][cls_prefixed] = len(tok_feats_vocab["stoi"])
                    tok_feats_vocab["itos"][len(tok_feats_vocab["itos"])] = cls_prefixed

        # Binary mask - encoder/decoder
        token_masks = np.zeros((len(toks), len(tok_feats_vocab["stoi"])), dtype=np.bool)
        for i, (encu, decu, enctagu, dectagu, oth, cls) in enumerate(
            zip(
                encoder_uniques,
                decoder_uniques,
                encoder_tag_uniques,
                decoder_tag_uniques,
                other_features,
                class_features,
            )
        ):
            # Tokens
            if tokens_feats:
                for prefix, toks in [("pre", encu), ("hyp", decu)]:
                    for t in toks:
                        ts = self.itos[t]
                        t_prefixed = f"{prefix}:tok:{ts}"
                        if t_prefixed in tok_feats_vocab["stoi"]:
                            ti = tok_feats_vocab["stoi"][t_prefixed]
                            token_masks[i, ti] = 1

            # PoS
            if tags_feats:
                for prefix, tags in [("pre", enctagu), ("hyp", dectagu)]:
                    for t in tags:
                        ts = self.fitos[t].lower()
                        t_prefixed = f"{prefix}:{ts}"
                        assert t_prefixed in tok_feats_vocab["stoi"]
                        ti = tok_feats_vocab["stoi"][t_prefixed]
                        token_masks[i, ti] = 1

            # Other features
            assert len(oth) == len(oth_names)
            if overlap_feats:
                for (oth_name, oth_type), oth_u in zip(oth_names, oth):
                    oth_prefixed = f"oth:{oth_type}:{oth_name}"
                    oi = tok_feats_vocab["stoi"][oth_prefixed]
                    token_masks[i, oi] = oth_u
            if cls_feats:
                for (cls_name, cls_type), cls_u in zip(class_names, cls):
                    cls_prefixed = f"cls:{cls_type}:{cls_name}"
                    ci = tok_feats_vocab["stoi"][cls_prefixed]
                    token_masks[i, ci] = cls_u
        return token_masks, tok_feats_vocab
    
    def load_vecs(self, path):
        vecs = []
        vecs_stoi = {}
        vecs_itos = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                tok, *nums = line.split(" ")
                nums = np.array(list(map(float, nums)))

                assert tok not in vecs_stoi
                new_n = len(vecs_stoi)
                vecs_stoi[tok] = new_n
                vecs_itos[new_n] = tok
                vecs.append(nums)
        vecs = np.array(vecs)
        return vecs, vecs_stoi, vecs_itos
    
    def add_old_neighbors_feat(self, vocab, feats):
        """
        Get neighbors of lemma given glove vectors.
        """

        new_vocab = copy.deepcopy(vocab)
        for word, index in vocab['stoi'].items():
            # We removed this part to simulate the replication of
            # the bugged neighbors (so we can't filterby vecs)
            # They compute neighbors only for non_zero_concepts for each neuron

            # part, pos, lemma = word.split(":", maxsplit=2)
            # if lemma not in VECS_STOI:
            #     # No neighbors
            #     continue
            # lemma_i = VECS_STOI[lemma]
            # lvec = VECS[lemma_i][np.newaxis]
            # dists = cdist(lvec, VECS, metric="cosine")[0]
            # # first dist will always be the vector itself
            # nearest_i = np.argsort(dists)[1 : neighborhood_size + 1]
            # nearest = [VECS_ITOS[i] for i in nearest_i]
            
            # part_neighbors = [f"{part}:{pos}:{word}" for word in nearest]
            # neighbor_idx = [
            #     vocab["stoi"][word_neighbor]
            #     for word_neighbor in part_neighbors
            #     if word_neighbor in vocab["stoi"]
            # ]
            neighbor_idx = [vocab["stoi"][word]]
            neighbor_idx = np.array(list(set(neighbor_idx)))
            neighbors_mask = np.logical_or.reduce(feats[:, neighbor_idx], 1)
            # Add mask to feats and update vocab
            new_feat_name = f"neighbor:{word}"
            if new_feat_name not in new_vocab['stoi']:
                new_index = len(new_vocab['stoi'])
                new_vocab['stoi'][new_feat_name] = new_index
                new_vocab['itos'][new_index] = new_feat_name
            feats = np.concatenate(
                [feats, neighbors_mask[:, np.newaxis]], axis=1
            )                
        return feats, new_vocab
    
    def add_neighbors_feat(self, vocab, feats, vecpath, neighborhood_size=5):
        """
        Get neighbors of lemma given glove vectors.
        """
        print(f"Loading vecs generated using {vecpath}...")
        VECS, VECS_STOI, VECS_ITOS = self.load_vecs(vecpath)
        new_vocab = copy.deepcopy(vocab)
        constraints = [ [ ] for _ in range(len(vocab['stoi']))]
        covered_lemmas = 0
        total_lemmas = len(vocab['stoi'])
        for word, index in vocab['stoi'].items():
            part, pos, lemma = word.split(":", maxsplit=2)
            if pos == "tag" or pos == "overlap":
                # Skip pos tags and overlaps for which we don't want neighbors
                continue
            if lemma not in VECS_STOI:
                # No neighbors
                continue
            else:
                covered_lemmas += 1
            lemma_i = VECS_STOI[lemma]
            lvec = VECS[lemma_i][np.newaxis]
            dists = cdist(lvec, VECS, metric="cosine")[0]
            # first dist will always be the vector itself
            nearest_i = np.argsort(dists)[1 : neighborhood_size + 1]
            nearest = [VECS_ITOS[i] for i in nearest_i]

            # words_to_check = ["male", "female"]
            # if lemma == "human":
            #     nearest = ["human", "person", "people"]
            #     nearest = ["human", "person", "people", "child", "children", "kids", "kid", "humans"]
            # for word_check in words_to_check:
            #     if word_check == lemma:
            #         print(f"Word: {word}, Nearest: {nearest}")
            #         for word_neighbor in nearest:
            #             if word_neighbor not in words_to_check:
            #                 print(f"    Neighbor: {word_neighbor}")

            part_neighbors = [f"{part}:{pos}:{word}" for word in nearest]
            neighbor_idx = [
                vocab["stoi"][word_neighbor]
                for word_neighbor in part_neighbors
                if word_neighbor in vocab["stoi"]
            ]

            neighbor_idx.append(vocab["stoi"][word])
            neighbor_idx = np.array(list(set(neighbor_idx)))
            if len(neighbor_idx) > 1:
                neighbors_mask = np.logical_or.reduce(feats[:, neighbor_idx], 1)
                for index_neighbor in neighbor_idx:
                    neighbor_word = vocab['itos'][index_neighbor]
                    neighbor_neighbor_feat_name = f"neighbor:{neighbor_word}"

                    neighbor_neighbor_index = new_vocab['stoi'].get(neighbor_neighbor_feat_name, None)

                    # Remove the neighbor feature if it is identical to one of the existing neighbor features for the neighbors that generated it 
                    if  neighbor_neighbor_index is not None and np.all(neighbors_mask == feats[:, neighbor_neighbor_index]):
                        # If one of the neighbor masks is identical to the
                        # combined one, we can skip adding the new feature
                        neighbors_mask = None
                        # print(f"    Adding neighbor feature for word: {neighbor_word}")
                        # print(f"    Neighbor feature: {neighbor_neighbor_feat_name}")
                        # print("    Skipping addition of neighbor feature since identical to existing one.")
                        # print(f"    Similar {neighbor_neighbor_feat_name} and {word} neighbor features.")
                        # print("    ---------------------")
                        break
                    # if  neighbor_neighbor_index is not None and np.all(neighbors_mask == feats[:, index_neighbor]):
                    #     # If one of the neighbor masks is identical to the
                    #     # combined one, we can skip adding the new feature
                    #     # print(f" Skipping addition of neighbor feature for word: {word} since it converges to existing feature for {neighbor_word}.")
                    #     # print(feats[:, index_neighbor])
                    #     # print(feats[:, index_neighbor].sum())
                    #     # print(neighbors_mask)
                    #     # print(neighbors_mask.sum())
                    #     # print(nearest)
                    #     neighbors_mask = None

                        break
                if neighbors_mask is None or np.sum(neighbors_mask) == 0:
                    continue
                # Add mask to feats and update vocab
                new_feat_name = f"neighbor:{word}"
                if new_feat_name not in new_vocab['stoi']:
                    new_index = len(new_vocab['stoi'])
                    new_vocab['stoi'][new_feat_name] = new_index
                    new_vocab['itos'][new_index] = new_feat_name

                    ### ADD CONSTRAINTS ###
                    constraints[index].append(new_index)
                    # A word can't be associated to neighbor(word)
                    constraints.append(new_index)
                    # A neighbor(word) can't be associated the neighbors that generated it
                    constraints[new_index] = neighbor_idx.tolist()
                    assert len(constraints) == len(new_vocab['stoi']), f"Constraints length mismatch with vocab size {len(new_vocab['stoi'])} vs {len(constraints)}"
                # if lemma == "human":
                #     print(f"Adding neighbor feature for word: {word}. Name: {new_feat_name}")
                feats = np.concatenate(
                    [feats, neighbors_mask[:, np.newaxis]], axis=1
                )
        print(f"Finished adding neighbors features. Added total of {covered_lemmas} lemmas over {total_lemmas}.")
        # exit()
        # print(f"Artificially adding the color concept")
        # #colors = ['red', 'blue', 'yellow', 'purple', 'orange', 'pink', 'green', 'brown', 'black', 'white', 'grey', 'beige']
        # colors = ['red', 'green', 'white', 'blue', 'black']
        # pre_color = "pre:tok:specific_color"
        # hyp_color = "hyp:tok:specific_color"

        # # compute color masks for premise and hypothesis based on the presence of any color token in the sentence, and add to feats and vocab
        # pre_color_mask = np.zeros(feats.shape[0], dtype=np.bool)
        # hyp_color_mask = np.zeros(feats.shape[0], dtype=np.bool)
        # for color in colors:
        #     color_feat_name_pre = f"pre:tok:{color}"
        #     if color_feat_name_pre in vocab['stoi']:
        #         color_index_pre = vocab['stoi'][color_feat_name_pre]
        #         pre_color_mask = np.logical_or(pre_color_mask, feats[:, color_index_pre])

        #     color_feat_name_hyp = f"hyp:tok:{color}"
        #     if color_feat_name_hyp in vocab['stoi']:
        #         color_index_hyp = vocab['stoi'][color_feat_name_hyp]
        #         hyp_color_mask = np.logical_or(hyp_color_mask, feats[:, color_index_hyp])

        # # Add color features to vocab and feats
        # for color_mask, color_feat_name in [(pre_color_mask, pre_color), (hyp_color_mask, hyp_color)]:
        #     if color_feat_name not in new_vocab['stoi']:
        #         new_index = len(new_vocab['stoi'])
        #         new_vocab['stoi'][color_feat_name] = new_index
        #         new_vocab['itos'][new_index] = color_feat_name
        #     else:
        #         raise RuntimeError(f"Color feature {color_feat_name} already in vocab, cannot add color feature.")
        #         new_index = new_vocab['stoi'][color_feat_name]
        #     feats = np.concatenate(
        #         [feats, color_mask[:, np.newaxis]], axis=1
        #     )
        # print("Artificially adding the my_man concept")
        # my_man = ['man', 'men', 'boy', 'boys', 'male']
        # my_woman = ['woman', 'women', 'girl','lady', 'girls', 'female']
        # pre_my_man = "pre:tok:my_man"
        # hyp_my_man = "hyp:tok:my_man"
        # pre_my_woman = "pre:tok:my_woman"
        # hyp_my_woman = "hyp:tok:my_woman"
        # pre_my_man_mask = np.zeros(feats.shape[0], dtype=np.bool)
        # hyp_my_man_mask = np.zeros(feats.shape[0], dtype=np.bool)
        # pre_my_woman_mask = np.zeros(feats.shape[0], dtype=np.bool)
        # hyp_my_woman_mask = np.zeros(feats.shape[0], dtype=np.bool)
        # for man_word in my_man:
        #     if man_word in vocab['stoi']:
        #         man_index = vocab['stoi'][man_word]
        #         pre_my_man_mask = np.logical_or(pre_my_man_mask, feats[:, man_index])
        #         hyp_my_man_mask = np.logical_or(hyp_my_man_mask, feats[:, man_index])

        # for woman_word in my_woman:
        #     if woman_word in vocab['stoi']:
        #         woman_index = vocab['stoi'][woman_word]
        #         pre_my_woman_mask = np.logical_or(pre_my_woman_mask, feats[:, woman_index])
        #         hyp_my_woman_mask = np.logical_or(hyp_my_woman_mask, feats[:, woman_index])
        # for my_man_mask, my_man_feat_name in [(pre_my_man_mask, pre_my_man), (hyp_my_man_mask, hyp_my_man)]:
        #     if my_man_feat_name not in new_vocab['stoi']:
        #         new_index = len(new_vocab['stoi'])
        #         new_vocab['stoi'][my_man_feat_name] = new_index
        #         new_vocab['itos'][new_index] = my_man_feat_name
        #     else:
        #         raise RuntimeError(f"My man feature {my_man_feat_name} already in vocab, cannot add my man feature.")
        #         new_index = new_vocab['stoi'][my_man_feat_name]
        #     feats = np.concatenate(
        #         [feats, my_man_mask[:, np.newaxis]], axis=1
        #     )
        # for my_woman_mask, my_woman_feat_name in [(pre_my_woman_mask, pre_my_woman), (hyp_my_woman_mask, hyp_my_woman)]:
        #     if my_woman_feat_name not in new_vocab['stoi']:
        #         new_index = len(new_vocab['stoi'])
        #         new_vocab['stoi'][my_woman_feat_name] = new_index
        #         new_vocab['itos'][new_index] = my_woman_feat_name
        #     else:
        #         raise RuntimeError(f"My woman feature {my_woman_feat_name} already in vocab, cannot add my woman feature.")
        #         new_index = new_vocab['stoi'][my_woman_feat_name]
        #     feats = np.concatenate(
        #         [feats, my_woman_mask[:, np.newaxis]], axis=1
        #     )

        return feats, new_vocab, constraints

    def keep_tokens_feat(self, vocab, feats, vecpath, neighborhood_size=5):
            """
            Get neighbors of lemma given glove vectors.
            """
            print(f"Loading vecs generated using {vecpath}...")
            VECS, VECS_STOI, VECS_ITOS = self.load_vecs(vecpath)
            new_vocab = copy.deepcopy(vocab)
            constraints = [ [ ] for _ in range(len(vocab['stoi']))]
            covered_lemmas = 0
            total_lemmas = len(vocab['stoi'])
            for word, index in vocab['stoi'].items():
                part, pos, lemma = word.split(":", maxsplit=2)
                if pos == "tag" or pos == "overlap":
                    # Skip pos tags and overlaps for which we don't want neighbors
                    print(f"Removing non-lemma feature {word} from vocab and feats.")
                    feats[ :, index] = np.zeros(feats.shape[0], dtype=np.bool)

            constraints = None
            print(f"Finished adding neighbors features. Added total of {covered_lemmas} lemmas over {total_lemmas}.")
            return feats, new_vocab, constraints