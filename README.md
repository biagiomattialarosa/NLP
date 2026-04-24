## Pre-reququisite

```
pip install spacy-wordnet
pip install spacy==2.2.4
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
python -m nltk.downloader wordnet
pip install benepar
```

```
python3 nlp_utils/pre_process_nli.py --path=data/datasets/snli_1.0/snli_1.0_dev.txt
```
```
python3 nlp_utils/extract_wordvecs.py --path=data/datasets/snli_1.0/snli_1.0_dev.tok
```
```
python3 nlp_utils/annotate.py --path=data/datasets/snli_1.0/snli_1.0_dev.tok
```

# Scripts
python3 run_clustering.py --heuristic=beam_optimal --length=5 --num_clusters=0


run_clustering.py --dataset=ade20k_150_test_sem_seg --model=resnet18 --layer=layer4 --pretrained=places365 --beam_variant=new --length=1 --beam_limit=5 --num_clusters=5 --counter_variant=True

# LAST
python3 run_clustering.py --dataset=ade20k_150_test_sem_seg --model=resnet18 --layer=layer4 --pretrained=None --beam_variant=new --length=1 --beam_limit=5 --num_clusters=5 --counter_variant=False --random_units=50



python3 run_clustering.py --dataset=broden --model=resnet18 --layer=layer4 --pretrained=places365 --beam_variant=new --length=3 --beam_limit=5 --num_clusters=5 --counter_variant=False --random_units=50


# THRESHOLD
python3 run_clustering.py --beam_limit=10 --version=my --num_clusters=1 --quantile=0.005 --random_units=50 --heuristic=beam_optimal --length=5 --beam_variant=new --diff_threshold=0.8


python3 run_clustering.py --beam_limit=5 --version=my --dataset=broden --model=resnet18 --layer=layer4 --pretrained=places365 --num_clusters=1 --quantile=0.005 --random_units=50 --heuristic=beam_optimal --length=1 --beam_variant=new --diff_threshold=0.8


## AFTER EDITS
python3 run_clustering.py --beam_limit=20 --num_clusters=1 --quantile=0.005 --random_units=50 --heuristic=beam_optimal --length=5 --beam_variant=compound --neighbors_type=baseline


## first_n_interpretable
python3 run_clustering.py --beam_limit=10 --num_clusters=1 --quantile=0.005 --heuristic=beam_optimal --length=5 --beam_variant=compound --neighbors_type=baseline --first_n_interpretable_units=25

# Vision
python3 run_clustering.py --beam_limit=5 --num_clusters=1 --quantile=0.005 --heuristic=beam_optimal --length=3 --beam_variant=baseline --neighbors_type=baseline --dataset=broden --model=resnet18 --layer=layer4 --pretrained=places365

python3 run_clustering.py --beam_limit=5 --num_clusters=1 --quantile=0.005 --heuristic=beam_optimal --length=3 --beam_variant=compound --neighbors_type=baseline --dataset=broden --model=resnet18 --layer=layer4 --pretrained=places365