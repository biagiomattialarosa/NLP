
import re
import pandas as pd
import spacy
from tqdm import tqdm

def df_to_list(df):
    return list(zip(df['sentence1'], df['sentence2'], df['gold_label']))

def tokenize(string, nlp):
    return ' '.join([token.text for token in nlp.tokenizer(string)])

def tokenize_data(data):
    tok = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

    for (sent1, sent2, label) in tqdm(data):
        sent1_tok = tokenize(sent1, nlp)
        sent2_tok = tokenize(sent2, nlp)
        sent1_tok = re.sub(r'\s+', ' ', sent1_tok).strip()
        sent2_tok = re.sub(r'\s+', ' ', sent2_tok).strip()
        # These are specific rules to reproduce the same tokenization used in Compositional Explanation's repo: https://github.com/jayelm/compexp
        sent1_tok = sent1_tok.replace('3 - d','3-d')
        sent1_tok = sent1_tok.replace('18 - wheeler','18-wheeler')
        sent1_tok = sent1_tok.replace('12 - speed','12-speed')
        sent2_tok = sent2_tok.replace('3 - d','3-d')
        sent2_tok = sent2_tok.replace('18 - wheeler','18-wheeler')
        sent2_tok = sent2_tok.replace('12 - speed','12-speed')

        tok.append(sent1_tok)
        tok.append(sent2_tok)
    return tok

def generate_tok_from_text(text_path, out_path):
    df = pd.read_csv(text_path, sep='\t', keep_default_na=False)
    data = df_to_list(df)
    data_tok = tokenize_data(data)
    with open(out_path, 'w') as f:
        for line in data_tok:
            f.write(line + '\n')
    return data_tok

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='extract wordvecs',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path', required=True)

    args = parser.parse_args()

    # Generate .tok from raw dataset
    compact_data_tok = generate_tok_from_text(args.path, args.path.replace('.txt', '.tok'))



