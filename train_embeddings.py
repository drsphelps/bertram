import os.path
import re
import sys
import log
import random
import argparse
from os import listdir
from os.path import isfile, join

from bertram import BertramWrapper
from transformers import BertForMaskedLM, BertTokenizer

logger = log.get_logger('root')


def get_more_examples(idioms, examples_folder, n=10, lower=True):
    examps = {}
    for idi in idioms:
        if '(' not in idi:
            # formatted = f'ID{"".join(idi.split())}ID'
            with open(f'{examples_folder}/{idi}.txt', 'r') as f:
                lines = f.readlines()
                if lower:
                    lines = [li.lower().replace('/', ' ') for li in lines]
                # lines = [re.sub(f'{idi}', f' {formatted} ', li) for li in lines]
                lines = [' '.join(li.split()) for li in lines]
                examps[idi] = random.sample(lines, k=min(len(lines), n))
    return examps


def get_idioms(examples_folder):
    onlyfiles = [f for f in listdir(examples_folder) if isfile(join(examples_folder, f))]
    return [f.split('.')[0] for f in onlyfiles]


def train_embeddings(bert_model, bertram_model, output_dir, examples_folder, no_examples):
    model = BertForMaskedLM.from_pretrained(bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    print(tokenizer.tokenize("I have 10 malignant neoplasms"))
    print(tokenizer.encode("I have 10 malignant neoplasms"))
    bertram = BertramWrapper(bertram_model, device='cuda')
    idioms = get_idioms(examples_folder)
    logger.info(f'Found {len(idioms)} en-idioms in {examples_folder}')

    examples = get_more_examples(idioms, examples_folder, no_examples)
    logger.info('Fetched examples from files')

    bertram.add_word_vectors_to_model(examples, tokenizer, model)
    logger.info('Added en-idioms and embeddings to bert model')

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(tokenizer.tokenize("I have 10 malignant neoplasms"))
    print(tokenizer.encode("I have 10 malignant neoplasms"))
    logger.info(f'Saved model and tokenizer to {output_dir}')


def main(args):
    parser = argparse.ArgumentParser("Train embeddings from a bertram model")

    parser.add_argument('--bert_model', default='bert-base-uncased', help='The underlying bert model')
    parser.add_argument('--bertram_model', help='The bertram model that will generate the embeddings')
    parser.add_argument('--output_dir',
                        help='The directory where the final model will be saved ({output_dir}-tokenizer'
                             ' will also be used to store the tokenizer for the model)')
    parser.add_argument('--examples_folder', help='The folder containing the examples')
    parser.add_argument('--no_examples', help='Number of examples used to train each idiom embedding')
    parser.add_argument('--lower', type=bool, help='Whether to lower case all examples, for uncased model')

    args = parser.parse_args(args)
    train_embeddings(args.bert_model, args.bertram_model, args.output_dir, args.examples_folder, int(args.no_examples))


if __name__ == "__main__":
    main(sys.argv[1:])
