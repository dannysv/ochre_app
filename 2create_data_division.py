#!/usr/bin/env python
import click
import codecs
import os
import json
import numpy as np

#from nlppln.utils import create_dirs, get_files

@click.command()
@click.argument('in_dir', type=click.Path(exists=True))
@click.argument('in_insertion_chance', type=click.STRING)
#@click.option('--out_dir', '-o', default=os.getcwd(), type=click.Path())
#@click.option('--out_name', default='datadivision.json')
def command(in_dir, in_insertion_chance):
    out_name = 'datadivision-'+ str(in_insertion_chance) + '.json'
    """Create a division of the data in train, test and validation sets.

    The result is stored to a JSON file, so it can be reused.
    """
    # TODO: make seed and percentages options
    SEED = 4
    TEST_PERCENTAGE = 10
    VAL_PERCENTAGE = 10

    #create_dirs(out_dir)

    #in_files = get_files(in_dir)
    in_files = os.listdir(in_dir)

    np.random.seed(SEED)
    np.random.shuffle(in_files)

    n_test = int(len(in_files)/100.0 * TEST_PERCENTAGE)
    n_val = int(len(in_files)/100.0 * VAL_PERCENTAGE)

    validation_texts = in_files[0:n_val]
    test_texts = in_files[n_val:n_val+n_test]
    train_texts = in_files[n_val+n_test:]

    division = {
        'train': [os.path.basename(t) for t in train_texts],
        'val': [os.path.basename(t) for t in validation_texts],
        'test': [os.path.basename(t) for t in test_texts]
    }

    #out_file = os.path.join(out_dir, out_name)
    with codecs.open(out_name, 'wb', encoding='utf-8') as f:
        json.dump(division, f, indent=4)


if __name__ == '__main__':
    command()
