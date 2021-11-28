import argparse
import glob
import os
import shutil

import numpy as np
from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The new files are symlinks from the files in source folder.
    New folders are named train, val and test. Records are split into folders in a 80:10:10 ratio (train:val:test).

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    train_dir = os.path.join(destination, 'train')
    val_dir = os.path.join(destination, 'val')
    test_dir = os.path.join(destination, 'test')

    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = np.array(glob.glob(os.path.join(source, 'segment-*.tfrecord')))
    np.random.shuffle(files)

    train_paths, remaining_paths = np.split(files, [int(len(files) * 0.9)])
    val_paths, test_paths = np.split(remaining_paths, [int(len(remaining_paths) * 0.5)])

    [os.symlink(path, os.path.join(train_dir, os.path.basename(path))) for path in train_paths]
    [os.symlink(path, os.path.join(val_dir, os.path.basename(path))) for path in val_paths]
    [os.symlink(path, os.path.join(test_dir, os.path.basename(path))) for path in test_paths]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)