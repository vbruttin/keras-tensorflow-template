import argparse
import pickle
from pathlib import Path
from typing import Any

from .logger import Logger

logger = Logger()


def get_args() -> argparse.Namespace:

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-cd',
                           '--config-dir',
                           dest='config',
                           metavar='C',
                           default='None',
                           help='The Configuration file')

    argparser.add_argument('-e',
                           '--evaluate',
                           dest='evaluate',
                           action='store_true',
                           help='Option to evaluate on test data')

    return argparser.parse_args()


def save_as_pickle(obj: Any, path: Path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)
    logger.info(f"Saved {path.stem} successfully.")


def load_pickle(path_to_obj: Path) -> Any:
    with open(path_to_obj, 'rb') as file:
        obj = pickle.load(file)
    return obj
