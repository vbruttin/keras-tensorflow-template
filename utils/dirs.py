from pathlib import Path
from typing import List


def create_dirs(dirs: List[str]):

    for dir_ in dirs:
        Path(dir_).mkdir(parents=True, exist_ok=True)
