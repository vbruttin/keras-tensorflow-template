from datetime import datetime
from pathlib import Path

import yaml
from munch import Munch, munchify

from utils.logger import Logger

logger = Logger()

Config = Munch


def get_config_from_yaml(yaml_file: Path) -> Config:
    with open(yaml_file, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)

    # convert the dictionary to a namespace
    return munchify(config_dict)


def process_config(yaml_file: Path) -> Config:
    config = get_config_from_yaml(yaml_file)
    yaml_string_name = yaml_file.stem
    dataloader_string_name = 'dl' + config.data_loader.name.rsplit('_', 1)[1].rsplit('.', 1)[0]
    model_string_name = 'm' + config.model.name.rsplit('_', 1)[1].rsplit('.', 1)[0]

    exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path("experiments") / exp_name
    tensorboard_dir = Path("experiment_notes") / exp_name
    checkpoint_dir = exp_dir / f"{model_string_name}-{yaml_string_name}-{dataloader_string_name}-checkpoints/"

    config.callbacks.exp_dir = str(exp_dir)
    config.callbacks.tensorboard_dir = str(tensorboard_dir)
    config.callbacks.checkpoint_dir = str(checkpoint_dir)

    logger.set_log_file(tensorboard_dir)

    return config
