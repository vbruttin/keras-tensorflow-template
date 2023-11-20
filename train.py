from pathlib import Path
from typing import Tuple

from base.base_data_loader import BaseDataLoader
from base.base_model import BaseModel
from base.base_trainer import BaseTrain
from evaluate import evaluate_test
from utils import factory
from utils.config import Config, process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

logger = Logger()


def setup_components(config: Config) -> Tuple[BaseDataLoader, BaseModel, BaseTrain]:
    create_dirs([config.callbacks.checkpoint_dir])

    logger.info('Create the data generator.')
    data_loader = factory.create(f"data_loader.{config.data_loader.name}")(config, config.exp.data_dir)
    logger.info('Create the model.')
    model = factory.create(f"models.{config.model.name}")(config)
    logger.info('Create the trainer')
    trainer = factory.create(f"trainers.{config.trainer.name}")(
        model,
        data_loader.get_train_data(),
        data_loader.get_validation_data(),
        config,
    )

    return data_loader, model, trainer


def main(config_arg, evaluate_enabled: bool):

    try:
        config_root = Path(config_arg)
        list_config_files = [f for f in config_root.iterdir() if f.suffix == '.yml']
        if not list_config_files:
            raise Exception('Empty config directory !')

        for config_file in list_config_files:
            logger.info(f'Processing Config File: {config_file.name}')
            config = process_config(config_file)

            data_loader, model, trainer = setup_components(config)
            logger.info('Start training the model.')
            trainer.train()
            logger.info(f"Max validation accuracy: {max(trainer.val_acc)}")

            if evaluate_enabled:
                logger.info('Evaluating..')
                path_to_models = config.callbacks.exp_dir
                logger.info(path_to_models)
                test_x, test_y = data_loader.get_test_data()
                best_model_path = evaluate_test(path_to_models, test_x, test_y)

    except Exception as e:
        logger.error(e)
        raise e


if __name__ == '__main__':
    args = get_args()
    main(args.config, args.evaluate)
