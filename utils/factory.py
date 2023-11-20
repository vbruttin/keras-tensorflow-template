import importlib

from .logger import Logger

logger = Logger()


def create(cls: str):
    """expects a string that can be imported as with a module.class name"""
    module_name, class_name = cls.rsplit(".", 1)

    try:
        somemodule = importlib.import_module(module_name)
        cls_instance = getattr(somemodule, class_name)

    except ImportError as err:
        logger.error(f"Import error while creating instance: {err}")
        raise

    logger.debug(f"Created an instance of {cls} using create function.")

    return cls_instance
