import logging


class Logger:
    _instance = None

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance.logger = None
        return cls._instance

    def __init__(self, name: str = 'ExperimentLogger'):
        if not self._initialized:
            self.name = name

            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

            self._initialized = True

    def set_log_file(self, log_path):
        file_handler = logging.FileHandler(log_path.parent / f'{self.name}_{log_path.name}')
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def __getattr__(self, attr):
        return getattr(self.logger, attr)
