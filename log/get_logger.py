import json
from logging import getLogger, config
import os
from utils import get_config

config_ = get_config()


def get_logger(name: str, filepath=None):
    with open(config_['log_config_path'], 'r') as f:
        log_conf = json.load(f)

    if filepath is not None:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        filename = config_['log_dir'] + filename + '.log'
        log_conf['handlers']['fileHandler']['filename'] = filename

    config.dictConfig(log_conf)
    logger = getLogger(name)
    return logger
