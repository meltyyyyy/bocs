import json
from logging import getLogger, config
import os

LOG_DIR = '/root/bocs/log/'
LOG_CONF = '/root/bocs/log/config.json'


def get_logger(name: str, filepath=None):
    with open(LOG_CONF, 'r') as f:
        log_conf = json.load(f)

    if filepath is not None:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        filename = LOG_DIR + filename + '.log'
        log_conf['handlers']['fileHandler']['filename'] = filename

    config.dictConfig(log_conf)
    logger = getLogger(name)
    return logger
