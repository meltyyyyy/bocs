import json
from logging import getLogger, config

LOG_CONF = '/root/bocs/log_config.json'


def get_logger(name: str):
    with open(LOG_CONF, 'r') as f:
        log_conf = json.load(f)
    config.dictConfig(log_conf)
    logger = getLogger(name)
    return logger
