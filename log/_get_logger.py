import json
from logging import getLogger, config

LOG_CONFIG = "/root/bocs/log/config.json"

with open(LOG_CONFIG, 'r') as f:
    log_conf = json.load(f)
    config.dictConfig(log_conf)


def get_logger(name: str):
    logger = getLogger(name)
    return logger


def get_sublogger(name: str):
    return getLogger("__main__").getChild(name)
