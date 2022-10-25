import configparser

CONFIG_PATH = '/root/bocs/config.ini'


def get_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH, encoding='utf-8')

    return config['DEFAULT']
