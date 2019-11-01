"""Classes to read YAML confguration files."""

import logging
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Helper class to access configuration values from a configration file.

    The configuraton file must be in YAML format.
    """

    # default values in case they are not defined in the config file
    _default = {
        "twitter_token": None,
        "twitter_secret": None,
        "consumer_key": None,
        "consumer_secret": None,
    }

    def __init__(self, filename: str):
        """
        Load the configuration values from a YAML fle.

        :param filename: the YAML file
        :raises FileNotFoundError: if the file cannot be read
        """
        self.root = None

        try:
            with open(filename, "r") as ymlfile:
                self.root = yaml.load(ymlfile, Loader=yaml.FullLoader)
        except FileNotFoundError as e:
            logger.error("Could not load config file %s", filename)
            raise e
        # data = yaml.dump(config, Dumper=yaml.CDumper)
        # print(data)

    @staticmethod
    def get_config_value(key: str, config_dict: Dict):
        """
        Get the value of a key in the specified configuration dictionary.

        :param key: the key to lookup
        :param config_dict: the dictionary to use (can be the root, or a sub-node
        :return: the value (if defined) or the default value (if defined)
        """
        # get a default value (if any)
        default_value = Config._default.get(key, None)

        logger.debug("Reading %s (default %s)", key, default_value)
        value = config_dict.get(key, default_value)
        logger.debug("Config value %s = %s", key, value)
        return value

    @staticmethod
    def _get_config_value(key: str, config_dict: Dict, default):
        try:
            value = config_dict[key]
        except KeyError:
            logger.debug("Using default for config value %s = %s", key, default)
            return default

        return value
