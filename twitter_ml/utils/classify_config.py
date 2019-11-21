"""Support for loading configuration values from external YAML files."""
import logging
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Wrapper around a YAML configuration file."""

    default = {"voting": Dict}

    def __init__(self, filename: str):
        """
        Create an instance of Config by loading values from the specified YAML filename.

        :param filename: the YAML file to load
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

    def get_config_value(self, key: str):
        """
        Get the value of the specific key.

        :param key: the key to configuration value to lookup
        :return the value of the specific key (if specified) or a default value (if specified)
        """
        default_value = Config.default.get(key, None)
        logger.debug("Reading %s (default %s)", key, default_value)

        value = self.root.get(key, default_value)
        logger.debug("Config value %s = %s", key, value)
        return value
