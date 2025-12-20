"""
    Implement yaml file parser
"""
import yaml
from yaml.loader import SafeLoader


class YamlParser(object):
    def __init__(self, file_path):
        # Load the data
        with open(file_path) as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        # Convert the data to dictionary
        self.config = dict(data)

    @property
    def data(self):
        return self.config
