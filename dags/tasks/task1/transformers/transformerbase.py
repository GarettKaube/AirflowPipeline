"""Module containing base class for Transformers
"""
import yaml

class Transformer:
    """ Base class for the other transformers that extract from google sheets.
    The extract and transform methods are implemented for each transformer.
    """
    def __init__(self, config_path) -> None:
        """Reads the config file so that a transformer instance
        can call get_sheets_by_category method.
        """
        self.yaml = self.read_config_file(config_path)


    def extract(self):
        raise NotImplementedError


    def transform(self):
        raise NotImplementedError


    def read_config_file(self, config_path):
        """ Loads the config_path yaml file

        Parameters
        ----------
        config_path: str
          Path to yaml file

        Returns
        -------
        dict
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    
    def get_sheets_by_category(self, category:str):
        """Accesses the category in self.yaml.

        Parameters
        ----------
        category  str
          category to be retrived from the loaded yaml file from config_path
        
        Returns
        -------
        list
        """
        return self.yaml[category]