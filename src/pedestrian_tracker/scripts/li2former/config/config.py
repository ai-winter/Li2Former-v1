'''
* @file: config.py
* @breif: Configure class.
* @auther: Yang Haodong
* @update: 2023.2.16 
'''
import yaml
import json
import os
from pathlib import Path

class Config:
    '''
    * @breif: Configure class for loading/updating parameters.
    '''
    def __init__(self, config=None) -> None:
        self.config = config if config else {}

    def __call__(self, entry: str, default=None):
        return self.config.get(entry) or default

    def __str__(self):
        return json.dumps(self.config, indent=4)
        
    def load(self, config) -> None:
        '''
        * @breif: Load configure parameters.
        '''
        # reload yaml.Loader Class to recognize `include` key word
        class Loader(yaml.SafeLoader):
            def __init__(self, stream):
                self._root = os.path.split(stream.name)[0]
                super(Loader, self).__init__(stream)

            def include(self, node):
                filename = os.path.join(self._root, self.construct_scalar(node))
                with open(filename, 'r') as f:
                    return yaml.load(f, Loader)
        Loader.add_constructor('!include', Loader.include)

        # update from file
        if isinstance(config, str):
            with open(config, 'r', encoding="utf-8") as f:
                self.config = yaml.load(f, Loader=Loader)
        
        # update from dictionary
        if isinstance(config, dict):
            self.config.update(config)
        
        # update from Path object
        if isinstance(config, Path):
            with open(str(config), 'r', encoding="utf-8") as f:
                self.config = yaml.load(f, Loader=Loader)
    
    def sub(self, field_name: str) -> "Config":
        '''
        * @breif: Return a sub dict wrapped in Config().
        '''
        sub_dict = self.config.get(field_name)
        if isinstance(sub_dict, dict):
            return Config(sub_dict)
        return Config()


class AllConfig(Config):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        config_path = os.path.abspath(os.path.join(__file__, "../all_config.yaml"))
        self.load(config_path)


