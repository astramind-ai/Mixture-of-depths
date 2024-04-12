import os
import json

from .models import *  # noqa: F401


class AutoMoDModelForCausalLM:
    @classmethod
    def load_model_conf(cls, path):
        if not os.path.exists(os.path.join(path, 'config.json')):
            raise ValueError(
                f"Cannot find a config file (config.json), at the specified directory please check the path")
        with open(os.path.join(path, 'config.json'), 'r') as f:
            model_conf = json.load(f)
            return model_conf['architectures'][0]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model_class_name = cls.load_model_conf(args[0])
        # Uses Mapping to get the model class name
        if not model_class_name:
            raise ValueError(
                f"The model {args[0]} was not found, this mean it has not been mapped yet, please open an issue on the github repository")

        # Get the model class using its name
        model_class = globals()[model_class_name]

        return model_class.from_pretrained(*args, **kwargs)

    @classmethod
    def from_config(cls, *args, **kwargs):
        model_class_name = cls.load_model_conf(args[0])
        # Uses Mapping to get the model class name
        if not model_class_name:
            raise ValueError(
                f"The model {args[0]} was not found, this mean it has not been mapped yet, please open an issue on the github repository")

        # Get the model class using its name
        model_class = globals()[model_class_name]

        return model_class.from_config(*args, **kwargs)