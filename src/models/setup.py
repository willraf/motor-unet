"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel
"""

import importlib
from torch.optim import lr_scheduler
from omegaconf import OmegaConf 

import logging


def find_model_using_name(model_name: str):
    """
    Import the module "src/models/[model_name].py".
    The class [ModelName] must exist inside and inherit from BaseModel.
    """
    module_name = f"src.models.{model_name}"
    try:
        modellib = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import module {module_name}") from e

    target_class_name = f"{model_name.capitalize()}"

    for name in dir(modellib):
        obj = getattr(modellib, name)
        if name.lower() == target_class_name.lower() and isinstance(obj, type): # issubclass(obj, BaseDataset):
            return obj

    raise ImportError(f"Expected class '{target_class_name}' in '{module_name}'.")


def setup_model(cfg: OmegaConf, device: str, mode: str):
    """Create a model given the configuration.

    This is the main interface between this package and train.py/validate.py
    """
    model_name = cfg.model.name
    model = find_model_using_name(model_name)
    instance = model(cfg)
    instance.to(device)
    logging.info("model [{0}] was created".format(type(instance).__name__))
    return instance