import sys
from io import StringIO
import fire
from ..config.base_config import MasterConfig
from ..config import base_config
import inspect


def choose_config(config_str):
  # Create config object
  available_configs = get_available_configs()
  try:
    config_class = available_configs[config_str]
    config_obj = config_class(config_str)
  except KeyError as e:
    err_msg = f"Config class '{config_str}' wasn't found. Feel free to create it as a new config class or use one of the existing ones -> {set(available_configs)}"
    raise KeyError(err_msg) from e

  # Overwrite parameters via optional input flags
  config_obj = overwrite(config_obj)

  # Freezes config
  if config_obj.freeze_config:
    config_obj.freeze()
  return config_obj


def get_available_configs():
  available_configs = {}
  for name, obj in inspect.getmembers(base_config):
    if inspect.isclass(obj) and issubclass(obj, MasterConfig):
      available_configs[name] = obj
  available_configs.pop('MasterConfig')
  return available_configs


def overwrite(config_obj):
  ''' Overwrites parameters with input flags. Function is needed for the
  convenience of specifying parameters via a combination of the config classes
  and input flags. '''
  class NullIO(StringIO):
    def write(self, txt):
      pass

  def parse_unknown_flags(**kwargs):
    return kwargs

  sys.stdout = NullIO()
  extra_arguments = fire.Fire(parse_unknown_flags)
  sys.stdout = sys.__stdout__

  for key, val in extra_arguments.items():
    if key not in vars(config_obj):
      err_str = f"The input parameter '{key}' isn't allowed. It's only possible to overwrite attributes that exist in the DefaultConfig class. Add your input parameter to the default class or catch it before this message"
      raise NotImplementedError(err_str)
    setattr(config_obj, key, val)

  return config_obj