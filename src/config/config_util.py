import sys
from io import StringIO
import fire
from .base_config import *


def choose_config(config_str):
  try:
    config_obj = eval(config_str)(config_str)  # Create config object
  except NameError as e:
    err_msg = f"Config object '{config_str}' wasn't found. You can add more config classes"
    raise NameError(err_msg) from e

  # Overwrite parameters via optional input flags
  return overwrite(config_obj)


def overwrite(config_obj):
  ''' Overwrites parameters with input flags. Function is needed for the convenience of specifying parameters via a combination of the config classes and input flags. '''
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