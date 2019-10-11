import visdom


def clear_envs(vis):
  [vis.close(env=env) for env in vis.get_env_list()]  # Kills wind
  # [vis.delete_env(env) for env in vis.get_env_list()] # Kills envs


def log_if_active(func):
  ''' Decorator which only calls logging function if logger is active '''
  def wrapper(self, *args, **kwargs):
    if self.log_data:
      func(self, *args, **kwargs)

  return wrapper


class Logger():
  def __init__(self, config):
    self.config = config
    self.log_data = config.log_data
    if self.log_data:
      try:
        self.vis = visdom.Visdom()
        clear_envs(self.vis)
      except Exception as e:
        err_msg = "Couldn't connect to Visdom. Make sure to have a Visdom server running or turn of logging in the config"
        raise ConnectionError(err_msg) from e

  @log_if_active
  def log_image(self, image):
    self.vis.image(image)