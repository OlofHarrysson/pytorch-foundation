import visdom
import pandas as pd

def clear_envs(viz):
  [viz.close(env=env) for env in viz.get_env_list()] # Kills wind
  # [viz.delete_env(env) for env in viz.get_env_list()] # Kills envs

class Logger():
  def __init__(self, config):
    self.config = config
    self.viz = visdom.Visdom(port='6006')
    clear_envs(self.viz)


class EMAverage(object):
  ''' Smooths the curve with Exponential Moving Average '''
  def __init__(self, time_steps):
    self.vals = deque([], time_steps)

  def update(self, val):
    self.vals.append(val)
    df = pd.Series(self.vals)
    return df.ewm(com=0.5).mean().mean()