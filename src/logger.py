import functools

import visdom
from anyfig import get_config

from .utils import plotly_plots as plts


def clear_old_data(vis):
    [vis.close(env=env) for env in vis.get_env_list()]  # Kills wind
    # [vis.delete_env(env) for env in vis.get_env_list()] # Kills envs


def log_if_active(func):
    """Decorator which only calls logging function if logger is active"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if get_config().misc.log_data:
            return func(self, *args, **kwargs)

    return wrapper


class Logger:
    def __init__(self):
        if get_config().misc.log_data:
            try:
                self.vis = visdom.Visdom()
                clear_old_data(self.vis)
            except Exception as e:
                err_msg = (
                    "Couldn't connect to Visdom. Make sure to have a Visdom server running or turn of "
                    "logging in the config"
                )
                raise ConnectionError(err_msg) from e

    @log_if_active
    def log_image(self, image):
        self.vis.image(image)

    @log_if_active
    def log_accuracy(self, accuracy, step, name="train"):
        title = f"{name} Accuracy".title()
        plot = plts.accuracy_plot(self.vis.line, title)
        plot(X=[step], Y=[accuracy])
