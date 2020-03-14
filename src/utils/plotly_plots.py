import functools


def accuracy_plot(func, title):
  opts = dict(xlabel='Steps', ylabel='Accuracy', title=title)
  return functools.partial(func, update='append', win=title, opts=opts)
