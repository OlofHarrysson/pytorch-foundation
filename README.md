# pytorch-foundation
A barebone project structure for pytorch projects that can be built upon. To avoid having to rewrite all the boilerplate code for every new project

TODO
environment file

modular config sub-modules that can be used in config?
transforms = config.small_transforms()
for k,v in vars(transforms):
  setattr(k,v) ish


Print all available config classes if wrong was chosen.

Use pathlib to get git_dir. Create a saved/output dir in git dir.