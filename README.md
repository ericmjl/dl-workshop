# deep-learning-workshop

In this workshop, I will build your intuition in deep learning, without using a framework.

## Getting Started

You can get started using one of the following methods.

### 1. Setup using `conda` environments

```bash
$ conda env create -f environment.yml
$ source activate dl-workshop
$ python -m ipykernel install --user --name dl-workshop
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

If you want `jax` with GPU, you will need to build from source, or follow the [installation instructions](https://github.com/google/jax#installation)

### 2. "just click Binder"

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ericmjl/dl-workshop/master)