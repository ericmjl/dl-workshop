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

## Key Ideas

The key idea for this tutorial is that if we really study deep learning's fundamental model, linear regression, then we can get a better understanding of the components - a model with parameters, a loss function, and an optimizer to change the parameters to minimize the loss. Most of us who become practitioners (rather than researchers) can then take for granted that the same ideas apply to any more complex/deeper model.

## Feedback

I'd love to hear how well this workshop went for you. Please consider [leaving feedback so I can improve the workshop](https://ericma1.typeform.com/to/Tv185B).
