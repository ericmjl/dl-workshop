# Differential Programming with JAX

Thanks for stopping by to read this online book on differential programming!

## What you will learn

From a practical standpoint, this book will teach you the basics of how to use JAX,
in particular the idioms and how they map onto what we might alrady know in Python.

From a more abstract standpoint, this book will give you practice with a more "functional" style of programming
(in contrast to an object-oriented style or an imperative style).

My goal for you is to finish reading the book
having the confidence to write differentiable numeric models of the world.
The key operative word here being "differentiable" - you can calculate and evaluate
the gradient of a model (written as a function) w.r.t. its parameters (which are passed in as inputs).

Along the way, you might see the connections between
topics that you might be familiar with (Bayesian statistics, deep learning, and more)
and differntial computing.
If they pop out to you through this book and the examples in there,
then I know you'll likely enjoy the thrill of seeing
a new connection in your personal knowledge graph.

## How to use this book

### For online readers

This website, which is freely available to all, can be read in order from start to end.
If you're already familiar with differential computing and are curious about how to write JAX programs,
head over to the section on JAX programming.

If you're curious about how to write neural network models, head over to the `stax` section.

There's also a collection of "case study"/"recipe"-like chapters,
in which we set up a computing problem of relevance and walk through how to write a JAX program there,
leveraging what we have learned in the rest of the book.

### For interactive coding learners

If you're the type who likes to execute code and break it in order to learn about what's going on,
or if you're in an online interactive learning session with me,
then you're in luck!
The entire book has been written using Jupyter notebooks and Markdown files,
and any section written as a Jupyter notebook has an "open in Binder" badge available at the top.
Look for the button that looks like the one below:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ericmjl/dl-workshop/master)

This will bring you to a pre-built Binder sesion that you can use to execute the code,
break it, and play around with the ideas in the book.
There are exercises interspersed throughout the book that you can stop to read through as well.

If you prefer to set up an environment locally, here are instructions for you:

```bash
conda env create -f environment.yml
conda activate dl-workshop  # older versions of conda use `source activate` rather than `conda activate`
python -m ipykernel install --user --name dl-workshop
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

If you want `jax` with GPU, you will need to build from source, or follow the [installation instructions](https://github.com/google/jax#installation)

If you are using Jupyter Lab, you will want to also ensure that `ipywidgets` is installed:

```bash
# only if you don't have ipywidgets installed.
conda install -c conda-forge ipywidgets
# the next line is necessary.
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Further Reading

- [Demystifying Different Variants of Gradient Descent Optimization Algorithm](https://hackernoon.com/demystifying-different-variants-of-gradient-descent-optimization-algorithm-19ae9ba2e9bc)
