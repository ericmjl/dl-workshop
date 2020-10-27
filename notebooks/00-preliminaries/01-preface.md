# Preface

The years between 2010 to 2020 was a breakout decade for deep learning.
There, we saw an explosion of tooling, model building, flashy demos of performance gains, and more.
But beneath the surface hype of deep learning models and methods,
I witnessed the maturity of tooling surrounding this idea of "differential programming"
and composable program transforms.
That tooling is the focus of this material you're reading here.

## On differential programming

What do we mean by _differential programming_?
From one perspective, it is the core of modern-day learning systems,
where _mathematical derivatives_, also known as _gradients_,
are used in optimization and learning tasks.
For example, we might use gradient descent to optimize
a linear model that maps predictor variables to and output variable of interest.
As another example, we might use gradient descent to optimize
the parameters of a neural network model
that classifies molecules as being biodegradable or not
based on their descriptors alone.
(As you will see in this book, the "learning" in deep learning
is nothing more than a optimization of parameters by gradient descent.)
With differential computing, the key activity that we engage in
is the calculation of derivatives,
and tooling that helps us compute derivatives _automatically_,
such that we do not have to calculate them by hand,
is central to differential computing.
If you took a calculus class, the chain rule will feature prominently here,
and JAX provides the tooling that gives us _automatic differentiation_,
i.e. a "program transformation" that automatically calculates a derivative function
for any other function that calculates a scalar-valued output.

## On program transformations

As mentioned in the last paragraph, when structured in the way that JAX does it,
derivative computation falls under this umbrella of "program transformations".
That is to say, JAX provides functions
that _transforms_ a program from one form into another.
(We will see by exactly what syntax we'll need to automatically create gradient functions
later in the book.)
Now, gradients aren't the only program transformation that exist.
A function that maps a scalar function over a vector of inputs,
thereby producing a vector of outputs instead, is another example of a program transformation.
In this book, we will explore through
some of the program transformations that are available in JAX,
and see how they can be used to write beautifully structured array programs
that are more flat than nested,
more explicit than implicit,
and are tens to hundreds of times more performant than vanilla Python/NumPy programs.

## The choice of JAX vs. other array frameworks

As of the time of writing, there are two dominant deep learning frameworks
that also provide automatic differentiation.
They are PyTorch and TensorFlow, which have each enjoyed their zenith of fame.
JAX generally distinguishes itself from PyTorch and TensorFlow in two ways by

1. being developed against the idiomatic NumPy and SciPy APIs, thereby being extremely compatible with the rest of the PyData ecosystem;
2. extending the Python language with more program transforms than just differential computing transformations,
3. properly documenting the reasons why they depart from the very small subset of idioms that they don't follow.

In particular, I would like to highlight the first point.
API interoperability between computing packages is crucial for a thriving data science ecosystem.
JAX's NumPy and SciPy wrappers ensure that all computations done using existing NumPy and SciPy code
can very easily be transformed into differential-compatible computations
for which program transforms provided by JAX can be easily applied.
As you'll see in the book, you can even plot JAX arrays in `matplotlib`,
the venerable Python plotting library,
because of its developer's compatibility efforts.
