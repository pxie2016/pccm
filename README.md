# pfcm: Python-based Feature-specific Changepoint Modeling

*Zimeng (Parker) Xie*

---

## Introduction

This data science/machine learning personal project is based on
my [master's thesis](https://github.com/pxie2016/UWThesis) at
[University of Washington](https://www.washington.edu) 
[Department of Biostatistics](https://www.biostat.washington.edu),
which is in turn derived from work of Dr. Vito M. R. Muggeo 
from [University of Palermo](https://www.unipa.it). Specifically,
changepoint estimation methods from 
[Estimating regression models
with unknown break‐points (2003)](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.1545)
and [Segmented mixed models with random changepoints: A maximum
likelihood approach with application to treatment for depression
study (2014)](https://journals.sagepub.com/doi/abs/10.1177/1471082X13504721)
are synthesized to estimate both fixed and feature-specific changepoints
using a custom simulated annealing algorithm inspired by 
[Wood (2001)](https://www.jstor.org/stable/2676866).

The project was originally written by myself in R, a language commonly used in statistics
and data science communities worldwide. As a personal exercise at the intersection of statistics,
DS/ML, and full-stack web development, the development and deployment of this project is planned
in three large phases (that will be broken down into smaller phases):

1. The porting of the core algorithm from R to Python. 🚧
2. Building RESTful APIs using [Flask](https://flask.palletsprojects.com/), 
and a front end using [Angular](https://angular.io/), with frequent and deliberate coordination.
3. Polishing the end product from 2. with elements in UI/UX design.

For those of you made it to this line, the author heartily appreciates your enthusiasm in this
piece of work in progress. Please let the author know your feedback using the Issues and
Discussion features on GitHub.