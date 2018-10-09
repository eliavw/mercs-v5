# MERCS

MERCS stands for **m**ulti-directional **e**nsembles of **c**lassification and **r**egression tree**s**. It is a novel ML-paradigm under active development at the [DTAI-lab at KU Leuven](https://dtai.cs.kuleuven.be/).

## News


## Resources

### Tutorials

We offer a small collection of tutorials in the form of Jupyter Notebooks (cf. [github-repo](https://github.com/eliavw/mercs-v5/tree/master/note/tutorials) for the actual `.ipynb` files) of quick walkthroughs MERCS' most common functionalities. These are intended as the most user-friendly entry point to our system. 

**MERCS 101**
1. [Classification](tutorials/01_mercs_basics_classification.html)
2. [Classification](tutorials/02_mercs_basics_regression.html)
3. [Mixed](tutorials/03_mercs_basics_mixed.html)

### Documentation

Our documentation can be found at [read the docs](https://mercs.readthedocs.io/en/latest/#)

### Code

MERCS is fully open-source cf. our [github-repository](https://github.com/eliavw/mercs-v5/)

## Publications

MERCS is an active research project, hence we periodically publish our findings;

### MERCS: Multi-Directional Ensembles of Regression and Classification Trees

**Abstract**
*Learning a function f(X) that predicts Y from X is the archetypal Machine Learning (ML) problem. Typically, both sets of attributes (i.e., X,Y) have to be known before a model can be trained. When this is not the case, or when functions f(X) that predict Y from X are needed for varying X and Y, this may introduce significant overhead (separate learning runs for each function). In this paper, we explore the possibility of omitting the specification of X and Y at training time altogether, by learning a multi-directional, or versatile model, which will allow prediction of any Y from any X. Specifically, we introduce a decision tree-based paradigm that generalizes the well-known Random Forests approach to allow for multi-directionality. The result of these efforts is a novel method called MERCS: Multi-directional Ensembles of Regression and Classification treeS. Experiments show the viability of the approach.*

**Authors**
Elia Van Wolputte, Evgeniya Korneva, Hendrik Blockeel

**Open Access**
A pdf version can be found at [AAAI-publications](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16875/16735)
