# bayes-nets
A playground for testing and comparing different inference methods on Bayes Nets.

## Setup
Create conda environment with all required dependencies:
```
conda env create -f environment.yml
```

## File Structure
### Probability Distributions
The basic probability distribution classes (tabular and Gaussian) are located in `distributions.py`. These allow you
to specify the parameters or tabular values for unconditional distributions.

Conditional distribution classes (tabular and Gaussian) are located in `conditionals.py`. For these, you must specify
a mapping from evidence variables to probability distributions.


### Bayes Nets
`dag.py` contains the implementation for a directed acyclic graph (DAG), along with useful methods like topological sort.

The BayesNet class (located in `bayes_net.py`) is a subclass of DAG that contains extra attributes and methods for sampling.
Setting up a Bayes net involves specifying all connections and a CPT for each node, then calling `build()`. 

Full documentation on what you can do with BayesNets (sampling, fitting, inference) is here:
https://paper.dropbox.com/doc/Bayes-Nets--Ay~lI2da1ow6iSvNuJsbXs3AAg-pQj20OftCqLRPLbGTW9di
