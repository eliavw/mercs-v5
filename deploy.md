# Deployment Information

We use conda for dependency management.

## Create

Environment made with conda. To make an environment;

```bash
conda create --name <whatever> python=3.7
```

## Export
This environment can be exported to a `.yml` file through the following command:

```bash
conda env export > environment.yml
```

Which creates the `.yml` file present in the root dir.

This really dumps the entire environment into the `.yml` file. This can be overkill. What we currently do _(until an easier workflow arrives)_ is that afterwards, we edit this dump to include only those packages which were manually installed and thus really necessary. Conda will take care of secondary dependencies automatically.


## Load
To recreate this environment, it suffices to run;

```bash
conda env create -f environment.yml -n <whatever>
```

Which presupposes that you have an anaconda/miniconda install running on your own machine.

## Add kernel to Jupyter

To add this python environment to the list of Jupyter environments, do the following. 
```bash
source activate <whatever>
python -m ipykernel install --user --name <whatever> --display-name "Py-<whatever>"
```

_N.b.: This requires ipykernel to be installed in the environment._

