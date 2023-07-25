# ATS modeling workflow

This is a Jupyter Book about modeling workflows using the Advanced Terrestrial Simulator (ATS) {cite:p}`Coon2020`. Thoughout this book, you will learn the basics of integrated watershed modeling ranging from model setup, executation, and post-processes. This book is intended to introduce beginers to process-based hydrologic modeling and does not cover the basics in programming using Python.

```{note}
To start, you will either need to install ATS locally or use pre-installed ATS on HPC clusters. The installation instruction can be found under [ATS Github repo](https://github.com/amanzi/amanzi/blob/master/INSTALL_ATS.md)
```
## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
This workflow uses Docker to run all notebooks. You will need to install Docker on your machine. Please follow the [Docker installation guide](https://docs.docker.com/get-docker/).

## Instructions

1. Open a terminal and clone this repository
```bash
# specify a tag (e.g., v1.0) to clone a specific version of the repo
git clone -b v1.0 https://github.com/pinshuai/ats-workflow 

cd ats-workflow
```

1. Pull the Docker image
```bash
docker pull pshuai/ats_workflow:v1.0
```

```{admonition} Important
Make sure the tag name is the same as the docker image tag name. For example, if you are using `v1.0` tag, you will need to use `pshuai/ats_workflow:v1.0` as the docker image name.
```

1. Run the Docker image

```bash
docker run -it --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v $(pwd):/home/jovyan/workdir:delegated -v $(pwd)/data:/home/jovyan/data:delegated pshuai/ats_workflow:v1.0
```

```{note}
- Make sure the local port number (8888) is not occupied by other processes. Alternatively, you can change the local port number to other numbers (e.g., `-p 8890:8888`).

- On Windows, you may need to replace `$(pwd)` with the absolute path of the current directory (e.g., `C:\Home\Documents\ats-workflow`).
```

4. Follow the prompt on screen to open the Jupyter Lab in your browser. For example, copy the following URL to your browser.

```bash
http://127.0.0.1:8888/lab?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

```{admonition} Important
- If you are using a local port number different than `8888`, you will need to manually replace `8888` with the new port number (e.g., `8890`) specified in the last step in the url. 
```
Now you should be able to run all notebooks within the Jupyter Lab.

You can find more about [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) and [ATS](https://amanzi.github.io/).

## Changelog

### v1.0
- 2023-07-21: Initial release. The meshing, downloading forcing, and generating input files workflow has been updated to work with ATS v1.5.x using Coal Creek Watershed as an example. 

## Bibliography
```{bibliography} references.bib
:style: plain
```
