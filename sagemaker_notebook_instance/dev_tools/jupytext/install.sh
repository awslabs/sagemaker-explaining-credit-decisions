#!/bin/bash
set -e
source activate JupyterSystemEnv
pip install jupytext --upgrade
jupyter serverextension enable jupytext
source deactivate