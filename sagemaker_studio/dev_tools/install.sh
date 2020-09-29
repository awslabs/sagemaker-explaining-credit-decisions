#!/bin/bash
set -e
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
bash ./jupytext/install.sh
bash ./theia/install.sh
echo "Success: restarting jupyter-server."
sudo initctl restart jupyter-server --no-wait