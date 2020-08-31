set -e
export PIP_DISABLE_PIP_VERSION_CHECK=1
# install amazon-ecr-credential-helper
if [ ! -f /usr/bin/docker-credential-ecr-login ]; then
    sudo wget -P /usr/bin https://amazon-ecr-credential-helper-releases.s3.us-east-2.amazonaws.com/0.4.0/linux-amd64/docker-credential-ecr-login
    sudo chmod +x /usr/bin/docker-credential-ecr-login
fi
# install Python dependencies in the python3 conda environment
source /home/ec2-user/anaconda3/bin/activate python3
if ! pip freeze | grep 'pip==20.0.2'; then
    pip install --upgrade pip==20.0.2
fi
# fix to upgrade `docutils` that was installed with `distutils` (hence pip can't uninstall)
rm -rf /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/docutils
rm -rf /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/docutils-*
pip install --upgrade entrypoints --ignore-installed
pip install --upgrade ptyprocess --ignore-installed
pip install --upgrade terminado --ignore-installed
cd /home/ec2-user/SageMaker
pip install -r ./containers/dashboard/requirements.txt
pip install -r ./containers/model/requirements.txt
pip install -r ./package/requirements.txt
pip install -e ./package/
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
if ! pip freeze | grep 'pip==20.0.2'; then
    pip install --upgrade pip==20.0.2
fi
# update jupyter-server-proxy
if ! pip freeze | grep -q 'jupyter-server-proxy==1.3.2'; then
    pip uninstall -y nbserverproxy || true
    pip install --upgrade jupyter-server-proxy==1.3.2
    echo 'Restarting jupyter-server. Jupyter notebook/terminals will freeze, so please refresh page to resume interaction.'
    sudo initctl restart jupyter-server --no-wait
fi