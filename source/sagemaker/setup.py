from distutils.core import setup


with open('requirements.txt') as f:
    notebook_requirements = f.readlines()

with open('./containers/requirements.txt') as f:
    container_requirements = f.readlines()

requirements = notebook_requirements + container_requirements

setup(
    name='explain',
    version='1.0',
    install_requires=requirements,
    description='A package to organize explainability code.',
    author='Thom Lane',
    url='https://github.com/awslabs/sagemaker-explaining-credit-decisions',
    packages=['src.package'],
)
