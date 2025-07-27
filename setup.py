# Copyright 2023 Joe Worsham

from setuptools import find_packages, setup

setup(
    name='tensorflow_dynamics',
    version='0.0.1',
    python_requires='>=3.7.0',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='',
    author='Joe Worsham',
    author_email='jworsha2@uccs.edu',
    description='Core components of python differentiable systems framework',
    install_requires=['jupyter',
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'scipy',
                      'tensorflow>=2.11.0',
                      'tensorflow_probability[tf]>=0.19.0',
                      'tqdm'],
)
