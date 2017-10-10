#!/usr/bin/env Python
# -*- coding=utf-8 -*-

from distutils.core import setup
from setuptools import find_packages

setup(
    name = 'tensorflow_nlp',
    version = '0.0.1',
    description = 'Deep Learning NLP Tasks implemented on Tensorflow',
    long_description=open("README.rst").read(),
    author = 'endymecy',
    author_email = 'endymecy@sina.cn',
    url = 'https://github.com/koala-ai/tensorflow_nlp',
    license="Apache License 2.0",
    keywords='DeepLearning Chinese-NLP Tensorflow Koala-ai',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=['numpy', 'tensorflow', 'sklearn', 'jieba', 'scipy', 'tqdm', 'nltk', 'pandas'
    ],
)