#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'six',
    'pydicom',
    'numpy',
    'scipy',
    'matplotlib',
    'scikit-learn>=0.18'
    'joblib',
    'h5py',
    'mahotas',
]

setup(
    name='geodosic',
    version='0.1.0',
    description='Finding geometric patterns in dose distributions',
    author='David Hall',
    author_email='dhcrawley@gmail.com',
    url='https://github.com/davidchall/geodosic',
    packages=[
        'geodosic',
    ],
    package_dir={'geodosic':
                 'geodosic'},
    install_requires=requirements,
    test_suite='tests'
)
