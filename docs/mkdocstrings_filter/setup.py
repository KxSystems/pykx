#!/usr/bin/env python

from setuptools import find_packages, setup


setup(
    name='mkdocstrings_filter',
    version='0.0.1',
    packages=find_packages(),
    entry_points={
        'mkdocs.plugins': [
            'mkdocstrings_filter = mkdocstrings_filter:MkdocstringsFilter',
        ]
    }
)
