#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for data.

    This file was generated with PyScaffold 2.3, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import sys
from setuptools import setup


def setup_package():
    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner'] if needs_pytest else []
    setup(setup_requires=['six', 'pyscaffold>=2.3rc1,<2.4a0'] + pytest_runner,
          tests_require=['pytest_cov', 'pytest', 'markupsafe', 'entrypoints'],
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
