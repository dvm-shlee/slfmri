#!/usr/scripts/env python
"""
Personal module collection for fMRI data analysis.
"""
from distutils.core import setup
from setuptools import find_packages
import re, io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('slfmri/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

__author__ = 'SungHo Lee'
__email__ = 'shlee@unc.edu'
__url__ = 'https://github.com/dvm-shlee/slfmri'

setup(name='slfmri',
      version=__version__,
      description='Helper for fmri analysis for dvm-shlee',
      python_requires='>3.5',
      author=__author__,
      author_email=__email__,
      url=__url__,
      license='GNLv3',
      packages=find_packages(),
      install_requires=['shleeh>=0.0.4',
                        'nibabel>=3.0.2',
                        'SimpleITK>=1.2.4',
                        'numpy>=1.18.0',
                        'pandas>=1.0.0',
                        'scipy>=1.4.0',
                        'scikit-learn>=0.22.0'],
      classifiers=[
          'Development Status :: 1 - Planning',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows :: Windows 10',
          'Programming Language :: Python :: 3.7',
          'Topic :: Software Development',
      ],
      keywords='personal helper'
     )
