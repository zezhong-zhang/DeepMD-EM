# -*- coding: utf-8 -*-

from os import path
from pathlib import Path
import setuptools

# define constants
INSTALL_REQUIRES = (Path(__file__).parent / "requirements.txt").read_text().splitlines()
setup_requires = ["setuptools_scm"]

readme_file = Path(__file__).parent / "README.md"
readme = readme_file.read_text(encoding="utf-8")

setuptools.setup(
    name="deepmdem",
    use_scm_version={'write_to': 'deepmdem/_version.py'},
    setup_requires=['setuptools_scm'],
    author="Zezhong Zhang",
    author_email="zezhong.zhang@uantwerpen.be",
    description="deepmdem: electron scattering simulation with realistic phonons.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/zezhong-zhang/deepmdem",
    packages=[],
    package_data={'deepmdem':['*.json']},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    keywords='deep potential concurrent learning',
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "deepmdem = deepmdem.entrypoint.main:main"
        ]
    },
    extras_require={
        'docs': [
            'sphinx',
            'recommonmark',
            'sphinx_rtd_theme>=1.0.0rc1',
            'numpydoc',
            'myst_parser',
            'deepmodeling_sphinx',
            'sphinx-argparse',
            "dargs>=0.3.1",
            'ase'
        ],
    }
)

