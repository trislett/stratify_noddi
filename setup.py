import os
import sys

from distutils.command.sdist import sdist
from setuptools import setup, find_packages

PACKAGE_NAME = "stratify_noddi"
BUILD_REQUIRES = ["numpy", "nibabel", "argparse", "matplotlib", "scipy", "scikit-image", "dmri-amico"]

CLASSIFIERS = ["Development Status :: 3 - Alpha",
  "Environment :: Console",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  "Operating System :: OS Independent",
  "Programming Language :: Python",
  "Topic :: Scientific/Engineering :: Medical Science Apps."]

def parse_setuppy_commands():
  info_commands = ['--help-commands', '--name', '--version', '-V',
    '--fullname', '--author', '--author-email',
    '--maintainer', '--maintainer-email', '--contact',
    '--contact-email', '--url', '--license', '--description',
    '--long-description', '--platforms', '--classifiers',
    '--keywords', '--provides', '--requires', '--obsoletes']
  info_commands.extend(['egg_info', 'install_egg_info', 'rotate'])
  for command in info_commands:
    if command in sys.argv[1:]:
      return False
  return True

def configuration(parent_package = "", top_path = None):
  from numpy.distutils.misc_util import Configuration
  CONFIG = Configuration(None)
  CONFIG.set_options(ignore_setup_xxx_py = True,
    assume_default_configuration = True,
    delegate_options_to_subpackages = True,
    quiet = True)

  CONFIG.add_scripts(os.path.join("pipeline", "run_full_noddi_preprocessing"))
  CONFIG.add_scripts(os.path.join("pipeline", "01_stratify_unwarp"))
  CONFIG.add_subpackage(PACKAGE_NAME)

  return CONFIG

cmdclass = {"sdist": sdist}

if os.path.exists('MANIFEST'):
  os.remove('MANIFEST')

if parse_setuppy_commands():
  from numpy.distutils.core import setup

exec(open('stratify_noddi/version.py').read())
setup(name = PACKAGE_NAME, version = __version__, include_package_data=True,
  maintainer="Tristram Lett",
  maintainer_email="tristram.lett@charite.de",
  description="stratify_noddi",
  long_description="NODDI processing for STRATIFY sample",
  url="https://github.com/trislett/stratify_noddi",
  download_url="",
  platforms=["Linux", "Solaris", "Mac OS-X", "Unix"],
  license="GNU General Public License v3 or later (GPLv3+)",
  classifiers=CLASSIFIERS,
  zip_safe=False,
  cmdclass=cmdclass,
  install_requires=BUILD_REQUIRES,
  packages=['stratify_noddi'],
  package_dir={'stratify_noddi': ''},
  package_data={'stratify_noddi': ['ants_oasis_template_ras/*.nii.gz']},
  configuration=configuration
)
