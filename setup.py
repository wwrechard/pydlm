from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name = 'pydlm',
    author = 'Xiangyu Wang',
    author_email = 'wwrechard@gmail.com',
    description = ('A python library for the Bayesian dynamic linear ' +
      'model for time series modeling'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    license = 'BSD',
    keywords = 'dlm bayes bayesian kalman filter smoothing dynamic model',
    url = 'https://github.com/wwrechard/pydlm',
    packages = find_packages(),
    zip_safe= False,
    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: BSD License',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    include_package_data = False,
    install_requires = [
      'numpy',
      'matplotlib',
    ],
    tests_require = [
    ],
    extras_require = {
        'docs': [
          'Sphinx',
        ],
    },
)
