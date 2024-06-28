from setuptools import setup, find_packages

setup(
    name='radgraph',
    version='0.1.2',
    author='Jean-Benoit Delbrouck',
    license='MIT',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    install_requires=['torch>=1.8.1',
                      'transformers>=4.23.1',
                      "appdirs",
                      'jsonpickle',
                      'filelock',
                      'h5py',
                      'spacy',
                      'nltk',
                      'dotmap',
                      ],
    packages=find_packages(),
    zip_safe=False)
