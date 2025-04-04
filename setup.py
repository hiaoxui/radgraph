from setuptools import setup, find_packages

setup(
    name="radgraph",
    version="0.1.17",
    author="Jean-Benoit Delbrouck",
    license="MIT",
    url="https://github.com/Stanford-AIMI/radgraph",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.8,<3.11",
    install_requires=[
        "torch>=2.1.0,<=2.6",
        "transformers>=4.39.0,<4.50.0",
        "appdirs",
        "jsonpickle",
        "filelock",
        "h5py",
        "spacy",
        "nltk",
        "dotmap",
        "pytest",
    ],
    packages=find_packages(),
    zip_safe=False,
)
