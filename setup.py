import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "primitives",
    version = "0.0.1",
    author = "Christopher 'ckt' Tomaszewski",
    author_email = "christomaszewski@gmail.com",
    description = ("A library of primitives used throughout multiple research projects"),
    license = "BSD",
    keywords = "measurement track grid primitive",
    url = "https://github.com/christomaszewski/primitives.git",
    packages=['primitives', 'tests', 'examples'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)