from distutils.core import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyhacrf',
    version='0.0.5',
    packages=['pyhacrf'],
    install_requires=['numpy>=1.9', 'PyLBFGS>=0.1.3'],
    url='https://github.com/dirko/pyhacrf',
    download_url='https://github.com/dirko/pyhacrf/tarball/0.0.5',
    license='BSD',
    author='Dirko Coetsee',
    author_email='dpcoetsee@gmail.com',
    description='Hidden alignment conditional random field, a discriminative string edit distance',
    long_description=long_description,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        ],
    )
