from distutils.core import setup

setup(
    name='pyhacrf',
    version='0.0.3',
    packages=['pyhacrf'],
    install_requires=['numpy>=1.9', 'PyLBFGS>=0.1.3'],
    url='https://github.com/dirko/pyhacrf',
    download_url='https://github.com/dirko/pyhacrf/tarball/0.0.3',
    license='BSD',
    author='Dirko Coetsee',
    author_email='dpcoetsee@gmail.com',
    description='Hidden alignment conditional random field, a discriminative string edit distance',
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
