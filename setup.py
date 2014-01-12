from setuptools import setup
setup(
    name = "pyTSA",
    version = "0.2.0",
    packages=['pytsa'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "Numpy >= 1.6.1",
        "Scipy >= 0.10.1",
        "Pandas >= 0.12.0",
        "patsy",
        "statsmodels",
        "python-dateutil >= 1.5",
        "matplotlib >= 1.3.0",
        "numexpr",
        "bottleneck"
    ],

    # metadata for upload to PyPI
    author = "L. De Sano, G. Caravagna",
    author_email = "l.desano@campus.unimib.it",
    description = "Time Series analysis with pandas",
    license = "LICENSE.txt",
    keywords = "timeseries pandas bio",
    url = "https://github.com/luca-dex/pyTSA"   
    
)