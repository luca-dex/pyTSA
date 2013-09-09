from setuptools import setup
setup(
    name = "RedPanda",
    version = "0.1.5",
    packages=['redpanda'],

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
    author = "L. De Sano",
    author_email = "l.desano@campus.unimib.it",
    description = "Time Series analysis with pandas",
    license = "LICENSE.txt",
    keywords = "timeseries pandas bio",
    url = "https://github.com/luca-dex/RedPanda"   
    
)