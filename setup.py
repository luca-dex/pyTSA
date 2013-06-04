from distutils.core import setup

setup(
    name='RedPanda',
    version='0.1.1',
    author='L. De Sano',
    author_email='l.desano@campus.unimib.it',
    packages=['redpanda',],
    license='LICENSE.txt',
    description='Time Series analysis with pandas',
    url='https://github.com/luca-dex/RedPanda',
    long_description=open('README.txt').read(),
    install_requires=[
        "Numpy >= 1.6.1",
        "Scipy >= 0.10.1",
        "Pandas >= 0.11.0",
        "python-dateutil >= 1.5",
        "matplotlib",
        "statsmodels",
    ],
)