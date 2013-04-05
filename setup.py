from distutils.core import setup

setup(
    name='bio-d3',
    version='0.1.0',
    author='L. De Sano',
    author_email='l.desano@campus.unimib.it',
    packages=['biodf',],
 #   scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
 #   url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Bio data analysis with pandas and d3',
    long_description=open('README.txt').read(),
    install_requires=[
        "Numpy >= ",
        "Scipy == ",
        "Pandas >= ",
    ],
)