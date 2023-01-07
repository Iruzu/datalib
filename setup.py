from setuptools import setup

setup(
   name='datalib',
   version='0.1.0',
   author='Irune Zubiaga',
   author_email='irune.zubiaga@ehu.eus',
   packages=['datalib', 'datalib.test'],
   url='https://github.com/Iruzu/datalib.git',
   license='LICENSE.txt',
   description='This package includes some basic functions to analyze datasets.',
   long_description=open('README.md').read(),
   tests_require=['pytest'],
   install_requires=[
      "pandas >= 0.25.1",
      "matplotlib >= 3.1.1",
      "numpy >=1.17.2"
   ],
)