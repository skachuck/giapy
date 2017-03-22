from setuptools import setup, find_packages

setup(name='giapy',
      version='0.1',
      description='Compute glacial isostacy in python',
      url='http://github.com/skachuck/giapy/',
      author='Samuel B Kachuck',
      author_email='sbk83@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'basemap',
            'pyspharm'
      ],
      zip_safe=False)
