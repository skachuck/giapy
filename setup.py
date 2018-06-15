from setuptools import setup, find_packages

setup(name='giapy',
    version='1.0.0',
    description='Compute glacial isostacy in python',
    long_description=""" """,
    url='http://github.com/skachuck/giapy/',
    author='Samuel B Kachuck',
    author_email='sbk83@cornell.edu',
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python 2',
      'Programming Language :: Python 3',
      'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords='geophysics deformation isostasy',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': ['giapy-ellove=giapy.command_line:ellove',
                            'giapy-velove=giapy.command_line:velove'],
    },
)
