from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='tcpyVPI',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'xarray',
        'netCDF4',
        'tcpyPI',
        'pandas',
    ],
    extras_require={
        'plot': ['matplotlib', 'cartopy'],
        'dev': ['pytest'],
    },
    author='Dan Chavas, Jose Ocegueda Sanchez',
    author_email='drchavas@gmail.com',
    description='Calculate the tropical cyclone ventilated Potential Intensity (vPI) and the Genesis Potential Index using vPI (GPIv) from gridded datafiles. Supports both monthly mean and hourly ERA5 data. See Chavas Camargo Tippett (2025, J. Clim.) for details.',
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/drchavas/tcpyVPI',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    license='MIT',
    python_requires='>=3.8',
)
