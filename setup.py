from setuptools import setup, find_packages

setup(
    name='fidA',
    version='0.1.0',
    author='Colleen Bailey',
    author_email='colleen.em.bailey@gmail.com',
    description='A Python package for FID-A processing based on the MATLAB code by Jamie Near',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/stonesunturned/fidA',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'spec2nii'
        # Add other dependencies if any
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
