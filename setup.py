from setuptools import setup, find_packages

setup(
    name='mixture-of-depth',
    version='1.1.8',
    author='Marco Lironi',
    author_email='marcolironi@astramind.ai',
    description='Unofficial implementation for the paper "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/astramind-ai/Mixture-of-depths',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
