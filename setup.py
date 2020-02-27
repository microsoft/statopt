from setuptools import setup, find_packages

setup(
    name='statopt',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Statistical adaptive stochastic optimization methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy', 'torch'],
    url='https://github.com/microsoft/statopt'
)
