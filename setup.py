from setuptools import setup, find_packages

REQUIRED = ['numpy']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='linear-rl',
    version='0.1',
    packages=['linear_rl',],
    install_requires=REQUIRED,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    long_description=long_description,
    url='https://github.com/LucasAlegre/linear-rl',
    license="MIT",
    description='Reinforcement Learning (RL) with linear function approximation.'
)
