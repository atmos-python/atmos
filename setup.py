from setuptools import setup

with open('requirements.txt') as f:
    install_reqs = f.read().strip().split('\n')

setup(
    name='atmos',
    packages=['atmos'],
    version='0.2.5-develop',
    description='Atmospheric sciences utility library',
    author='Jeremy McGibbon',
    author_email='mcgibbon@uw.edu',
    install_requires=install_reqs,
    url='https://github.com/mcgibbon/atmos',
    keywords=['atmos', 'atmospheric', 'equations', 'geoscience', 'science'],
    classifiers=[],
    license='MIT',
)
