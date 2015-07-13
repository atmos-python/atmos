from setuptools import setup
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='atmos',
    packages=['atmos'],
    version='0.2.3',
    description='Atmospheric sciences utility library',
    author='Jeremy McGibbon',
    author_email='mcgibbon@uw.edu',
    install_requires=reqs,
    url='https://github.com/mcgibbon/atmos',
    keywords=['atmos', 'atmospheric', 'equations', 'geoscience', 'science'],
    classifiers=[],
)
