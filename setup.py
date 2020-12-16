from setuptools import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
# Newer pip versions use the ParsedRequirement class which has the `requirement` attribute instead of `req`
try:
    reqs = [str(ir.req) for ir in install_reqs]
except AttributeError:
    reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name='atmos',
    packages=['atmos'],
    version='0.2.5-develop',
    description='Atmospheric sciences utility library',
    author='Jeremy McGibbon',
    author_email='mcgibbon@uw.edu',
    install_requires=reqs,
    url='https://github.com/mcgibbon/atmos',
    keywords=['atmos', 'atmospheric', 'equations', 'geoscience', 'science'],
    classifiers=[],
    license='MIT',
)
