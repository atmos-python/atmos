from setuptools import setup

def parse_requirements(filename):
    requirements = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                requirements.append(line)
    return requirements

install_reqs = parse_requirements('requirements.txt')

setup(
    name='atmos',
    packages=['atmos'],
    version='0.2.5',
    description='Atmospheric sciences utility library',
    author='Jeremy McGibbon',
    author_email='mcgibbon@uw.edu',
    install_requires=install_reqs,
    url='https://github.com/mcgibbon/atmos',
    keywords=['atmos', 'atmospheric', 'equations', 'geoscience', 'science'],
    classifiers=[],
    license='MIT',
)
