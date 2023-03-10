from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements.
    """
    requirements = []
    with open(file_path) as file_objec:
        requirements = file_objec.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='ml project',
    version='0.0.1',
    author='girish',
    author_email='girishkrish1998yuv@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

