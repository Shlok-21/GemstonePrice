from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        requirement = [req.replace("\n","") for req in requirement]

        if HYPHEN_E_DOT in requirement:
            requirement.remove(HYPHEN_E_DOT)



setup(
    name='GemstonePrice',
    version='0.1',
    packages=find_packages(),
    author = 'Shlok',
    author_email='shlokshivkar21@gmail.com',
    install_requires = get_requirements("requirements.txt") 
)

