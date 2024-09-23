from setuptools import find_packages, setup

def read_requirements(filename: str):
    with open(filename) as requirements_file:
        requirements = []
        for line in requirements_file:
            line = line.strip()
    return requirements


setup(
    name="hw1",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=read_requirements("requirements.txt"),
    python_requires="~=3.10",
)
