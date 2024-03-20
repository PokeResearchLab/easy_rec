import os
from setuptools import setup, find_packages

# Read the contents of the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Initialize an empty list for the installation requirements
install_requires = []

# Check if a requirements.txt file exists and if so, read its contents
if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()

# Define the package setup configuration
setup(
    name='Rec Utils',  # Replace with your package name
    packages=find_packages(),  # List of all packages included in your project
    description='Rec Utils: A set of utilities for recommendation systems',
    long_description=long_description,  # Use the contents of README.md as the long description
    long_description_content_type="text/markdown",
    version='1.0.0',  # Specify the version of your package
    install_requires=install_requires,  # List of required dependencies
    url='https://github.com/siciliano-diag/rec_utils.git',  # Replace with the URL of your GitHub repository
    author='Federico Siciliano',
    author_email='siciliano@diag.uniroma1.it',
    keywords=['MachineLearning', 'Recommendation Systems']  # Keywords related to your package
)
