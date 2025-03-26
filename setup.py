from setuptools import setup, find_packages

# Method 1: Read requirements from file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="Research",
    version="0.1",
    packages=find_packages(),  # Automatically finds `Algorithm/`
    install_requires=requirements,  # Now properly referenced
    python_requires=">=3.10",  # Ensures compatibility with Python 3.10+
)