from setuptools import setup, find_packages


# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line and not line.startswith("#")]


setup(
    name="trbdv0",
    version="0.1.0",
    author="Yewen Zhou",
    author_email="thefirstzyw@hotmail.com",
    description="TRBD V0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BCM-Neurosurgery/TRBD-null-pipeline",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "send-email=trbdv0.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md"],
        "trbdv0": ["data/*.dat"],
    },
)
