from setuptools import find_packages, setup

setup(
    name="ml_frameworks_bot",
    version="0.1.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/soumik12345/ml-framework-bot",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").read().splitlines(),
)
