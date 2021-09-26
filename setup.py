from hideandseek import __version__
print(f'Installing hideandseek@{__version__}')

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hideandseek", # Replace with your own username
    version=__version__,
    author="JaeSung Yoo",
    author_email="jsyoo61@korea.ac.kr",
    description="library for deep learning and privacy preserving deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jsyoo61/hideandseek",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','pandas','matplotlib','hydra-core','tools-jsyoo61']
)
