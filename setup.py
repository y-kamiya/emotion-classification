import setuptools

# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    # Make sure we strip all comments and options (e.g "--extra-index-url")
    # that arise from a modified pip.conf file that configure global options
    # when running kedro build-reqs
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

    # nvidia apex
    requires.append('pytorch-extension')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emotion-classification",
    version="0.0.1",
    author="Yuji Kamiya",
    author_email="y.kamiya0@gmail.com",
    description="classify text with transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com:y-kamiya/emotion-classification",
    project_urls={
        "Bug Tracker": "https://github.com:y-kamiya/emotion-classification/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=requires,
)
