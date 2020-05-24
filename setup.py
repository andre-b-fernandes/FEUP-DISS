import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="increc",  # Replace with your own username
    version="0.0.6",
    author="Fernando André Fernandes",
    author_email="fernandoandre49@gmail.com",
    description="Software library on stream based recommender systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Marko50/FEUP-DISS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
