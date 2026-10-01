import setuptools


setuptools.setup(
    name="waterflood",
    version="1.0.0",
    description="Watercut calculation for specific reservoir drainage volume",
    url="https://github.com/wracpe/models_waterflood",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
)
