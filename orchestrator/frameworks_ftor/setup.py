import setuptools


setuptools.setup(
    name="ftor",
    version="2.0.0",
    description="Forecast of Transient Oil Rate",
    url="https://github.com/wracpe/frameworks_ftor",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numba>=0.53.1",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
)
