import setuptools


setuptools.setup(
    name="wolfram",
    version="1.0.0",
    description="Well OiL Forecast RAte by Ml",
    url="https://github.com/wracpe/frameworks_ftor",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "xgboost>=1.4.2",
    ],
)
