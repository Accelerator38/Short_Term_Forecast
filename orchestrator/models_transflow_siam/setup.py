import setuptools

from transflow.analytical_model import cc as cc_model
from transflow.analytical_optimize import cc as cc_optimize


setuptools.setup(
    name="transflow",
    version="1.0.0",
    description="Analytical solutions for transient flow modeling in hydrocarbon reservoir",
    url="https://github.com/wracpe/models_transflow_siam",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "llvmlite>=0.36.0",
        "numba>=0.53.1",
        "numpy>=1.21.0",
    ],
    ext_modules=[
        cc_model.distutils_extension(),
        cc_optimize.distutils_extension(),
    ],
)
