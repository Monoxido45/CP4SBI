from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="CP4SBI",
    version="1.0.0",
    description="Conformal Prediction for Simulation Based Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=".",
    author="Luben M. C. Cabezas, Vagner S. Santos",
    author_email=".",
    packages=["CP4SBI"],
    license="MIT",
    keywords=[
        "credible regions",
        "calibration",
        "conformal prediction",
        "local coverage",
        "simulation-based inference",
    ],
    install_requires=[
        "numpy>=2.2.4",
        "scikit-learn==1.5.1",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "tqdm==4.66.5",
        "torch>=2.5.1",
        "sbi>=0.24.0",
    ],
    python_requires=">=3.10",
    zip_safe=False,
)
