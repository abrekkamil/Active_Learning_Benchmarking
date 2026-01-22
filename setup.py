from setuptools import setup, find_packages

setup(
    name="active_learning_benchmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "scikit-learn>=0.24.2",
        "wandb>=0.12.0",
        "matplotlib>=3.4.2",
        "pyyaml>=5.4.1",
    ],
    python_requires=">=3.7",
)