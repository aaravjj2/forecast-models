"""Setup script for ML research pipeline."""

from setuptools import setup, find_packages

setup(
    name="ml-research-pipeline",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "yfinance>=0.2.28",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
    ],
    python_requires=">=3.8",
)


