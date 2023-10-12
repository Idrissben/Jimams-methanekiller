from setuptools import setup, find_packages

setup(
    name="methane-killer",
    version="0.1.0",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jimams",
    packages=find_packages(),
    install_requires=[
        "python>=3.9,<3.9.7 || >3.9.7,<3.13",
        "streamlit>=1.27.2,<2.0",
        "scikit-learn>=1.3.1,<2.0",
        "torch>=2.1.0+cpu",
        "torchvision>=0.16.0+cpu",
        "pandas>=2.1.1,<3.0",
        "seaborn>=0.13.0,<1.0",
        "matplotlib>=3.8.0,<4.0",
        "tqdm>=4.66.1,<5.0",
        "ipywidgets>=8.1.1,<9.0",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cpu",
    ],
    extras_require={
        "dev": [
            "ipykernel>=6.25.2,<7.0",
        ],
    },
)
