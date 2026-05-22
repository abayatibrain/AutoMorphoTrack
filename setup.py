from setuptools import setup, find_packages
import os

# Read README file (handle missing file gracefully for pip install from sdist)
long_description = ""
readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="automorphotrack",
    version="2.3.0",
    author="Armin Bayati, Ph.D.",
    author_email="a.bayati.brain@gmail.com",
    description="Automated pipeline for mitochondrial and lysosomal detection, tracking, morphology, and colocalization analysis in microscopy images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abayatibrain/AutoMorphoTrack",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.6.0",
        "scikit-image>=0.19.0",
        "scipy>=1.9.0",
        "tifffile>=2022.8.12",
    ],
    extras_require={
        "mcp": ["mcp>=1.0.0"],  # Optional MCP server for Claude Code integration
        "napari": [
            "napari>=0.4.18",
            "magicgui>=0.7.0",
            "qtpy>=2.3.0",
            "PyQt5>=5.15;platform_system!='Darwin' or platform_machine!='arm64'",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "build>=1.0",
            "twine>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "automorphotrack=automorphotrack.cli:main",
            "amt=automorphotrack.cli:main",
        ],
        "napari.manifest": [
            "automorphotrack = automorphotrack:napari.yaml",
        ],
    },
    package_data={
        "automorphotrack": ["napari.yaml"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    include_package_data=True,
    license="MIT",
    project_urls={
        "Documentation": "https://github.com/abayatibrain/AutoMorphoTrack",
        "Source": "https://github.com/abayatibrain/AutoMorphoTrack",
        "Tracker": "https://github.com/abayatibrain/AutoMorphoTrack/issues",
    },
)
