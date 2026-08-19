"""
Setup script for the Function Graph Generator package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A tool for analyzing and visualizing function call relationships in source code."

setup(
    name="function-graph-generator",
    version="2.0.0",
    author="Function Graph Generator Team",
    author_email="",
    description="Generate function call graphs from source code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/waifuai/function-graph-generator",
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT No Attribution License (MIT-0)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Software Development :: Documentation",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=[
        "networkx>=3.6",
        "matplotlib>=3.11",
        "pillow>=12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0.0",
            "pytest-cov>=7.0.0",
            "black>=26.0.0",
            "flake8>=7.0.0",
            "mypy>=2.0.0",
        ],
        "yaml": ["pyyaml>=6.0.3"],
        "dot": ["pydot>=4.0.0"],
        "animation": ["manim>=0.21.0"],
    },
    entry_points={
        "console_scripts": [
            "function-graph-generator=src.graph:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.txt"],
    },
    keywords="function call graph visualization analysis python parser",
    project_urls={
        "Bug Reports": "https://github.com/waifuai/function-graph-generator/issues",
        "Source": "https://github.com/waifuai/function-graph-generator",
    },
)