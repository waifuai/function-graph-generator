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

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

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
        "License :: OSI Approved :: MIT-0 License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Documentation",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
        "yaml": ["pyyaml>=6.0"],
        "dot": ["pydot>=1.4.0"],
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