"""Setup configuration for agent-reliability-patterns."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agent-reliability-patterns",
    version="0.1.0",
    author="Goutham Patley",
    author_email="goutham@example.com",
    description="Production-ready reliability patterns for AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hamley241/agent-reliability-patterns",
    packages=find_packages(exclude=["tests", "benchmarks", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
