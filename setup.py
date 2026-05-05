from setuptools import setup, find_packages

setup(
    name="gulls_pipeline",
    version="0.1.0",
    description="Python wrapper and Web UI for the GULLS microlensing simulator",
    author="Malpas",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
    ],
    entry_points={
        "console_scripts": [
            "gulls-ui = gulls_pipeline.server:main",
            "gulls-runner = gulls_pipeline.cli:main",
        ]
    },
)