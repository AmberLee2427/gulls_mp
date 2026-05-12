from setuptools import setup, find_namespace_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    """Mark wheels as platform-specific because they can contain Gulls binaries."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat

setup(
    name="gulls_pipeline",
    version="0.1.0",
    description="Python wrapper and Web UI for the GULLS microlensing simulator",
    author="Malpas",
    packages=find_namespace_packages(
        include=["gulls_pipeline", "gulls_pipeline.*", "smoke_test", "smoke_test.*"]
    ),
    include_package_data=True,
    package_data={
        "gulls_pipeline": ["web/*.html", "bin/*", "runtime/src/*"],
        "gulls_pipeline.bin": ["*"],
        "gulls_pipeline.runtime": ["src/*"],
        "gulls_pipeline.web": ["*.html"],
        "smoke_test": [
            "README.md",
            "smoke.yml",
            "assets/lenses/*",
            "assets/observatories/*",
            "assets/planets/*",
            "assets/rates/*",
            "assets/sources/*",
            "assets/starfields/*",
            "assets/weather/*",
            "parameterfiles/*.prm",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "fastapi",
        "uvicorn",
    ],
    entry_points={
        "console_scripts": [
            "gulls-ui = gulls_pipeline.server:main",
            "gulls-runner = gulls_pipeline.cli:main",
            "gulls-smoke-test = smoke_test.runner:main",
        ]
    },
    cmdclass={"bdist_wheel": bdist_wheel},
)
