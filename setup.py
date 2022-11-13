from setuptools import find_packages, setup

setup(
    name="HybridNetworksProject",
    version="1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "generate_bash_script = common_packages.generate_bash_script:main",
        ],
    },
)
