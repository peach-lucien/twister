from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="twister",
    version="1.0.1",
    description="Twstr analysis package",
    author="Robert Peach",
    author_email="peach_r@ukw.de",
    packages=find_packages(),
    include_package_data=True,  # This line is important
    install_requires=requirements,
    entry_points={"console_scripts": ["twister=twister.app:cli"]},
)
