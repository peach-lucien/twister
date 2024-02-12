from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        subprocess.call(['python', 'post_install.py'])

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
    # Your package setup configuration here
    cmdclass={
        'install': PostInstallCommand,
    }
)

