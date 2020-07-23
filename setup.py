from setuptools import setup, find_packages
from glob import glob

setup(
    name = 'starme',
    packages = find_packages(),
    include_package_data=True,
    version = '20.07.23.0',
    license = 'GNU GPLv3',
    platforms = 'any',
    description = 'Star Me Project',
    author = 'Julio Trevisan',
    author_email = 'juliotrevisan@gmail.com',
    url = 'http://github.com/trevisanj/starme',
    keywords= [],
    install_requires = ['a107'],
    python_requires = '>=3',
    scripts = glob('starme/scripts/*.py')
)
