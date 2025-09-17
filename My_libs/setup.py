from setuptools import setup, find_packages

setup(
    name='db-utils-postgres',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'psycopg2-binary',
    ],
    author='Felipe',
    description='Uma biblioteca para gerenciar a conexão com PostgreSQL.',
)