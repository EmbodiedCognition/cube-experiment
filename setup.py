import os
import setuptools

setuptools.setup(
    name='lmj.cubes',
    version='0.0.1',
    namespace_packages=['lmj'],
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='analysis code for cube experiment',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.rst')).read(),
    license='MIT',
    url='http://github.com/lmjohns3/cube-experiment',
    install_requires=['climate', 'numpy', 'pandas', 'scipy'],
)
