import os
import setuptools

setuptools.setup(
    name='cubes',
    version='0.0.1',
    packages=setuptools.find_packages(),
    description='analysis code for cube experiment',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.rst')).read(),
    license='MIT',
    url='http://github.com/EmbodiedCognition/cube-experiment',
    install_requires=[
        'click', 'joblib', 'matplotlib', 'numpy',
        'pandas', 'scikit-learn', 'scipy', 'seaborn',
    ],
)
