from setuptools import setup, Extension, find_packages

setup(  
    name='em1d',
    author='Matt Griffiths',
    author_email='mpg@geo.au.dk',
    packages=['em1d', 'em1d.experiment', 'em1d.physics', 'em1d.transforms', 'em1d.inversion'],
    package_dir={'':'src'},        
    install_requires=['numpy', 'matplotlib', 'scipy', 'torch'],
    license='MIT',
    description='electromagnetic forward',        
)
