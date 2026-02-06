from setuptools import setup, Extension, find_packages

setup(  
    name='anemone',
    author='Matt Griffiths',
    author_email='mpg@geo.au.dk',
    packages=['anemone', 'anemone.system', 'anemone.physics', 'anemone.transforms', 'anemone.inversion', 'anemone.forwards'],
    package_dir={'':'src'},        
    install_requires=['numpy', 'matplotlib', 'scipy', 'torch', 'nosj'],
    license='MIT',
    description='electromagnetic forward',       
    version='0.1.0' 
)
