from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
# print(find_packages('src', exclude=['tests', 'em1d.cphysics']))

physics_ext = cpp_extension.CppExtension('em1d.cphysics', ['src/em1d/cphysics/cphysics.cpp'])

setup(  
    name='em1d',
    author='Matt Griffiths',
    author_email='mpg@geo.au.dk',
    packages=['em1d', 'em1d.experiment', 'em1d.physics', 'em1d.transforms'],
    package_dir={'':'src'},        
    ext_modules=[physics_ext],
    install_requires=['numpy', 'matplotlib', 'scipy'],
    license='MIT',
    description='electromagnetic forward',        
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
