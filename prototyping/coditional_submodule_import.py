# submodule.py
import sys

def submodule_function():
    if 'torch' in sys.modules:
        import torch
        print("Using torch in the submodule")
        # Your code using torch goes here
    elif 'numpy' in sys.modules:
        import numpy
        print("Using numpy in the submodule")
        # Your code using numpy goes here
    else:
        print("Neither torch nor numpy is imported in the main script")

# main_script.py
import torch
# import numpy  # Uncomment this line to import numpy instead

def main_function():
    print("This is the main script")

if __name__ == "__main__":
    main_function()
    submodule_function()
