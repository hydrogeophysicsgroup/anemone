import sys
import os

file_path = os.path.split(__file__)[0]
example_path = os.path.join(file_path, '..\\examples\\')
sys.path.append(example_path)

print(example_path)

pyfiles = [x for x in os.listdir(example_path) if x.endswith('.py') and not x.startswith('__init__')]

def silence(s):
    pass

def test_examples(log=silence):
    import importlib.util

    for pyfile in pyfiles:
        successful_run = []
        try:
            module_name = pyfile  # Remove the .py extension
            
            module_path = os.path.join(example_path, pyfile)

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            log(f"✅\t{module_name}")
            successful_run.append(True)
        except Exception as e:
            #raise ImportError(f"Failed to import {module_name}: {e}")
            log(f"❌\t{module_name}")
            successful_run.append(False)
            continue
    assert all(successful_run), f"Some examples failed to run"

if __name__ == "__main__":
    test_examples(log=print)
    print("All examples tested successfully.")