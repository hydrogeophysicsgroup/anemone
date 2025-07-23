import sys
import os
import subprocess
import tempfile

import pytest

file_path = os.path.split(__file__)[0]
example_path = file_path.replace('\\tests', '\\examples\\')#os.path.join(file_path, '..\\examples\\')
sys.path.append(example_path)

print(example_path)

pyfiles = [x for x in os.listdir(example_path) if x.endswith('.py') and not x.startswith('__init__')]
nbfiles = [x for x in os.listdir(example_path) if x.endswith('.ipynb') and 'gpu' not in x.lower()]

def silence(s):
    pass

def test_example_scripts(log=silence):
    """Test only Python files"""
    import importlib.util
    
    successful_run = []
    log("Testing Python files:")
    
    for pyfile in pyfiles:
        try:
            module_name = pyfile  # Remove the .py extension
            
            module_path = os.path.join(example_path, pyfile)

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            log(f"✅\t{module_name}")
            successful_run.append(True)
        except Exception as e:
            log(f"❌\t{module_name} - Error: {str(e)}")
            successful_run.append(False)

    assert all(successful_run), "Some example scripts failed to run"
    


@pytest.mark.skipif(reason="test takes too long")
def test_notebooks(log=silence):
    """Test Jupyter notebooks using nbconvert"""
    successful_run = []
    log("Testing Jupyter notebooks:")
    
    # Try different ways to run jupyter nbconvert
    jupyter_commands = [
        [sys.executable, '-m', 'nbconvert'],
        ['jupyter', 'nbconvert'],
        ['python', '-m', 'jupyter', 'nbconvert'],
        ['py', '-m', 'jupyter', 'nbconvert'],
        ['py', '-m', 'nbconvert']
    ]
    
    # Test which jupyter command works
    working_cmd = None
    for cmd in jupyter_commands:
        try:
            result = subprocess.run(cmd + ['--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                working_cmd = cmd
                log(f"Using command: {' '.join(cmd)}")
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    if working_cmd is None:
        log("❌ Could not find jupyter nbconvert. Please install with: pip install jupyter nbconvert")
        return [False] * len(nbfiles)
    
    # nbfiles = [x for x in nbfiles if 'gpu' not in x]  # Exclude GPU notebooks

    for nbfile in nbfiles:
        try:
            nb_path = os.path.join(example_path, nbfile)
            # log(f"Running notebook: {nbfile}")
            
            # Use nbconvert to execute the notebook
            with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Execute notebook and save output to temporary file
                result = subprocess.run(
                    working_cmd + ['--execute', '--to', 'notebook', '--output', tmp_path, nb_path],
                    capture_output=True, text=True, timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    log(f"✅\t{nbfile}")
                    successful_run.append(True)
                else:
                    log(f"❌\t{nbfile} - Error: {result.stderr}")
                    successful_run.append(False)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except subprocess.TimeoutExpired:
            log(f"❌\t{nbfile} - Timeout (>5 minutes)")
            successful_run.append(False)
        except Exception as e:
            log(f"❌\t{nbfile} - Error: {str(e)}")
            successful_run.append(False)
    
    assert all(successful_run), "Some notebooks failed to run"

if __name__ == "__main__":
    test_example_scripts(log=print)
    test_notebooks(log=print)