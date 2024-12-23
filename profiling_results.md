# Profiling results
## Forward
### initial_run 
| Function | Calls | Total Time | Per Call | Cumulative Time | Per Call |
| --- | --- | --- | --- | --- | --- |
| c:\Users\au646858\OneDrive - Aarhus universitet\Documents\_repositories\anemone\tests\forward_profiling.py:19(your_function_to_profile) | 1 | 0.000 | 4.279 | 4.279 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:127(__call__) | 1 | 0.000 | 4.279 | 4.279 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:87(forward) | 1 | 0.000 | 4.279 | 4.279 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:63(calc_ft) | 1 | 0.000 | 4.277 | 4.277 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:355(__init__) | 1 | 0.000 | 4.277 | 4.277 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:238(create_interpolator) | 3/2 | 0.007 | 4.266 | 2.133 | 0.020 |
| anemone\src\anemone\transforms\transforms.py:330(__init__) | 1 | 0.000 | 4.244 | 4.244 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:331(f1) | 1 | 0.004 | 4.187 | 4.187 | 0.004 |
| anemone\src\anemone\forwards\forwards.py:64(frq_integrand) | 1 | 0.001 | 4.183 | 4.183 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:204(__init__) | 2 | 0.000 | 3.782 | 1.891 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:67(wvlngth_integrand) | 1 | 0.550 | 3.580 | 3.580 | 0.550 |
| anemone\src\anemone\physics\physics.py:112(refl_transverse_electric) | 1 | 1.395 | 3.021 | 3.021 | 1.395 |
| {built-in method torch.sqrt} | 124 | 0.009 | 1.097 | 0.009 | 1.097 |
| {built-in method torch.tanh} | 1 | 0.532 | 0.532 | 0.532 | 0.532 |
| anemone\src\anemone\experiment\experiment.py:225(integrate_around_loop) | 1 | 0.142 | 0.408 | 0.408 | 0.142 |
| anemone\src\anemone\transforms\transforms.py:309(__call__) | 110 | 0.000 | 0.238 | 0.002 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:518(__call__) | 110 | 0.002 | 0.233 | 0.002 | 0.225 |
| anemone\src\anemone\transforms\transforms.py:268(_range) | 3 | 0.009 | 0.131 | 0.044 | 0.026 |
| anemone\src\anemone\transforms\transforms.py:439(__init__) | 3 | 0.000 | 0.117 | 0.039 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:447(compute_coefficients) | 3 | 0.037 | 0.117 | 0.039 | 0.111 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\functional.py:190(einsum) | 130 | 0.000 | 0.103 | 0.001 | 0.003 |
| {built-in method torch.einsum} | 130 | 0.001 | 0.096 | 0.001 | 0.096 |
| anemone\src\anemone\transforms\transforms.py:264(range) | 1 | 0.000 | 0.096 | 0.096 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:347(range) | 2 | 0.000 | 0.036 | 0.018 | 0.000 |
| {built-in method torch.zeros} | 20 | 0.001 | 0.020 | 0.001 | 0.020 |
| anemone\src\anemone\experiment\experiment.py:208(getRz) | 4 | 0.000 | 0.017 | 0.004 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:195(hankel_coefficients) | 3 | 0.000 | 0.016 | 0.005 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:143(optimized_coefficients_christensen1990) | 3 | 0.000 | 0.016 | 0.005 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:101(christensen1990_eq41) | 3 | 0.001 | 0.011 | 0.004 | 0.003 |
| anemone\src\anemone\transforms\transforms.py:44(christensen1990_eq16) | 27 | 0.000 | 0.009 | 0.000 | 0.006 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:34(wrapped) | 345 | 0.000 | 0.007 | 0.000 | 0.001 |
| anemone\src\anemone\experiment\experiment.py:204(norm) | 116 | 0.000 | 0.007 | 0.000 | 0.001 |
| anemone\src\anemone\physics\physics.py:96(freespace_u0) | 1 | 0.002 | 0.005 | 0.005 | 0.002 |
| {method 'pow' of 'torch._C._TensorBase' objects} | 341 | 0.000 | 0.005 | 0.000 | 0.005 |
| <frozen importlib._bootstrap>:1165(_find_and_load) | 2/1 | 0.000 | 0.004 | 0.004 | 0.000 |
| <frozen importlib._bootstrap>:1120(_find_and_load_unlocked) | 2/1 | 0.000 | 0.004 | 0.004 | 0.000 |
| {built-in method builtins.dir} | 111 | 0.000 | 0.003 | 0.000 | 0.003 |
| {built-in method torch.diff} | 9 | 0.000 | 0.003 | 0.000 | 0.003 |
| {built-in method torch.any} | 220 | 0.000 | 0.003 | 0.000 | 0.003 |
| <frozen importlib._bootstrap>:666(_load_unlocked) | 1 | 0.000 | 0.003 | 0.003 | 0.000 |
| <frozen importlib._bootstrap>:1054(_find_spec) | 2 | 0.000 | 0.003 | 0.001 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:2421(all) | 415 | 0.000 | 0.003 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:1496(find_spec) | 2 | 0.000 | 0.003 | 0.001 | 0.000 |
| <frozen importlib._bootstrap_external>:1464(_get_spec) | 2 | 0.000 | 0.003 | 0.001 | 0.000 |
| <frozen importlib._bootstrap_external>:934(exec_module) | 1 | 0.000 | 0.003 | 0.003 | 0.000 |
| <frozen importlib._bootstrap_external>:1604(find_spec) | 11 | 0.000 | 0.002 | 0.000 | 0.000 |
| {built-in method torch.sum} | 116 | 0.000 | 0.002 | 0.000 | 0.002 |
| {built-in method torch.exp} | 2 | 0.001 | 0.002 | 0.001 | 0.002 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:71(_wrapreduction) | 415 | 0.000 | 0.002 | 0.000 | 0.001 |
| <frozen importlib._bootstrap>:233(_call_with_frames_removed) | 3/2 | 0.000 | 0.002 | 0.001 | 0.000 |
| {built-in method builtins.exec} | 1 | 0.000 | 0.002 | 0.002 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\backends\opt_einsum\__init__.py:1(<module>) | 1 | 0.000 | 0.002 | 0.002 | 0.000 |
| {built-in method torch.searchsorted} | 110 | 0.000 | 0.002 | 0.000 | 0.002 |
| {built-in method torch.log10} | 121 | 0.000 | 0.002 | 0.000 | 0.002 |
| anemone\src\anemone\transforms\transforms.py:36(johansen1979_Hhat_nu) | 267 | 0.000 | 0.002 | 0.000 | 0.002 |
| anemone\src\anemone\physics\physics.py:21(calc_wavenumber) | 1 | 0.001 | 0.001 | 0.001 | 0.001 |
| <frozen importlib._bootstrap_external>:140(_path_stat) | 15 | 0.000 | 0.001 | 0.000 | 0.000 |
| {built-in method nt.stat} | 15 | 0.000 | 0.001 | 0.000 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:16(christensen1990_p_hat) | 362 | 0.000 | 0.001 | 0.000 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:85(christensen1990_eq17) | 12 | 0.000 | 0.001 | 0.000 | 0.001 |
| {method 'reduce' of 'numpy.ufunc' objects} | 415 | 0.000 | 0.001 | 0.000 | 0.001 |
| <frozen importlib._bootstrap_external>:96(_path_join) | 56 | 0.000 | 0.001 | 0.000 | 0.001 |
| anemone\src\anemone\physics\physics.py:12(calc_zy) | 1 | 0.001 | 0.001 | 0.001 | 0.001 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\warnings.py:96(_showwarnmsg) | 1 | 0.000 | 0.001 | 0.001 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\warnings.py:20(_showwarnmsg_impl) | 1 | 0.000 | 0.001 | 0.001 | 0.000 |
| {method 'to' of 'torch._C._TensorBase' objects} | 135 | 0.000 | 0.001 | 0.000 | 0.001 |
| {method 'write' of '_io.TextIOWrapper' objects} | 1 | 0.001 | 0.001 | 0.001 | 0.001 |
| <frozen importlib._bootstrap_external>:1007(get_code) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:46(xyz) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:980(__iter__) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'unbind' of 'torch._C._TensorBase' objects} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:72(<dictcomp>) | 415 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._has_torch_function} | 475 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:245(r_to_idx) | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:248(idx_to_r) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_VF.py:26(__getattr__) | 130 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.len} | 515 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:1127(get_data) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:230(_recenter_function) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:159(_path_isfile) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:150(_path_is_mode_type) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.isinstance} | 381 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.arange} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.zeros_like} | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'flatten' of 'torch._C._TensorBase' objects} | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:59(sigma) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:936(__rpow__) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:32(christensen1990_z_n) | 190 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.stack} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method io.open_code} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.pow} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:566(module_from_spec) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:911(__rdiv__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.clone} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.getattr} | 151 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:493(_init_module_attrs) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:119(<listcomp>) | 56 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:964(__len__) | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:1146(path_stats) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:169(__enter__) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.swapaxes} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.moveaxis} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:437(cache_from_source) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.floor} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'squeeze' of 'torch._C._TensorBase' objects} | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.from_numpy} | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:2416(_all_dispatcher) | 415 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._get_tracing_state} | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'read' of '_io.BufferedReader' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:392(cached) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:727(_compile_bytecode) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:567(_get_cached) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.ceil} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\warnings.py:117(_formatwarnmsg) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:179(_get_module_lock) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'items' of 'dict' objects} | 415 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:244(_verbose_message) | 53 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'endswith' of 'str' objects} | 174 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method marshal.loads} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'reciprocal' of 'torch._C._TensorBase' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'startswith' of 'str' objects} | 116 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'astype' of 'numpy.ndarray' objects} | 39 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\warnings.py:35(_formatwarnmsg_impl) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:1599(_get_spec) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:1421(_path_importer_cache) | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'reshape' of 'torch._C._TensorBase' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.complex} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.__build_class__} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\functools.py:479(lru_cache) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.squeeze} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\numeric.py:67(zeros_like) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'rstrip' of 'str' objects} | 170 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\lib\type_check.py:166(imag) | 72 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.ones_like} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.tensor} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:778(spec_from_file_location) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\functools.py:35(update_wrapper) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:112(xyz) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method '__exit__' of '_io._IOBase' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:132(_path_split) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'join' of 'str' objects} | 58 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:71(__init__) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\_distutils_hack\__init__.py:89(find_spec) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:173(__exit__) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method numpy.arange} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:100(acquire) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'append' of 'list' objects} | 59 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:67(_relax_case) | 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\contextlib.py:260(contextmanager) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:125(release) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.max} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\lib\type_check.py:162(_imag_dispatcher) | 72 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method _thread.allocate_lock} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:172(_path_isabs) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:748(find_spec) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:642(_classify_pyc) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.hasattr} | 16 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\linecache.py:26(getline) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:1030(__exit__) | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:134(<genexpr>) | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:920(find_spec) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'dim' of 'torch._C._TensorBase' objects} | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\backends\opt_einsum\__init__.py:93(OptEinsumModule) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method _imp.is_builtin} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:84(_unpack_uint32) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:1026(__enter__) | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:1207(_handle_fromlist) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'rpartition' of 'str' objects} | 16 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:357(__init__) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.setattr} | 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:198(cb) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\warnings.py:403(__init__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:675(_validate_timestamp_pyc) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'format' of 'str' objects} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\backends\opt_einsum\__init__.py:94(__init__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method _imp.acquire_lock} | 13 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\linecache.py:36(getlines) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:252(delta) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method _imp.release_lock} | 13 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:165(__init__) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:48(_new_module) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method numpy.zeros} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:599(_check_name_wrapper) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\backends\__init__.py:51(__init__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'rfind' of 'str' objects} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:138(__len__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:1097(__init__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\functools.py:65(wraps) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.iter} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'numel' of 'torch._C._TensorBase' objects} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._has_torch_function_unary} | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method nt._path_splitroot} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method _thread.get_ident} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.locals} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method _imp.find_frozen} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\lib\type_check.py:120(real) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'get' of 'dict' objects} | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\six.py:194(find_spec) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'disable' of '_lsprof.Profiler' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'pop' of 'list' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:7(christensen1990_calculate_smoothing) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method from_bytes} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method '__exit__' of '_thread.lock' objects} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method _imp._fix_co_filename} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'update' of 'dict' objects} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'strip' of 'str' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\backends\__init__.py:32(__init__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method nt.fspath} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'replace' of 'str' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\multiarray.py:85(empty_like) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:405(parent) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\multiarray.py:1080(copyto) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\lib\type_check.py:116(_real_dispatcher) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'pop' of 'dict' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap>:413(has_location) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\backends\opt_einsum\__init__.py:15(is_available) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\numeric.py:63(_zeros_like_dispatcher) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:1122(get_filename) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| <frozen importlib._bootstrap_external>:931(create_module) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\_distutils_hack\__init__.py:96(<lambda>) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.callable} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |

### run 2 
## Forward3
### initial_run 
| Function | Calls | Total Time | Per Call | Cumulative Time | Per Call |
| --- | --- | --- | --- | --- | --- |
| c:\Users\au646858\OneDrive - Aarhus universitet\Documents\_repositories\anemone\tests\forward_profiling.py:19(your_function_to_profile) | 1 | 0.000 | 2.647 | 2.647 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:127(__call__) | 1 | 0.000 | 2.647 | 2.647 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:480(forward) | 1 | 0.000 | 2.646 | 2.646 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:347(range) | 1 | 0.000 | 2.619 | 2.619 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:268(_range) | 2/1 | 0.006 | 2.619 | 2.619 | 0.012 |
| anemone\src\anemone\transforms\transforms.py:331(f1) | 1 | 0.000 | 2.608 | 2.608 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:481(frq_integrand) | 1 | 0.000 | 2.608 | 2.608 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:264(range) | 1 | 0.019 | 2.590 | 2.590 | 0.019 |
| anemone\src\anemone\forwards\forwards.py:484(wvlngth_integrand) | 1 | 0.337 | 2.499 | 2.499 | 0.337 |
| anemone\src\anemone\physics\physics.py:112(refl_transverse_electric) | 1 | 0.942 | 2.156 | 2.156 | 0.942 |
| {built-in method torch.sqrt} | 13 | 0.066 | 0.855 | 0.066 | 0.855 |
| {built-in method torch.tanh} | 1 | 0.359 | 0.359 | 0.359 | 0.359 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\functional.py:190(einsum) | 55 | 0.000 | 0.075 | 0.001 | 0.001 |
| {built-in method torch.einsum} | 55 | 0.001 | 0.074 | 0.001 | 0.074 |
| anemone\src\anemone\transforms\transforms.py:355(__init__) | 1 | 0.000 | 0.017 | 0.017 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:141(geomz) | 1 | 0.002 | 0.017 | 0.017 | 0.002 |
| anemone\src\anemone\transforms\transforms.py:330(__init__) | 1 | 0.000 | 0.017 | 0.017 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:195(hankel_coefficients) | 2 | 0.000 | 0.016 | 0.008 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:143(optimized_coefficients_christensen1990) | 2 | 0.000 | 0.016 | 0.008 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:204(__init__) | 2 | 0.000 | 0.013 | 0.006 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:10(lamontagne) | 8 | 0.001 | 0.012 | 0.002 | 0.006 |
| anemone\src\anemone\transforms\transforms.py:101(christensen1990_eq41) | 2 | 0.001 | 0.011 | 0.006 | 0.002 |
| anemone\src\anemone\transforms\transforms.py:44(christensen1990_eq16) | 23 | 0.000 | 0.011 | 0.000 | 0.008 |
| anemone\src\anemone\forwards\forwards.py:460(wv_to_t) | 1 | 0.001 | 0.006 | 0.006 | 0.001 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:34(wrapped) | 132 | 0.000 | 0.004 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:8(<lambda>) | 16 | 0.000 | 0.004 | 0.000 | 0.002 |
| {built-in method torch.exp} | 2 | 0.002 | 0.004 | 0.002 | 0.004 |
| {built-in method torch.log} | 10 | 0.000 | 0.003 | 0.000 | 0.003 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:907(__rsub__) | 96 | 0.000 | 0.003 | 0.000 | 0.000 |
| {built-in method torch.rsub} | 96 | 0.000 | 0.003 | 0.000 | 0.003 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:2421(all) | 304 | 0.000 | 0.003 | 0.000 | 0.000 |
| anemone\src\anemone\physics\physics.py:96(freespace_u0) | 1 | 0.000 | 0.003 | 0.003 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:7(<lambda>) | 16 | 0.000 | 0.002 | 0.000 | 0.001 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:71(_wrapreduction) | 304 | 0.000 | 0.002 | 0.000 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:16(christensen1990_p_hat) | 265 | 0.000 | 0.001 | 0.000 | 0.001 |
| anemone\src\anemone\experiment\experiment.py:208(getRz) | 4 | 0.000 | 0.001 | 0.000 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:36(johansen1979_Hhat_nu) | 172 | 0.000 | 0.001 | 0.000 | 0.001 |
| {method 'reduce' of 'numpy.ufunc' objects} | 304 | 0.000 | 0.001 | 0.000 | 0.001 |
| {built-in method torch.zeros} | 8 | 0.000 | 0.001 | 0.000 | 0.001 |
| {method 'sum' of 'torch._C._TensorBase' objects} | 4 | 0.000 | 0.001 | 0.000 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:85(christensen1990_eq17) | 8 | 0.000 | 0.001 | 0.000 | 0.000 |
| anemone\src\anemone\physics\physics.py:21(calc_wavenumber) | 1 | 0.001 | 0.001 | 0.001 | 0.001 |
| anemone\src\anemone\physics\physics.py:12(calc_zy) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'pow' of 'torch._C._TensorBase' objects} | 33 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'cpu' of 'torch._C._TensorBase' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:72(<dictcomp>) | 304 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:980(__iter__) | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:245(r_to_idx) | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:204(norm) | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:46(xyz) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'unbind' of 'torch._C._TensorBase' objects} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:248(idx_to_r) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:32(christensen1990_z_n) | 165 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_VF.py:26(__getattr__) | 55 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'to' of 'torch._C._TensorBase' objects} | 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.abs} | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.len} | 169 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:59(sigma) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:2416(_all_dispatcher) | 304 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._has_torch_function} | 187 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:911(__rdiv__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.sum} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.linspace} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'items' of 'dict' objects} | 304 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.stack} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'max' of 'torch._C._TensorBase' objects} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:936(__rpow__) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.clone} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.pow} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.log10} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.arange} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.moveaxis} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.ones_like} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.isinstance} | 63 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.from_numpy} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.swapaxes} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.squeeze} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'astype' of 'numpy.ndarray' objects} | 31 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:112(xyz) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\numeric.py:67(zeros_like) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'flatten' of 'torch._C._TensorBase' objects} | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._get_tracing_state} | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\lib\type_check.py:166(imag) | 58 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:964(__len__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.getattr} | 55 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'reciprocal' of 'torch._C._TensorBase' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'squeeze' of 'torch._C._TensorBase' objects} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.floor} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.ceil} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method numpy.arange} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\lib\type_check.py:162(_imag_dispatcher) | 58 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.complex} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:55(__len__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'dim' of 'torch._C._TensorBase' objects} | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'numel' of 'torch._C._TensorBase' objects} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:252(delta) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method numpy.zeros} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:138(__len__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.iter} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:7(christensen1990_calculate_smoothing) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\lib\type_check.py:120(real) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\multiarray.py:85(empty_like) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\multiarray.py:1080(copyto) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'disable' of '_lsprof.Profiler' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\numeric.py:63(_zeros_like_dispatcher) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\lib\type_check.py:116(_real_dispatcher) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._has_torch_function_unary} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |

### run 2 
## ForwardLaplace
### initial_run 
| Function | Calls | Total Time | Per Call | Cumulative Time | Per Call |
| --- | --- | --- | --- | --- | --- |
| c:\Users\au646858\OneDrive - Aarhus universitet\Documents\_repositories\anemone\tests\forward_profiling.py:19(your_function_to_profile) | 1 | 0.000 | 1.419 | 1.419 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:127(__call__) | 1 | 0.000 | 1.419 | 1.419 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:87(forward) | 1 | 0.000 | 1.419 | 1.419 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:293(calc_ft) | 1 | 0.000 | 1.418 | 1.418 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:204(__init__) | 1 | 0.000 | 1.323 | 1.323 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:238(create_interpolator) | 1 | 0.000 | 1.323 | 1.323 | 0.000 |
| anemone\src\anemone\forwards\forwards.py:294(wvlngth_integrand) | 1 | 0.000 | 1.314 | 1.314 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:417(__call__) | 1 | 0.022 | 1.312 | 1.312 | 0.022 |
| anemone\src\anemone\forwards\forwards.py:296(s_integrand) | 26 | 0.003 | 1.289 | 0.050 | 0.070 |
| anemone\src\anemone\physics\physics.py:112(refl_transverse_electric) | 26 | 0.018 | 1.216 | 0.047 | 0.461 |
| {built-in method torch.sqrt} | 170 | 0.003 | 0.532 | 0.003 | 0.532 |
| {built-in method torch.tanh} | 26 | 0.008 | 0.217 | 0.008 | 0.217 |
| anemone\src\anemone\experiment\experiment.py:225(integrate_around_loop) | 1 | 0.053 | 0.088 | 0.088 | 0.053 |
| anemone\src\anemone\transforms\transforms.py:309(__call__) | 109 | 0.000 | 0.027 | 0.000 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:518(__call__) | 110 | 0.000 | 0.025 | 0.000 | 0.020 |
| anemone\src\anemone\transforms\transforms.py:439(__init__) | 2 | 0.000 | 0.012 | 0.006 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:447(compute_coefficients) | 2 | 0.006 | 0.012 | 0.006 | 0.011 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:34(wrapped) | 420 | 0.000 | 0.006 | 0.000 | 0.001 |
| anemone\src\anemone\experiment\experiment.py:204(norm) | 116 | 0.000 | 0.004 | 0.000 | 0.001 |
| anemone\src\anemone\transforms\transforms.py:264(range) | 1 | 0.000 | 0.004 | 0.004 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:268(_range) | 1 | 0.001 | 0.004 | 0.004 | 0.001 |
| {method 'pow' of 'torch._C._TensorBase' objects} | 392 | 0.000 | 0.004 | 0.000 | 0.004 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\functional.py:190(einsum) | 14 | 0.000 | 0.003 | 0.000 | 0.000 |
| anemone\src\anemone\physics\physics.py:21(calc_wavenumber) | 26 | 0.000 | 0.003 | 0.000 | 0.003 |
| {built-in method torch.einsum} | 14 | 0.000 | 0.003 | 0.000 | 0.003 |
| anemone\src\anemone\physics\physics.py:12(calc_zy) | 26 | 0.000 | 0.003 | 0.000 | 0.003 |
| {built-in method builtins.dir} | 110 | 0.000 | 0.002 | 0.000 | 0.002 |
| {built-in method torch.any} | 220 | 0.000 | 0.002 | 0.000 | 0.002 |
| anemone\src\anemone\transforms\transforms.py:402(__init__) | 1 | 0.000 | 0.002 | 0.002 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:911(__rdiv__) | 27 | 0.000 | 0.001 | 0.000 | 0.001 |
| {built-in method torch.sum} | 116 | 0.000 | 0.001 | 0.000 | 0.001 |
| anemone\src\anemone\forwards\forwards.py:59(sigma) | 26 | 0.000 | 0.001 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:371(gaver_stehfest_coefficients) | 1 | 0.001 | 0.001 | 0.001 | 0.001 |
| {built-in method torch.zeros} | 15 | 0.000 | 0.001 | 0.000 | 0.001 |
| {built-in method torch.log10} | 115 | 0.000 | 0.001 | 0.000 | 0.001 |
| {built-in method torch.searchsorted} | 110 | 0.000 | 0.001 | 0.000 | 0.001 |
| {method 'reciprocal' of 'torch._C._TensorBase' objects} | 27 | 0.000 | 0.001 | 0.000 | 0.001 |
| anemone\src\anemone\experiment\experiment.py:208(getRz) | 4 | 0.000 | 0.001 | 0.000 | 0.001 |
| {built-in method torch.ones_like} | 52 | 0.000 | 0.001 | 0.000 | 0.001 |
| {built-in method builtins.len} | 72 | 0.000 | 0.001 | 0.000 | 0.000 |
| {built-in method torch.squeeze} | 29 | 0.000 | 0.001 | 0.000 | 0.001 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:964(__len__) | 29 | 0.000 | 0.001 | 0.000 | 0.000 |
| {built-in method torch.arange} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.diff} | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\physics\physics.py:96(freespace_u0) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'to' of 'torch._C._TensorBase' objects} | 122 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._get_tracing_state} | 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:46(xyz) | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method numpy.arange} | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._has_torch_function} | 434 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:2100(clip) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:53(_wrapfunc) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'clip' of 'numpy.ndarray' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\_methods.py:90(_clip) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.isinstance} | 290 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:245(r_to_idx) | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'flatten' of 'torch._C._TensorBase' objects} | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:248(idx_to_r) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.stack} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:363(factorial) | 84 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:980(__iter__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.exp} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.zeros_like} | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:230(_recenter_function) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.moveaxis} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.swapaxes} | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'squeeze' of 'torch._C._TensorBase' objects} | 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_tensor.py:936(__rpow__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\torch\_VF.py:26(__getattr__) | 14 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.tensor} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.pow} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:112(xyz) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'unbind' of 'torch._C._TensorBase' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'dim' of 'torch._C._TensorBase' objects} | 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.clone} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.floor} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.from_numpy} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch.ceil} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method torch._C._has_torch_function_unary} | 29 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'reshape' of 'torch._C._TensorBase' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.getattr} | 15 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'numel' of 'torch._C._TensorBase' objects} | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\experiment\experiment.py:138(__len__) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| anemone\src\anemone\transforms\transforms.py:252(delta) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method numpy.zeros} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {built-in method builtins.iter} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| {method 'disable' of '_lsprof.Profiler' objects} | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| C:\Users\au646858\Miniconda3\envs\em1d\Lib\site-packages\numpy\core\fromnumeric.py:2096(_clip_dispatcher) | 1 | 0.000 | 0.000 | 0.000 | 0.000 |

### run 2 
