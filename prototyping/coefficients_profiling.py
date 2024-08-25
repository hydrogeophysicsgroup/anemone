from em1d.transforms import hankel_coefficients

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()
print('ok')
hkl1 = hankel_coefficients(1, -20, 10, 10)
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
ps.print_stats()
print(s.getvalue())

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()
print('ok')
hkl1 = hankel_coefficients(1, -20, 10, 10)
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
ps.print_stats()
print(s.getvalue())