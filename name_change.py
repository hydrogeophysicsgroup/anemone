import os
import shutil

# for root, dirs, files in os.walk('.'):
#     for file in files:
#         if file.endswith('.py'):
#             fname = os.path.join(root, file)
#             newname = fname.replace('.py', '_tmp.py')
#             with open(fname, 'r') as fin:
#                 with open(newname, 'w') as fout:
#                     for line in fin.readlines():
#                         fout.write(line.replace('anemone', 'anemone'))
#             shutil.move(fname, fname.replace('.py', '_old.py'))
#             shutil.move(newname, fname)
            

#         if file.endswith('.ipynb'):
#             fname = os.path.join(root, file)
#             newname = fname.replace('.ipynb', '_tmp.ipynb')
#             with open(fname, 'r') as fin:
#                 with open(newname, 'w') as fout:
#                     for line in fin.readlines():
#                         fout.write(line.replace('anemone', 'anemone'))
#             shutil.move(fname, fname.replace('.ipynb', '_old.ipynb'))
#             shutil.move(newname, fname)

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('_old.py'):
            fname = os.path.join(root, file)
            os.remove(fname)

        if file.endswith('_old.ipynb'):
            fname = os.path.join(root, file)
            os.remove(fname)
