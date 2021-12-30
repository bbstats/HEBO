import subprocess

process = subprocess.Popen(["./src/AbsolutNoLib",'repertoire', '1ADQ' , 'dummyCDR.txt','1'])
process.communicate()
import pdb
pdb.set_trace()