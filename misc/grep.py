import subprocess as sub
import numpy as np

def grep(string, f, A = None, B = None):
    cmd = """grep "%s" %s""" % (string, f)
    if A is not None:
        cmd += " -A %d" % A
    if B is not None:
        cmd += " -B %d" % B
    cmd += "; exit 0"
    return sub.check_output(cmd, shell = True)[:-1]

