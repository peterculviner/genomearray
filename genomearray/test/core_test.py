# code for local testing of genomearray code on laublab server
import sys, os
import unittest
sys.path.append(os.path.relpath("home/laublab/notebooks/dropbox_link/culviner/repositories/genomearray/"))

import genomearray as ga
print 'dir output:'
print dir(ga)

## test core functionality ##
