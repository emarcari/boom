
from pyboom import *
import os

be = boomenv('lib')

home = os.environ['HOME']
homeinc = os.path.join(home, 'include')

# ev = os.environ
# for vname in ev:
#    print (vname, ev[vname])

### tr1 = tr1path()

Libhome = os.environ['PWD']
OtherInc = [homeinc]
flags = be['flags']

BuildDir('debug', 'src')
flags += " -g "
Libname= 'BOOMdebug'
Export('flags Libname Libhome OtherInc be')
SConscript('debug/SConscript')

BuildDir('opt', 'src')
flags += " -O2 -DNDEBUG -funroll-loops "
Libname= 'BOOM'
Export('flags Libname Libhome OtherInc be')
SConscript('opt/SConscript')

prof = ARGUMENTS.get('prof',0)
if prof:
   BuildDir('prof','src')
   print 'building profile library'
   flags += " -O2 -DNDEBUG -pg -funroll-loops "
   Libname= 'BOOMprof'
   Export('flags Libname Libhome OtherInc be')
   SConscript('prof/SConscript')
