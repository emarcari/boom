"""
Set up scons to build programs using BOOM.
   usage:  boomenv(build_type)
   where build_type is one of ['opt', 'debug', 'prof']

Handles boost, lapack, and BOOM libraries.
"""
import os
###======================================================================
def check_path(path_list):
    ans = []
    for p in path_list :
        if(os.path.exists(p)):
            ans+= [p]
    return ans
###======================================================================
def tr1path():
    ans = os.path.join(os.environ['HOME'], 'src', 'boost_1_37_0', 'boost', 'tr1', 'tr1')
    ans= os.path.join(os.environ['HOME'], 'include')
    return ans
###======================================================================
def boomenv(build_type='opt'):
    home = os.environ['HOME']
    boomhome = os.path.join(home, 'BOOM2')

## set path names
    libpath = [boomhome, os.path.join(home, 'lib')]
    libpath = check_path(libpath)

    incpath = [ tr1path(),
                os.path.join(boomhome, 'src'),
                os.path.join(home, 'include')
                ]

    incpath = check_path(incpath)

    boost_libs = ['program_options', 'filesystem', 'thread',
                  'signals', 'system']
    boost_prefix     = 'boost_'
    boost_suffix     = '-gcc43-mt'

## set compiler flags
###    linkflags = '-Wl,--enable-auto-import '
    linkflags = ''
    flags = ' -std=c++0x '
    if build_type=='opt':
        flags += ' -O3 '
        boomlib = ['BOOM']
        Wflags = " -Wall -Wunused -Wextra -Wctor-dtor-privacy "
        moreW = " -Wfloat-equal -Werror -Wsign-promo -Wno-uninitialized "
        evenmoreW = " -Woverloaded-virtual "
        flags = flags + Wflags + moreW + evenmoreW

    elif build_type=='debug':
        flags += ' -g '
        boost_suffix += ''
        boomlib = ['BOOMdebug']
    elif build_type=='prof':
        flags += ' -O3 -pg '
        boomlib = ['BOOMprof']
        linkflags = linkflags + ' -pg '
    elif build_type=='lib':
        flags += ''
        boomlib = []
    else:
        err= "unrecognized option for build_type "+ build_type
        raise err

    boost_suffix += '-1_37'
    for i in range(len(boost_libs)):
        x = boost_prefix + boost_libs[i] + boost_suffix
        boost_libs[i] = x

###    lapack_libs = ['lapack', 'f77blas', 'g2c', 'cblas', 'atlas', 'gfortran']

    lapack_libs = ['lapack', 'f77blas', 'cblas', 'atlas', 'gfortran']
    libs = boomlib + lapack_libs + boost_libs + ['m']

    ans = {}
    ans['CC'] =   "/home/stevescott/bin/gcc"
    ans['CXX'] =   "/home/stevescott/bin/g++"
    ans['libpath'] = libpath
    ans['incpath'] = incpath
    ans['flags'] = flags
    ans['libs'] = libs
    ans['linkflags'] = linkflags
    return(ans)
