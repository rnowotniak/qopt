#!/usr/bin/python

'''
tester
=======
'''

import sys
import os.path as osp
from shutil import rmtree
from subprocess import Popen

from sim import utils
from analysis import harvester


def ttest(simfolder, c, i, delim):
    '''
    Run a T-test with Gnu R

    :param string simfolder: path to simulation folder
    :param ConfigParser c: the config file where this test is described
    :param int i: number of test in that config file
    '''
    tmp_dir = '%s/tmp_tester' % simfolder

    # -- prepare data --
    # for each of the two datasets, retrieve files and then values
    sets = []
    for dset in [1,2]:
        d = utils.decode_search_from_confstr(c.get('ttest%i' % i, 'set%i' % dset), sim = c.get('meta', 'name'))
        if d.has_key('_name') and d.has_key('_col') and d['_col'].isdigit():
            sets.append(d['_name'])
            searches = {d['_name'] : [(k, d[k]) for k in d.keys() if not k in ['_name', '_col']]}
            harvester.collect_files(searches, "%s/data" % simfolder, tmp_dir)
            #TODO: custom selectors?
            if not d.has_key('_select'):
                d['_select'] = 'all'
            harvester.collect_values("%s/%s" % (tmp_dir, d['_name']),
                                     delim,
                                     '%s/%s.dat' % (tmp_dir, d['_name']),
                                     cols=[int(d['_col'])],
                                     selector=d['_select']
                                    )
        else:
            print '[NICESSA] Warning: Incomplete T-test specification for test %i in Experiment %s, dataset number %i. '\
                  'Specify at least _name and _col.' % (i, c.get('meta', 'name'), dset)

    # -- run test --
    if osp.exists('%s/%s.dat' % (tmp_dir, sets[0])) and osp.exists('%s/%s.dat' % (tmp_dir, sets[1])):
        if c.has_option('ttest%i' %i, 'custom-script'):
            custom_script = c.get('ttest%i' % i, 'custom-script')
            if not osp.exists(custom_script):
                print "[Nicessa] Cannot find custom script at [%s]. Aborting ..." % (custom_script)
                print
                return
            print '[NICESSA] Using custom script at %s' % custom_script
            Popen('cp %s %s/ttest.r' % (custom_script, tmp_dir), shell=True).wait()
        else:
            rscript = open('%s/ttest.r' % tmp_dir, 'w')
            rscript.write("%s <- read.table('%s.dat')\n" % (sets[0], sets[0]))
            rscript.write("%s <- read.table('%s.dat')\n" % (sets[1], sets[1]))
            rscript.write("t.test(%s,%s)" % (sets[0], sets[1]))
            rscript.close()
        Popen('cd %s; R --vanilla --silent < ttest.r' % tmp_dir, shell=True).wait()

    if osp.exists(tmp_dir) and not '-k' in sys.argv:
        rmtree(tmp_dir)



