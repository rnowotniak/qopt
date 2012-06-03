#!/usr/bin/python
'''
screener
========
'''

import sys
import os
from subprocess import Popen
import time
import os.path as osp


def run(screen_name, command):
    '''
    Run a command in a background screen. If necessary, retry 10 times if the screen didn't start.

    :params string screen_name: Name for the screen (to be identifiable)
    :params string command: Command to run in this screen
    '''
    command = "touch %s_started;%s" % (screen_name, command)
    abs_path = osp.join(osp.dirname(osp.abspath(__file__)))

    f = open('screenrcs/%s.rc' % screen_name, 'w')
    f.write('shell bash\n')
    f.write('deflog on\n')
    f.write('logfile screenlogs/%s.log\n' % screen_name)
    f.close()

    counter = 1
    while counter < 10:
        # here I assume that bgscreen lies in the same directory as
        # screener.py (this file)
        Popen("%s/bgscreen %s '%s'" % (abs_path, screen_name, command), shell=True).wait()
        time.sleep(1)
        if os.path.exists("%s_started" % screen_name):
            print "[Nicessa] I ran %s on screen %s ..." % (command, screen_name)
            os.remove("%s_started" % screen_name)
            counter = 10
        else:
            print 're-trying to start screen %s' % screen_name
            Popen("kill `ps aux | awk '/%s/{print $2}'`;" % (screen_name), shell=True).wait()
            counter += 1
    print "Done. Screen %s is running." % screen_name


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
