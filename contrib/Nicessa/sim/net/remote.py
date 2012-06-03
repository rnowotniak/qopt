'''
remote
======
Code to realise runnning simulations remotely
All remote communication is done via SSH with the help of paramiko
'''

import sys
import os
import os.path as osp
import time
from subprocess import Popen
from getpass import getpass
# ignore deprecation warnings that paramiko currently delivers
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
import paramiko
import scp

from sim import utils


def ssh(client, cmd, ignore=[]):
    '''
    Run cmd on remote client (retry if connection fails), show errors (if we care about them)

    :params paramiko.SSHClient client:
    :params string cmd:
    :params string ignore: Error messages to ignore for this cmd
    :returns: stdout from host (minus some things we consider irrelevant)
    '''
    done = False
    while not done:
        try:
            stdin, stdout, stderr = client.exec_command(cmd)
            done = True
        except:
           time.sleep(4)
    err = stderr.read()
    dontcare_snippets = ['xset:', 'cannot remove', 'cannot access finished_*',\
                         'are the same file', 'finished_*: No such']
    dontcare_snippets.extend(ignore)
    err_out = ""
    for e_line in err.split('\n'):
        yell_it = True
        for s in dontcare_snippets:
            if s in e_line:
                yell_it = False
        if yell_it and e_line != "":
            err_out += '%s\n' % e_line
    if err_out.strip() != "":
        print "[Nicessa] Error while doing stuff on server: %s" % (err_out)
    return "[Nicessa] Log from remotely executing [%s]:\n\n\n%s" % (cmd, stdout.read())


def run_remotely(simfolder, conf):
    '''
    Run the simulation on remote hosts. Reads parameterisation and remote setup from the conf-directory. 

    :param string simfolder: relative path to simfolder
    :param ConfigParser conf: main config
    :returns: True if successful, False otherwise
    '''
    folder = simfolder
    remote_conf = utils.get_host_conf(simfolder)
    num_hosts = utils.num_hosts(simfolder)
    working_cpus_per_host = utils.working_cpus_per_host(folder)
    ssh_clients = {} # we login twice to each if it has work

    if not folder == ".":
        os.chdir(folder)

    logdir = 'deploy-logs'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    else:
        for logf in os.listdir(logdir):
            os.remove('%s/%s' % (logdir, logf))

    print "[Nicessa] Preparing hosts ..."
    for host in [h for h in xrange(1, num_hosts+1) if working_cpus_per_host[h] > 0]:
        ssh_clients[host] = _get_ssh_client(remote_conf, host)
        if ssh_clients[host] is None:
            print "[Nicessa] Cannot connect to host %d. Aborting ...  " % host
            return False
        if not remote_conf.has_section("host%i" % host):
            print "[Nicessa] Host %d is not configured!" % host
            return False

        path = '%s/%s' % (remote_conf.get("host%i" % host, "path"), utils.make_simdir_name(simfolder))
        # ------------- clean host (we want to be sure to use fresh code)
        # do this on all hosts before anything is run (e.g. they could operate on the same home dir)
        if not check_data(simfolder, host):
            print '[Nicessa] Aborting.'
            return False
        cleaning = "mkdir -p %s/%s;" % (path, folder)
        togo = "data conf nicessa.conf %s bgscreen screener.py starter.py _nicessa_bundle.tar.gz" \
                % (conf.get('control', 'executable'))
        if remote_conf.has_option('code', 'files'):
            for f in [f for f in remote_conf.get('code', 'files').split(',') if f is not ""]:
                togo += " %s/%s " % (simfolder, f.strip())
        if remote_conf.has_option('code', 'folders'):
            for f in [f for f in remote_conf.get('code', 'folders').split(',') if f is not ""]:
                togo += " %s/%s " % (simfolder, f.strip())
        cleaning += "cd %s/%s; rm -r %s;" % (path, folder, togo)
        # make fresh dirs to config and log screens
        cleaning += 'mkdir -p screenrcs; rm -r screenrcs/*; mkdir -p screenlogs; rm -r screenlogs/*;'
        # clean old states, too - never know how the last run was finished (e.g. Ctrl-C)
        cleaning += clean_states(simfolder, host)
        ssh(ssh_clients[host], cleaning)

    used_hosts = 0
    for host in [h for h in xrange(1, num_hosts+1) if working_cpus_per_host[h] > 0]:
        # -------------  initialize each host
        # don't proceed if a host doesn't have work to do (TODO: maybe also don't clean before?)
        host_has_work = False
        if osp.exists("%s/conf/%d" % (folder, host)):
            host_has_work = True
            used_hosts += 1
        if not host_has_work:
            continue

        # ------------- start screen(s) on all of them
        # let him run the batch for this host in a background screen (for each simulation we might have)
        # TODO: remote works only with subsimulations right now? We should fix that
        screening = ""
        for cpu in xrange(1, working_cpus_per_host[host]+1):
            if osp.exists("%s/conf/%d/%d" % (simfolder, host, cpu)):
                screen_name = utils.make_screen_name(simfolder, host, cpu)
                nice_level = utils.get_nice_level(simfolder, host)
                screening += "rm finished_%s; ./screener.py %s 'nice -n %d ./starter.py . %i %i; touch finished_%s;exit;'; " \
                    % (screen_name, screen_name, nice_level, host, cpu, screen_name)

        #  --- local file shuffling
        # all host-side screen calls go in a script file, so screener.py can quietly make sure they all start without me waiting
        cmd = open("cmd_%d" % host, 'w')
        cmd.write(screening)
        cmd.flush()
        cmd.close()
        Popen("chmod +x cmd_%d;" % host, shell=True).wait();
        needed = " cmd_%d" % host

        # send everything needed of the simulation to run batches to the host in one go
        needed += " conf"
        needed += " nicessa.conf"
        # user should specify to copy this in remote.conf, since  control->executable can
        # contain any command, not only a filename
        #needed += " %s" % (conf.get('control', 'executable'))
        if remote_conf.has_option('code', 'files'):
            for f in [f for f in remote_conf.get('code', 'files').split(',') if f is not ""]:
                needed += " %s" %f
        if remote_conf.has_option('code', 'folders'):
            for f in [f for f in remote_conf.get('code', 'folders').split(',') if f is not ""]:
                needed += " %s" %f
        # also some nicessa files
        pth = osp.join(osp.dirname(osp.abspath(__file__)))
        copied_here = []
        for filename in ['bgscreen','screener.py','starter.py']:
            Popen("cp %s/%s ." % (pth, filename), shell=True).wait()
            copied_here.append(filename)
            needed += " %s" % filename.split('/')[-1:][0]

        # put all we need in a tar.gz archive
        Popen("tar -cf _nicessa_bundle.tar %s; gzip -f _nicessa_bundle.tar;" % (needed), shell=True).wait()

        # ------------ here we actually connect and do all these things online
        path = '%s/%s' % (remote_conf.get("host%i" % host, "path"), utils.make_simdir_name(simfolder))
        if ssh_clients[host] is None:
            return False
        try:
            print "[Nicessa] Running code on %s" %  remote_conf.get("host%i" % host, "name")
            scp_client = scp.SCPClient(ssh_clients[host]._transport)
            scp_client.put("_nicessa_bundle.tar.gz", remote_path="%s/%s" % (path, folder))
            time.sleep(2)
            initializing = "cd %s/%s; tar -zxf _nicessa_bundle.tar.gz;" % (path, folder)
            log = open('%s/log%d' % (logdir, host), 'w')
            log.write(ssh(ssh_clients[host], "%s ./cmd_%d; rm cmd_%d;" % (initializing, host, host)))
            log.flush()
            log.close()
        except scp.SCPException, e:
            print e
        ssh_clients[host].close()

        # ------------ clean locally
        os.remove("_nicessa_bundle.tar.gz")
        os.remove("cmd_%d" % (host))
        for c in copied_here:
            os.remove("%s" % c)
        # --- end local file shuffling

    if not folder == ".":
        for sf in folder.split("/"):
            if sf != '':
                os.chdir('..')

    print "[Nicessa] deployed simulation on %i host(s)" % (used_hosts)
    return True


def check_states(simfolder):
    '''
    Performs a check on the status of the simulations.
    For this, it looks at the marker files a job creates when it is done and the names
    of currently running screens.
    It prints out the contents of the first search as finished and the second as still running.
    If no jobs are finished or running, it prints a message.

    :param string simfolder: relative path to simfolder
    :returns: True if successful, False otherwise
    '''
    conf = utils.get_main_conf(simfolder)
    hosts = utils.num_hosts(simfolder)
    working_cpus_per_host = utils.working_cpus_per_host(simfolder)
    finished = {}
    running = {}
    for host in [h for h in xrange(1, hosts+1) if working_cpus_per_host[h] > 0]:
        finished[host] = []
        running[host] = []
    remote_conf = utils.get_host_conf(simfolder)
    found_jobs = 0

    # TODO: if shared-home is set to 1, we could save us some time (only connect
    # to one host), but here, the development time does not justify the reward,
    # I think.
    print "[Nicessa] Checking hosts: ",
    sys.stdout.flush()
    for host in xrange(1, hosts+1):
        if working_cpus_per_host[host] > 0:
            hostname = remote_conf.get("host%i" % host, "name")
            print "%s (host-nr:%d, cpus:%d)  " % (hostname, host, working_cpus_per_host[host]),
            sys.stdout.flush()
            ssh_client = _get_ssh_client(remote_conf, host)
            if ssh_client:
                path = '%s/%s' % (remote_conf.get("host%i" % host, "path"), utils.make_simdir_name(simfolder))
                fin = ssh(ssh_client, 'cd %s/%s; ls finished_*;' % (path, simfolder), ignore=['No such file'])
                run = ssh(ssh_client, 'screen -ls;')
                for cpu in xrange(1, working_cpus_per_host[host]+1):
                    screen_name = utils.make_screen_name(simfolder, host, cpu)
                    if "finished_%s" % screen_name in fin:
                        finished[host].append(cpu)
                        found_jobs += 1
                    if screen_name in run:
                        running[host].append(cpu)
                        found_jobs += 1
            else:
                print "[Nicessa] Cannot make connection to host %d" % host
    print
    if found_jobs == 0:
        print '[Nicessa] Could not find any running or finished jobs for this simulation.\n\
                Maybe I am checking for the wrong set of simulations? In that case, please use the "--simulations" option together with "--check" (or "--results").'
    else:
        print "[Nicessa] Finished cpus:"
        for host in finished.keys():
            print " %.27s:\t%s" % (remote_conf.get("host%i" % host, "name").ljust(24), str(finished[host]))
        print "[Nicessa] Still running cpus:"
        for host in running.keys():
            print " %.27s:\t%s" % (remote_conf.get("host%i" % host, "name").ljust(24), str(running[host]))
    return True


def check_data(simfolder, host):
    '''
    Performs a check on existing data, ask for confirmation to overwrite it.

    :param string simfolder: relative path to simfolder
    :param string host: host nr
    :returns: True if no data found or if data can be overwritten, False otherwise
    '''
    conf = utils.get_main_conf(simfolder)
    hosts = utils.num_hosts(simfolder)
    remote_conf = utils.get_host_conf(simfolder)

    hostname = remote_conf.get("host%i" % host, "name")
    ssh_client = _get_ssh_client(remote_conf, host)
    if ssh_client:
        # first look if simulation directory exists yet
        host_path = remote_conf.get("host%i" % host, "path")
        sim_dir = utils.make_simdir_name(simfolder)
        host_path_dirs = ssh(ssh_client, 'cd %s; ls' % host_path)
        if not sim_dir in host_path_dirs:
            return True
        # if so, look in simulation folder for 'data' dir
        path = '%s/%s' % (host_path, sim_dir)
        dirs = ssh(ssh_client, 'cd %s/%s; ls' % (path, simfolder))
        if 'data' in [d for d in dirs.split('\n') if not d.startswith('[Nicessa]') and not d == '']:
            # check if 'data' dir has (non-hidden) content
            data = ssh(ssh_client, 'cd %s/%s/data; ls' % (path, simfolder))
            if len([f for f in data.split('\n') if not f == '' and not f.startswith('.')\
                                                   and not f.startswith('[Nicessa]')]) > 0:
                print '[Nicessa] On host %s, I found older log data (in %s/data). Remove? [y/N]' % (hostname, path)
                if not raw_input().lower() == 'y':
                    return False
                else:
                    return True
            else:
                return True
        else:
            return True
    else:
        print "[Nicessa] Cannot make connection to host %d" % host
        return False
    return True


def get_results(simfolder, do_wait=True):
    '''
    Copy result logs from the remote host(s) if they are all available for the whole job.

    :param string simfolder: relative path to simfolder
    :param boolean do_wait: True if regular checks should be done until all data is available (default is True)
    '''
    print '*' * 80
    print "[Nicessa] Looking for results ... "

    remote_conf = utils.get_host_conf(simfolder)
    hosts_done = dict.fromkeys(xrange(1, utils.num_hosts(simfolder)+1), False)
    working_cpus_per_host = utils.working_cpus_per_host(simfolder)
    for host in hosts_done.keys():
        working_cpus_per_host[host] = 0
        if os.path.exists('%s/conf/%d' % (simfolder, host)):
            working_cpus_per_host[host] = len(os.listdir('%s/conf/%d' % (simfolder, host)))
    all_done = False
    if remote_conf.has_option('communication', 'wait-for'):
        if do_wait:
            waiting = remote_conf.getint('communication', 'wait-for')
            print "[Nicessa] waiting for %d seconds ... " % waiting
            time.sleep(waiting)
    if remote_conf.has_option('communication', 'check-every'):
        check_interval = remote_conf.getint('communication', 'check-every')
    else:
        check_interval = 10
    first_time_done = False

    if remote_conf.has_option('communication', 'shared-home'):
        if remote_conf.getboolean('communication', 'shared-home'):
            # check only the first who ran CPUs
            for host in hosts_done.keys():
                if working_cpus_per_host[host] > 0:
                    hosts_done = {host: False}
                    break
    while not all_done:
        for host in hosts_done.keys():
            if working_cpus_per_host[host] == 0:
                hosts_done[host] = True
            if not hosts_done[host]:
                hostname = remote_conf.get("host%i" % host, "name")
                if first_time_done:
                    print ".",
                ssh_client = _get_ssh_client(remote_conf, host)
                if ssh_client:
                    try:
                        path = '%s/%s' % (remote_conf.get("host%i" % host, "path"), utils.make_simdir_name(simfolder))
                        # check for status by looking for the marker files this host should generate
                        res = ssh(ssh_client, 'cd %s/%s; ls' % (path, simfolder))
                        # TODO: this is no good when we get the results on a different computer than we started from
                        #relevant_subsims = [subsim for subsim in utils.get_subsimulation_names(conf) if osp.exists("%s/conf/%s/%s" % (simfolder, subsim, host))]
                        if 'data' in res and reduce(lambda x, y: x and y, \
                                             map(res.__contains__, ["finished_%s" % utils.make_screen_name(simfolder, host, cpu) for cpu in xrange(1, working_cpus_per_host[host]+1)])):
                            scp_client = scp.SCPClient(ssh_client._transport)
                            try:
                                print "[Nicessa] contacting %s - compressing ... " % hostname ,
                                sys.stdout.flush()
                                ssh(ssh_client, 'cd %s/%s; tar -cf data_%d.tar data/*; gzip -f data_%d.tar;' % (path, simfolder, host, host))
                                time.sleep(2)
                                print "copying ..." , ; sys.stdout.flush()
                                scp_client.get("%s/%s/data_%d.tar.gz" % (path, simfolder, host), local_path='%s' % simfolder)
                                os.chdir(simfolder)
                                Popen("tar -zxf data_%d.tar.gz; rm data_%d.tar.gz" % (host, host), shell=True).wait()
                                if not simfolder == ".":
                                    for sf in simfolder.split("/"):
                                        if sf != '':
                                            os.chdir('..')
                            except OSError, e:
                                print e
                            hosts_done[host] = True
                            print "done."
                            ssh(ssh_client, 'cd %s/%s; %s' % (path, simfolder, clean_states(simfolder, host)))
                    except Exception, e:
                        print e
                    ssh_client.close()
                else:
                    print "cannot connect to %s " % host
                    pass # keep on trying
                    #hosts_done[host] = True # can't connect, so don't keep on trying
        print "_",
        sys.stdout.flush()
        # all done now?
        all_done = True
        for host_done in hosts_done.values():
            all_done = all_done and host_done
        if not first_time_done and not all_done:
            print "[Nicessa] now checking every %d seconds ... " % check_interval
            first_time_done = True
        if not all_done:
            time.sleep(check_interval)
    print "[Nicessa] Got all results."
    print '*' * 80
    print;


def show_screen(simfolder, host, cpu, lines=50):
    '''
    Show the screen log of the screen running on a specific host and cpu.

    :param string simfolder: relative path to simfolder
    :param int host: index of host
    :param int cpu: index of cpu
    :returns: True if successful, False otherwise
    '''
    screen_name = utils.make_screen_name(simfolder, host, cpu)
    remote_conf = utils.get_host_conf(simfolder)
    host_name = remote_conf.get("host%i" % host, "name")
    ssh_client = _get_ssh_client(remote_conf, host)

    running = ssh(ssh_client, 'screen -ls;')
    if not screen_name in running:
        print '[Nicessa] No screen for cpu %i on %s is running at the moment. Show the screenlog? [Y|n]' % (cpu, host_name)
        if raw_input().lower() == 'n':
            return False

    scp_client = scp.SCPClient(ssh_client._transport)
    path = '%s/%s' % (remote_conf.get("host%i" % host, "path"), utils.make_simdir_name(simfolder))
    print "[Nicessa] getting screen log from %s ... " % host_name
    try:
        scp_client.get("%s/%s/screenlogs/%s.log" % (path, simfolder, screen_name), local_path='%s' % simfolder)
    except scp.SCPException, e:
        print e
    local_file_name = '%s/%s.log' % (simfolder, screen_name)
    if os.path.exists(local_file_name):
        f = open(local_file_name, 'r')
        all_lines = f.readlines()
        f.close()
        os.remove('%s/%s.log' % (simfolder, screen_name))
        print '************* Begin Screen Content ********************'
        for line in all_lines[-1 * lines:]:
            print line,
        print '************* End Screen Content **********************'
    else:
        print '[Nicessa] Couldn\'t download the screen log for cpu %i on %s.' % (cpu, host_name)
    return True


def kill_screens(simfolder):
    '''
    Kill all screens that currently run the main simulation or the specified set of simulations.

    :param string simfolder: relative path to simfolder
    '''
    conf = utils.get_main_conf(simfolder)
    hosts = utils.num_hosts(simfolder)
    remote_conf = utils.get_host_conf(simfolder)
    working_cpus_per_host = utils.working_cpus_per_host(simfolder)
    sims = ','.join(utils.get_subsimulation_names(conf))

    print '[Nicessa] Kill all screens running simulations (%s)? [y/N]' % sims
    if not raw_input().lower() == 'y':
        print '[Nicessa] I did nothing.'
        return

    print "[Nicessa] Killing screens on hosts: "
    sys.stdout.flush()
    for host in xrange(1, hosts+1):
        if working_cpus_per_host[host] > 0:
            hostname = remote_conf.get("host%i" % host, "name")
            path = '%s/%s' % (remote_conf.get("host%i" % host, "path"), utils.make_simdir_name(simfolder))
            print "%s (host-nr:%d, cpus:%d)  " % (hostname, host, working_cpus_per_host[host]),
            sys.stdout.flush()
            ssh_client = _get_ssh_client(remote_conf, host)
            if ssh_client:
                killed = ssh(ssh_client, "ps -ef | grep '%s' | awk '{print $2}' | xargs kill -9"\
                                  % utils.make_simdir_name(simfolder),
                                  ignore=['usage: kill', 'kill ', 'No Sockets found', 'No such process'])
                time.sleep(1)
                ssh(ssh_client, 'screen -wipe')
                ssh(ssh_client, 'cd %s; rm -r *' % path)
                sys.stdout.flush()
    print '[Nicessa] Done.'


def _get_ssh_client(remote_conf, host):
    '''
    Make an SSH client and connect it.
    We first try a passwordless login, and then ask for credentials.

    :param ConfigParser remote_conf: host configuration
    :param int host: index of host
    :returns: paramiko.SSHClient if successful, None otherwise
    '''
    usr = remote_conf.get("host%i" % host, "user")
    hostname = remote_conf.get("host%i" % host, "name")

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh_client.connect(hostname, username=usr)
    except (paramiko.AuthenticationException, paramiko.SSHException):
        pass
        #print "[Nicessa] Could not connect to host '%s' as user '%s' with no password." % (hostname, usr)
        #print "          If you want password-less logon, please check your RSA key or shared/remembered connection setup."
    except Exception, e:
        print "[Nicessa] WARNING: Error while connecting with host %s: %s. " % (hostname, e)
        if "Unknown server" in str(e):
            print "          Is it a known host (look in ~/.ssh/known_hosts)?"
        return None
    else:
        return ssh_client
    ssh_client = None
    while ssh_client is None:
        print "Logging in user '%s' on host '%s' now (type 'exit' to abort): " % (usr, hostname)
        passwd = getpass()
        if passwd == 'exit':
            break
        ssh_client = paramiko.SSHClient()
        ssh_client.load_system_host_keys()
        try:
            ssh_client.connect(hostname, username=usr, password=passwd)
        except paramiko.AuthenticationException:
            print "[Nicessa] Authentication was not successful."
            ssh_client = None
        except Exception, e:
            print "[Nicessa] WARNING: Error while connecting with host %s: %s" % (hostname, e)
            ssh_client = None
    return ssh_client


def clean_states(simfolder, host):
    '''
    Build commands which remove traces of any (former) Nicessa activity on one host:
    clean running screens and files that are used to indicate states.
    For files, we assume to be located in the data dir Nicessa uses in that host.

    :param string simfolder: relative path to simfolder
    :param int host: index of host
    :returns: a string with all ``rm`` and ``kill`` commands
    '''
    clean = ""
    clean += "rm data_%d.tar.gz;" % host
    clean += "rm cmd_%d;" % host
    cpus = utils.cpus_per_host(simfolder)[host]
    # clean up marker files
    for cpu in xrange(1, cpus+1):
        clean += 'rm finished_%s;' % utils.make_screen_name(simfolder, host, cpu)
    # kill old screens by name
    pattern = '|'.join([utils.make_screen_name(simfolder, host, cpu) for cpu in xrange(1, cpus+1)])
    clean += "kill `ps aux | awk '/%s/{print $2}'`;" % pattern
    return clean

