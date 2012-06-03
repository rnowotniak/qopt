#!/usr/bin/python

'''
setup
=====

This module sets up the stage - with it, you can make a configuration file
for each mix of parameter settings that needs to be run.

It also distributes them among the available hosts and cpus:
Each of them gets a directory, in which a main.conf file lists the other configs.
'''

import sys
import os
from ConfigParser import ConfigParser
from shutil import rmtree

import utils



def create(conf, simfolder, limit_to={}, more=False):
    """
    Writes a conf file for each run that the parameters in conf suggest.
    Also divides all runs of an simulation in groups with a main conf each, to run on different hosts.

    :param ConfigParser conf: main configuration
    :param string simfolder: relative path to simfolder
    :param dict limit_to: dict of configuration settings we should limit to (default is an empty dict)
    :param boolean more: True if runs should be appended to existing data (default is False)
    """

    # ---------------------------------------------------------------------------------------------
    # define some helpful functions

    def combiner(lists):
        '''
        :param list lists: list of lists with inputs
        :returns: all possible combinations of the input items in a nested list
        '''
        res = [[]]
        for l in lists:
            tmp = []
            for i in l:
                for j in res:
                    tmp.append(j+[i])
            res = tmp
        return res


    def get_options_values(sim_params, limit_to):
        '''
        :param dict sim_params: dict of configuration for one simulation
        :param dict limit_to: dict of configuration settings we should limit to
        :returns: a list of option names and lists of values to use in every possible combination in one simulation.
        '''
        comb_options = [] # all my param names
        comb_values = [] # all my param values

        for k, v in sim_params.iteritems():
            comb_options.append(k)
            comb_values.append(v)
        comb_values = combiner(comb_values)

        # if we filter the todo-list, remove some configs
        num = len(comb_values)
        for i in xrange(num-1, -1, -1):
            tmpcv = comb_values[i]
            matches = True
            for k, vals in limit_to.iteritems():
                if not str(tmpcv[comb_options.index(k)]) in vals:
                    matches = False
            if not matches:
                comb_values.pop(i)

        return comb_options, comb_values

    def incrTo(val, highest):
        val += 1
        if val == highest+1:
            val = 1
        return val

    def decrTo(val, highest):
        val -= 1
        if val == 0:
            val = highest
        return val

    def make_dir(srv, cpu):
        '''make a new dir in the conf directory (if it exists, delete it)'''
        pdir = os.path.join(simfolder, 'conf', str(srv), str(cpu))
        if os.path.exists(pdir):
            rmtree(pdir)
        os.makedirs(pdir)
        return pdir

    def writeopt((opt, isint, sec), tosub=True):
        ''' helper function to copy values in conf file from meta and control section '''
        right_conf = sim_conf.has_option(sec, opt) and sim_conf or conf
        getter = isint and getattr(right_conf, 'getint') or getattr(right_conf, 'get')
        target_conf = tosub and sub_conf or main_conf
        target_conf.write('%s:%s\n' % (opt, getter(sec, opt)))


    # ---------------------------------------------------------------------------------------------
    # get parameters from subsimulations: combine them into our normal params
    default_params = {}
    for param in conf.options('params'):
        default_params[param] = [v.strip() for v in conf.get('params', param).split(',')]
    simulations = {'': default_params}
    if 'simulations' in conf.sections() and conf.get('simulations', 'configs') != '':
        simulations = {}
        for sim in conf.get('simulations', 'configs').split(','):
            sim = sim.strip()
            simulations[sim] = default_params.copy()
            sim_conf = ConfigParser()
            sim_conf_name = "%s/%s.conf" % (simfolder, sim)
            if not os.path.exists(sim_conf_name):
                print "[Nicessa] Error: Can't find %s !" % sim_conf_name
                sys.exit()
            sim_conf.read(sim_conf_name)
            if sim_conf.has_section('params'):
                for param in sim_conf.options('params'):
                    simulations[sim][param] = [v.strip() for v in sim_conf.get('params', param).split(',')]

    # ---------------------------------------------------------------------------------------------
    # find out how many confs each host should do for each simulation
    hosts = utils.num_hosts(simfolder)
    cpus_per_host = utils.cpus_per_host(simfolder)
    # this is the host we currently use, we don't reset this between simulations to get a fair distribution
    host = 0
    num_per_hosts = range(len(simulations.keys()))
    for n in range(len(num_per_hosts)):
        num_per_hosts[n] = dict.fromkeys(range(1, hosts+1), 0)
    unfulfilled = dict.fromkeys(range(1, hosts+1), 0)

    # these hold all parameter names and the different value configurations, per simulation
    comb_options = {} # param names
    comb_values = {} # param values

    for sim_name in simulations.keys():
        nums = num_per_hosts[simulations.keys().index(sim_name)]

        comb_options[sim_name], comb_values[sim_name] = get_options_values(simulations[sim_name], limit_to)
        confs = len(comb_values[sim_name])
        host = incrTo(host, hosts) # move to next host from where we left off
        if sum(cpus_per_host.values()) == 0:
            print "[Nicessa]: You have not configured any CPUs for me to use. Stopping Configuration ..."
            return
        while 1==1:
            available = min(int(cpus_per_host[host]), confs)
            # if earlier we gave the last host too few, let's even that out right now
            if hosts > 1:
                last_host = decrTo(host, hosts-1)
                if unfulfilled[last_host] > 0:
                    onlast = min(available, min(cpus_per_host[last_host], unfulfilled[last_host]))
                    nums[last_host] += onlast
                    confs -= onlast
                    unfulfilled[last_host] -= onlast
                    available = min(int(cpus_per_host[host]), confs)
                    if confs <= 0: break
            if int(cpus_per_host[host]) > available:
                unfulfilled[host] += int(cpus_per_host[host]) - available
            nums[host] += available
            confs -= available
            if confs <= 0: break
            host = incrTo(host, hosts)

    # ---------------------------------------------------------------------------------------------
    # now write all the conf files

    for host in xrange(1, hosts+1):
        # find out how much each cpu should get (in a helper list, distribute excess one by one)
        load = 0
        for sim in num_per_hosts:
            load += sim[host]
        my_cpus = cpus_per_host[host]
        if my_cpus == 0:
            continue
        cpuloads = dict.fromkeys(xrange(1, my_cpus+1), load / my_cpus)
        excess = load % my_cpus
        if excess > 0:
            cpu = 1
            while excess > 0:
                cpuloads[cpu] += 1
                cpu = incrTo(cpu, my_cpus)
                excess -= 1
        for cpu in xrange(1, my_cpus+1):
            if cpuloads[cpu] > 0:
                # open a main conf
                prefix_dir = make_dir(host, cpu)
                main_conf_name = '%s/main.conf' % prefix_dir
                main_conf = open(main_conf_name, 'w')

                if conf.has_section('seeds'):
                    main_conf.write('\n[seeds]\n')
                    num_seeds = len(conf.items('seeds'))
                    for dat in [(str(opt), isint, 'seeds') for (opt, isint) in zip(range(1, num_seeds+1), [1 for _ in range(num_seeds)])]:
                        writeopt(dat, tosub=False)

            # write as many confs as cpuloads prescribes for this cpu,
            # iterate over simulations while doing so (to even their
            # jobs out over cpus)
            simindex = 0
            jobindex = 0
            for job in range(cpuloads[cpu]):
                if not jobindex % hosts == 0:
                    init_simindex = incrTo(simindex+1, len(simulations.keys()))-1
                else:
                    init_simindex = 0
                simindex = init_simindex
                sim_name = simulations.keys()[simindex]
                while len(comb_values[sim_name]) == 0:
                    simindex = incrTo(simindex+1, len(simulations.keys()))-1
                    if simindex == init_simindex: break
                    sim_name = simulations.keys()[simindex]
                act_comb_values = comb_values[sim_name].pop()
                sub_name = "%s_" % sim_name
                for i in range(len(act_comb_values)):
                    sub_name += "%s%s" % (simulations[sim_name].keys()[i], act_comb_values[i])
                    if i < len(act_comb_values)-1:
                        sub_name += "_"
                sub_conf = open('%s/%s.conf' % (prefix_dir, sub_name), 'w')

                # these sections settings can be overwritten per simulation
                sim_conf = ConfigParser(); sim_conf.read("%s/%s.conf" % (simfolder, sim_name))
                sub_conf.write('[meta]\n')
                for dat in [(opt, isint, 'meta') for (opt, isint) in [('name', 0), ('maintainer', 0)]]:
                    if conf.has_option('meta', opt):
                        writeopt(dat)
                sub_conf.write('[control]\n')
                for dat in [(opt, isint, 'control') for (opt, isint) in [('runs', 1), ('executable', 0)]]:
                    writeopt(dat)
                if more:
                    sub_conf.write('start_run:%d\n' % (utils.runs_in_folder(simfolder, sub_name) + 1))
                sub_conf.write('\n')

                sub_conf.write('[params]\n')
                for i in range(len(act_comb_values)):
                    sub_conf.write('%s:%s\n' % (comb_options[sim_name][i], act_comb_values[i]))

                sub_conf.flush()
                sub_conf.close()

                # mention this sub-conf in the main-conf of this cpu
                main_conf.write('\n')
                main_conf.write('[%s]\n' % sub_name)
                main_conf.write('config_file = conf/%d/%d/%s.conf\n' % (host, cpu, sub_name))
                main_conf.flush()
                jobindex += 1

            if cpuloads[cpu] > 0:
                main_conf.close()

