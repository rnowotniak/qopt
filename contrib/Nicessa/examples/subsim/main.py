#!/usr/bin/python
"""
Example nicessa simulation,
showing advanced usage with a Prisoner Dilemma setup.
This example uses two subsimulations.

See the basic example if you have no idea what this file is for.
"""

from ConfigParser import ConfigParser
import random
import sys


class Agent(object):
    ''' class to represent the Agents competing in the Prisoners Dilemma '''

    def __init__(self, id, world, learning, coop_prob):
        ''' coop: float between 0 and 1, indicating probability to cooperate '''
        self.id = id
        self.learning = learning
        self.coop_prob = coop_prob
        self.payoff = 0
        self.games = 0
        self.world = world
        self.did_cooperate = None

    def act(self):
        self.did_cooperate = random.random() < self.coop_prob
        return self.did_cooperate

    def pay(self, amount):
        ''' this gives payoff to the agent '''
        self.payoff += amount
        self.games += 1
        if self.learning:
            if self.did_cooperate:
                self.coop_prob *= 1 + (.01 * amount)
            else:
                self.coop_prob *= 1 - (.01 * amount)
            self.coop_prob = min(self.coop_prob, 1)


class World(object):
    '''This class organises the agents and lets them play
    '''

    def __init__(self, log, nicessa_conf):
        # the two params passed by nicessa
        self.log = log
        nicessa_conf = nicessa_conf

        # Now I read params from the conf for this run
        self.n = nicessa_conf.getint('params','n')
        self.epochs = nicessa_conf.getint('params','epochs')
        ratio_learning = nicessa_conf.getfloat('params', 'ratio_learning')
        mean_coop = nicessa_conf.getfloat('params', 'mean_coop')

        # get Prisoners Dilemma config
        # t=temptation,r=reward,p=punishment,s=suckers payoff
        self.t = nicessa_conf.getint('params','pd_t')
        self.r = nicessa_conf.getint('params','pd_r')
        self.p = nicessa_conf.getint('params','pd_p')
        self.s = nicessa_conf.getint('params','pd_s')

        # make agents:
        # non-learners' probability to cooperate are roughly fixed,
        # learners start at around 0.5
        self.agents = []
        for i in range(self.n):
            if i < ratio_learning * self.n:
                a = Agent(i, self, True,
                  random.normalvariate(.5, .5/4.))
            else:
                a = Agent(i, self, False,
                  random.normalvariate(mean_coop, max(mean_coop, 1 - mean_coop)/4.))
            self.agents.append(a)

    def pd(self, a, b):
        ''' play a round of the Prisoners Dilemma'''
        c1 = a.act()
        c2 = b.act()

        # return payoffs
        if (c1, c2) == (0, 0): return (self.p, self.p)
        if (c1, c2) == (1, 0): return (self.s, self.t)
        if (c1, c2) == (0, 1): return (self.t, self.s)
        if (c1, c2) == (1, 1): return (self.r, self.r)

    def run(self):
        # log header
        self.log.write('# epoch, avg payoff of non-learners, ')
        self.log.write('avg payoff of learners, ')
        self.log.write('avg cooperation probability of learners\n')

        # for every epoch  starting from 1
        for e in xrange(1, self.epochs + 1):

            self.act_epoch = e

            # let each agent compete (at least) once
            for a in self.agents:
                b = self.agents[random.randint(0, self.n-1)]
                payoffs = self.pd(a, b)
                a.pay(payoffs[0])
                b.pay(payoffs[1])

            # log
            sum_payoff_non_learners = 0.0
            non_learners = [a for a in self.agents if not a.learning]
            for a in non_learners:
                sum_payoff_non_learners += a.payoff/float(a.games)
            sum_payoff_learners = 0.0
            sum_coop_prob_learners = 0.0
            learners = [a for a in self.agents if a.learning]
            for a in learners:
                sum_payoff_learners += a.payoff/float(a.games)
                sum_coop_prob_learners += a.coop_prob
            self.log.write('%d, %f, %f, %f\n' % (\
                            e, sum_payoff_non_learners/max(1., float(len(non_learners))),
                            sum_payoff_learners/float(len(learners)),
                            sum_coop_prob_learners/float(len(learners))
                            )
                    )

        self.log.flush()
        self.log.close()



if __name__ == '__main__':
    conf = ConfigParser()
    conf.read(sys.argv[2])
    w = World(open(sys.argv[1], 'w'), conf)
    w.run()
