#!/usr/bin/python

import sys
import numpy
import matplotlib.pyplot as plt

qiga= numpy.matrix(';\n'.join([s.strip().replace(' ',' ',1) for s in open('original-qiga-results.2').readlines()]))
tuned= numpy.matrix(';\n'.join([s.strip().replace(' ',' ',1) for s in open('tuned-qiga-results.2').readlines()]))
tuned2= numpy.matrix(';\n'.join([s.strip().replace(' ',' ',1) for s in open('tuned-mf2.txt').readlines()]))
sga= numpy.matrix(';\n'.join([s.strip().replace(' ',' ',1) for s in open('sga-plot.txt').readlines()]))

plt.plot(tuned2[:,0], tuned2[:,1], 'g^-', label='Tuned QIGA 2', markevery=50, markersize=12)
plt.plot(tuned[:,0], tuned[:,1], 'ro-', label='Tuned QIGA 1', markevery=50, markersize=10)
plt.plot(qiga[:,0], qiga[:,1], 's-', label='Original QIGA', markevery=50, markersize=10)
plt.plot(sga[:,0], sga[:,1], 'x-', label='SGA', markevery=5, markersize=10)

plt.xlabel('Evaluation count')
plt.ylabel('Fitness')
plt.grid(True)
plt.legend(loc='upper left')

# plt.annotate('Meta-fitness values\n of the algorithms',
#         xy=(4999, 1407.62), xytext=(2500,1340), textcoords='data',
#         bbox=dict(boxstyle="round", fc="0.8"),
#         size='x-large',
#         arrowprops=dict(arrowstyle="-|>", lw=2, ls='solid', lod=False, color='black'))
# plt.annotate('Meta-fitness values\n of the algorithms',
#         xy=(4999, 1457.71), xytext=(2500,1340), textcoords='data',
#         bbox=dict(boxstyle="round", fc="0.8"),
#         size='x-large',
#         arrowprops=dict(arrowstyle="-|>", lw=2, ls='solid', lod=False, color='black'))
# plt.annotate('Meta-fitness values\n of the algorithms',
#         xy=(4999, 1378.19), xytext=(2500,1340), textcoords='data',
#         bbox=dict(boxstyle="round", fc="0.8"),
#         size='x-large',
#         arrowprops=dict(arrowstyle="-|>", lw=2, ls='solid', lod=False, color='black'))

#        arrowprops=dict(arrowstyle="simple"))
#         arrowprops=dict(arrowstyle="simple",fc="black", ec="none",connectionstyle="arc3,rad=0")

        
#plt.annotate('Meta-fitness values', (2000, 1378.19))

plt.savefig('/tmp/cmp.pdf', bbox_inches='tight')


