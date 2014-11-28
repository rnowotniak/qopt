#!/usr/bin/python

import sys


from pylab import *

# DIM 50
XI = (
        0.465938, 
        0.416822, 
        0.415459, 
        0.497842, 
        0.311992, 
        )

DELTA = (
        0.986928,
        0.946898,
        0.934386,
        0.986275,
        0.898564,
        )

XI_worst = (
0.579114  ,
0.603145  ,
0.917436  ,
0.264703  ,
0.865176  ,
0.194553  ,
0.256631  ,
0.969614  ,
0.163682  ,
0.725037  ,
#0.736514  ,
0.887865  ,
0.817266  ,
0.60429   ,
0.69916   ,
)

DELTA_worst = (
   0.302611     ,
   0.293993     ,
   0.366648     ,
   0.0262873    ,
   0.303628     ,
   0.0358061    ,
   0.0273605    ,
   0.33598      ,
   0.014152     ,
   0.0734054    ,
   #0.998762     ,
   0.0528959    ,
   0.040472     ,
  0.0267464     ,
  0.020109      ,
)











# # DIM 30
# XI = (0.364177,
#         0.416822,
#         0.560395,
#         0.465938,
#         0.497842)
# 
# DELTA = (0.982871,
#         0.946898,
#         0.962158,
#         0.986928,
#         0.986275 )
# 
# 
# XI_worst = (
# 0.499078   ,
# 0.361049   ,
# #0.00393518 ,
# 0.969614   ,
# 0.200977   ,
# 0.194553   ,
# 0.163682   ,
# #0.97304    ,
# 0.873573   ,
# 0.814767   ,
# 0.887865   ,
# 0.60429    ,
# 0.977294   ,
# #0.736514   ,
# 0.729658   ,
# )
# 
# DELTA_worst = (
#   0.0947679     ,
#   0.0344567     ,
#     #0.938463    ,
#   0.33598       ,
#   0.0183306     ,
#   0.0358061     ,
#   0.014152      ,
#  #0.996816       ,
#   0.190194      ,
#   0.0681938     ,
#   0.0528959     ,
#  0.0267464      ,
#   0.0542578     ,
#   #0.998762      ,
#   0.0101053     ,
# )




# DIM 10
#    XI = (
#            0.638822,
#            0.53769, 
#            0.650384,
#            0.62614, 
#            0.508857,
#            0.489323,
#            )
#    
#    DELTA = (
#            0.898101  ,
#            0.967892  ,
#            0.95879   ,
#            0.975676  ,
#            0.880632  ,
#            0.913706  ,
#            )
#    
#    XI_worst = (
#    0.0191608  ,
#    0.105546   ,
#    0.0670178  ,
#    0.0934853  ,
#    0.163682   ,
#    0.0126948  ,
#    0.0727609  ,
#    0.719378   ,
#    0.046576   ,
#    0.0703358  ,
#    0.00393518 ,
#    0.00898533 ,
#    0.00749342 ,
#    )
#    
#    DELTA_worst = (
#       0.874844      ,
#      0.400585       ,
#       0.352891      ,
#       0.111135      ,
#      0.014152       ,
#       0.957948      ,
#       0.287763      ,
#      0.00203776     ,
#      0.236395       ,
#       0.0789305     ,
#        0.938463     ,
#        0.799368     ,
#        0.570414     ,
#    )




# DIM 2  -- bez zadnych zabiegow
# XI = (
#         0.734905,
#         0.924851,
#         0.882488,
#         0.745135,
#         #0.711523,
#         #0.810707,
#         #0.977657,
#         )
# 
# DELTA = (
#         0.698213 ,
#         0.544629 ,
#         0.643563 ,
#         0.721118 ,
#         #0.717335 ,
#         #0.478629 ,
#         #0.420812
#         )





#    # DIM2  --   ponizsze sa sposrod punktow, ktore byly probkowane dla wszystkich wymiarowosci
#    XI = (
#    0.741737 , 
#    0.955577 , 
#    0.792768 , 
#    0.936261 , 
#    #  0.910939 , 
#    #  0.93194  , 
#    #  0.732811 , 
#            )
#    
#    DELTA = (
#    
#       0.75595       , 
#       0.789475      , 
#       0.785005      , 
#       0.917006      , 
#    #   0.898484      ,       
#    #  0.76188        , 
#    #   0.773718      , 
#    )
#    
#    
#    XI_worst = (
#    0.00644673  ,
#    0.0346603   ,
#    0.00393518  ,
#    0.028626    ,
#    0.0025388   ,
#    0.00917114  ,
#    0.00280517  ,
#    0.00198027  ,
#    0.00187413  ,
#    )
#    
#    DELTA_worst = (
#    0.905895    ,
#    0.00438882   ,
#    0.938463    ,
#    0.00668218    ,
#    0.679524     ,
#    0.0541838   ,
#    0.230395    ,
#    0.0992839   ,
#    0.0976963   ,
#    )







xlim(0,1)
ylim(0,1)
xlabel('$\\xi$')
ylabel('$\\delta$')


# X = randn(733) * .20 + .8
# Y = randn(733) * .15 + .9
# plot(X,Y, 'gx', markersize=5, label='$N=2$')
# 
# X = randn(154) * .15 + .6
# Y = randn(154) * .10 + .9
# plot(X,Y, 'rs', markersize=5, label='$N=10$')
# 
# X = randn(106) * .10 + .45
# Y = randn(106) * .10 + .95
# plot(X,Y, 'bo', markersize=6, label='$N=30$')
# 
# X = randn(77) * .08 + .4
# Y = randn(77) * .10 + .92
# plot(X,Y, '^', color='#FFFF00', markersize=8, label='$N=50$')

f=open('results10.txt')

import re

XI=[]
DELTA=[]

mf={}

while True:
    line=f.readline()
    if not line:
        break
    xi,delta,val = re.sub(r'.*dim\d+-(.*)-(.*)\.npy +\d+ (.*)', r'\1,\2,\3', line).split(',')
    XI.append(xi)
    DELTA.append(delta)
    mf[(xi,delta)] = float(val)
    print xi,delta,val,

gca().set_aspect('equal')

import random


t=0
xi,delta=random.choice(mf.keys())
best=(xi,delta)
bestval=mf[best]
print type(bestval)
print best, bestval
while t<1000000:
    xi,delta=random.choice(mf.keys())
    if mf[(xi,delta)] < bestval:
        print best, bestval
        bestval=mf[(xi,delta)]
        gca().annotate('', xytext=best, xy=(xi,delta), arrowprops={'arrowstyle':'->','connectionstyle':'arc3','linewidth':2})
        print 'aaaaa'
        best=(xi,delta)
    t += 1



legend(loc=3, numpoints=1)

plot(XI,DELTA, 'go', markersize=3)
#plot(XI_worst,DELTA_worst, 'dr', markersize=10)

savefig('/tmp/plot2d.pdf', bbox_inches='tight')

