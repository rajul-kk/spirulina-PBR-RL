from simfit import *
for lag in [0,1]:
  E=[]
  for fn in sorted(glob.glob('../trials/b00[2-5]*.csv')):
    d=load(fn)
    for i in range(1,len(d['Tc'])-1):
        xp=step(np.array([d['Ca'][i],d['T'][i]]),d['Tc'][i-lag],P)
        E.append([d['Ca'][i+1]-xp[0],d['T'][i+1]-xp[1]])
  E=np.array(E); print('lag',lag,'rms',np.sqrt((E**2).mean(0)))
