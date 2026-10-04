from simfit import *
import sys
pat=sys.argv[1]
tot=[]
for fn in sorted(glob.glob('../trials/'+pat)):
    d=load(fn); sp=d['Ca_sp']; e=((d['Ca']-sp)/0.01)**2-0.04
    ch=[i for i in range(1,len(sp)) if sp[i]!=sp[i-1]]
    win=np.zeros(len(sp),bool); win[:15]=True
    for c in ch: win[c:c+15]=True
    trans=e[win].sum()/120; rest=e[~win].sum()/120
    print(fn[-22:],'meas-cost %.3f  transient(15 after start/sp change) %.3f  steady %.3f'%(e.mean(),trans,rest),
          'steps',[(c,round((sp[c]-sp[c-1])*100,2)) for c in ch], 'Tc range %.1f-%.1f'%(d['Tc'].min(),d['Tc'].max()))
