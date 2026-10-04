from simfit import *
for fn in sorted(glob.glob('../trials/b00*.csv')):
    d=load(fn)
    for k in ['Ca','T']:
        y=d[k]; s=y[2:]-2*y[1:-1]+y[:-2]
        print(fn[-20:],k,'sigma est (2nd diff/sqrt6)',round(np.std(s)/np.sqrt(6),4),end='  ')
    print()
