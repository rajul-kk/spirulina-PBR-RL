import numpy as np, sys
from fitgrowth import load
A={700:[],1400:[]}
for pat in ['lowA','lowB']:
    I=load(50,pat)
    for b in sorted(set(i[0] for i in I)):
        rows=[i for i in I if i[0]==b and i[1]<96]
        print(b, ' '.join('%d:L%d:x%d:mu%.4f:T%.1f'%(i[1],i[4].mean(),i[2],np.log(i[3]/i[2])/12,i[5].mean()) for i in rows))
        for i in rows:
            if i[1]>=12: A[int(round(i[4].mean(),-2))].append(np.log(i[3]/i[2])/12)
for k,v in A.items(): print(k,len(v),'%.4f +- %.4f'%(np.mean(v),np.std(v)/np.sqrt(len(v))))
