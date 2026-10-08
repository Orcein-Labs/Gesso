import json, sys
import torch
from torch.nn import functional as F
torch.set_num_threads(2)
rows=[[0,0,0,0],[1e-9,-1e-9,2e-9,-2e-9],[1,-2,3,-4],[1e10,-1e10,1e10,-1e10]]
scale=[1,.5,-1,2]
w=[[1,2,3,4],[-2,1,0,-1]]
scores=[[1e6,1e6-1,1e6-2],[1e6-2,1e6,1e6-1],[1e6-1,1e6-2,1e6]]
result={'schema':'gesso-independent-numerics-v1','torch':torch.__version__,'rows':rows,'scale':scale,'weights':w,'scores':scores,'eps':1e-6,'cases':{}}
for dt in (torch.float64,torch.float32):
 x=torch.tensor(rows,dtype=dt); sc=torch.tensor(scale,dtype=dt); weights=torch.tensor(w,dtype=dt)
 norm=F.rms_norm(x,(4,),weight=sc,eps=1e-6)
 sm=torch.tensor(scores,dtype=dt).masked_fill(torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1),-torch.inf)
 result['cases'][str(dt)]={'rmsnorm':norm.tolist(),'swiglu':(F.silu(x)*sc).tolist(),'matmul':(x@weights.T).tolist(),'softmax':torch.softmax(sm,dim=-1).tolist()}
open(sys.argv[1],'w').write(json.dumps(result,allow_nan=False)+'\n')
print('Independent native PyTorch float64/float32 operator reference written')
