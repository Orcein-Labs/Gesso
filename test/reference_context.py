import os,sys,json,hashlib
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,transformers
from transformers import AutoModelForCausalLM,LlamaConfig
from transformers.modeling_rope_utils import _compute_linear_scaling_rope_parameters,_compute_llama3_parameters
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb
torch.set_num_threads(2)
root=sys.argv[1]; output=sys.argv[2]
model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
base=[1,2,3,4,5,6,7,8,9,10,11,12,13]
cases=[]
with torch.inference_mode():
 for n in (17,33,65,129):
  ids=(base*((n+12)//13))[:n]
  logits=model(torch.tensor([ids])).logits[0,-1]
  cases.append({'ids':ids,'last_logits':logits.tolist(),'next_id':int(logits.argmax())})
 positions=torch.tensor([[0,1,8191,8192,32768]])
 q=torch.arange(1,5*8+1,dtype=torch.float32).reshape(1,1,5,8)/40
 ropes=[]
 for kind in ('default','linear','llama3'):
  params={'rope_type':kind,'rope_theta':100000.}
  if kind!='default':params.update(factor=8.)
  if kind=='llama3':params.update(low_freq_factor=1.,high_freq_factor=4.,original_max_position_embeddings=8192)
  cfg=LlamaConfig(hidden_size=8,num_attention_heads=1,max_position_embeddings=32768,rope_parameters=params)
  rope=LlamaRotaryEmbedding(cfg);cos,sin=rope(q,positions)
  got,_=apply_rotary_pos_emb(q,q,cos,sin)
  ropes.append({'kind':kind,'positions':positions[0].tolist(),'input':q[0,0].tolist(),'inv_freq':rope.inv_freq.tolist(),'output':got[0,0].tolist()})
receipt={'schema':'gesso-independent-context-v1','torch':torch.__version__,'transformers':transformers.__version__,'dtype':'float32','device':'cpu','checkpoint_sha256':hashlib.sha256(open(os.path.join(root,'config.json'),'rb').read()).hexdigest(),'cases':cases,'ropes':ropes}
open(output,'w').write(json.dumps(receipt,allow_nan=False)+'\n')
print('External HF context and rotary reference written')
