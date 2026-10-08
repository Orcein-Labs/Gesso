import time
start=time.perf_counter()
import torch,json,sys,statistics,transformers
from transformers import AutoModelForCausalLM
initialization=time.perf_counter()-start
ref=json.load(open(sys.argv[1])); checkpoint=sys.argv[2]
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(2)
t=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,dtype=torch.float32,attn_implementation='eager').to('cuda').eval();torch.cuda.synchronize();load=time.perf_counter()-t
records=[]
with torch.inference_mode():
    for c in ref['cases']:
        ids=torch.tensor([c['prompt_ids']],device='cuda')
        def run():
            result=model.generate(ids,max_new_tokens=8,do_sample=False,use_cache=True,pad_token_id=0,eos_token_id=0)
            torch.cuda.synchronize()
            return result[0].tolist()
        t=time.perf_counter();result=run();cold=time.perf_counter()-t
        assert result==c['generated_ids']
        for _ in range(2):assert run()==c['generated_ids']
        times=[]
        for _ in range(3):
            torch.cuda.synchronize();t=time.perf_counter();result=run();times.append(time.perf_counter()-t)
            assert result==c['generated_ids']
        assert max(times)/min(times)<2
        records.append(dict(prompt=c['prompt'],first_use_seconds=cold,samples_seconds=times,median_seconds=statistics.median(times),end_to_end_new_tokens_per_second=8/statistics.median(times)))
json.dump(dict(schema='gesso-independent-performance-v1',backend='PyTorch CUDA eager',torch=torch.__version__,transformers=transformers.__version__,dtype='float32',tf32=False,initialization_seconds=initialization,load_seconds=load,records=records,device=torch.cuda.get_device_name(),device_total_memory=torch.cuda.get_device_properties(0).total_memory,peak_allocated_bytes=torch.cuda.max_memory_allocated(),compile_excluded=True),open(sys.argv[3],'w'),indent=2)
