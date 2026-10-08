import json,sys
ref=json.load(open(sys.argv[1]))
def digest(ids):
    h=0xcbf29ce484222325
    for value in ids:
        for byte in int(value).to_bytes(8,'little',signed=False):
            h=((h^byte)*0x100000001b3)&0xffffffffffffffff
    return format(h,'016x')
json.dump({'schema':'gesso-output-digest-reference-v1','cases':[digest(c['generated_ids']) for c in ref['cases']]},open(sys.argv[2],'w'),indent=2)
