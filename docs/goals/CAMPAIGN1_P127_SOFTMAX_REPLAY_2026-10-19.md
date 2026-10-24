# Campaign 1: matched-input head1 softmax replay

## Objective
Locate the first whole127/partial63softmax stage difference for captured block4/query116/head1,without changing arithmetic or source.

## Permitted files
New external versioned helper/read-only softmax clone/capture/auditor/mutation and planning records only. Frozen6d3683c,all910trackedfiles,consumed helpers/environments/captures,CPUoracle/targets/gates/qualifiedLava unchanged. No libs,source,vendor,dependency,precision/operator/compiler/link edits.

## Interfaces
Captured completeQ/Kcoreinputs from qualifiedattentiontrace reconstruct completehead1scores with ordinary _attention_scores_device!. Same public PrefillWorkload softmax and same checked _lava_softmax! body under a new private diagnostic name,adding readbacks only at masked,rowmax,exp,denominator,output. Compare against ordinary public softmax controls on identical GPUscore bytes. Separate ordinaryCPUsoftmax comparison is operator-reference evidence only,never CPU fallback or model activation injection.

## Invariants
Two actual layouts:L127/K127/queryrow116andL63/K127/queryrow52(global116). Reconstructed score row must match archivedmodelscore;observed/publiccontrol fulloutputs and archivedmodelprobabilityrow must match exactly. Three repetitions each;source/weight/precision unchanged. RawfullFloat32scores,masked(-Infonlyinmaskedfuture),rowmax,e,den,out;complete axis mapping.64MiBartifact cap. Archive actualcontextcachedSPV/LLVM/WGinventory with explicit limitation:cachemembership is not a per-dispatch launch ledger. No candidate normalization/reduction arithmetic in this bounded baseline replay;scope a separate item after stage evidence.

## Tests
Prerequisite attentiontrace independentlyqualified:allouterinputs and query116scores match;first observed difference head1probabilities,116values,maxabs5.960464477539063e-8,up4ULP. Both ordinarymodelcontrols exact,910source files unchanged. Newreplay independently checks inverseclone/source/helper/environment/inputdigests,causalmask/max/finiteness/rawrowlayout,repeat/public/modelcontrol matches,CPUcomparison and stage differences;effective repaired-digest numericmutation required. Do not infer denominator vs division vs exponent cause before intermediate evidence. Keep all originalfullstress gates red/open until source repair passes them.

## Performance target
No throughput claim;readbacks and compilation included.64MiBartifacts,20GiBRAM/4GiBswap,4800MiBGPUbudget,1200seconds,16threads,BLAS/OMP1,sharedlocks. No modelweights loaded for this operator replay.

## Expected artifact
Qualified first differing checked-softmax intermediate with matched score inputs,or explicit observation/reproductionfailure. No source/model/fullstress/production admission,push,merge,deployment.

## Receipt
OPEN.Newattentiontrace passes independentraw/source/inverse/controlaudit:36outerstages,96headstages,159rawarrays,60698504artifactbytes,910unchanged sourcefiles. Firstobserved difference ishead1probabilitiesquery116;exactprimitive cause stillunknown. Baselinereplayunwritten/unrun.

### Terminal qualification
Runner terminal0; p127-softmax-stage-replay-validation-1.json independently qualifies all6 observed/control/model probability results. Masked scores, row maxima and exponentials match across layouts. First difference is denominator: whole3.8910722732543945(0x40790754), suffix3.8910715579986572(0x40790751),3ULP. Independent exact sum of capturedFloat32terms rounds to0x40790752; fixed64Float32tree matches the passing suffix0x40790751. This is not a CPU bit-exactness claim. All910source files unchanged,2,085,178artifact bytes. Copied-output numeric mutation remains pending until its receipt is terminal. The fixed-tree candidate belongs to its separate work item.

The baseline copied-output mutation is now terminal: p127-softmax-stage-replay-mutation-validation-1.json passes. Joint observed/control/staged probability tampering with repaired hashes and retained green flags is rejected; original artifacts unchanged.
