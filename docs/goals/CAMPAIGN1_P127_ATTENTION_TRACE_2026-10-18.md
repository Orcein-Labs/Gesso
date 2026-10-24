# Campaign 1: block4/token116 attention localization

## Objective
Identify the first differing attention operation for the raw-qualified whole127/chunk64model discrepancy at block4/token116.

## Permitted files
New external versioned helpers/generated read-only generator/head-function clones/raw captures/auditors/mutation controls and planning records only. Frozen source6d3683c,all910tracked files,consumed environments/helpers/captures,qualifiedLava,CPUoracle/targets/gates unchanged. No libs,source,vendor,dependency,precision/operator/compiler/link edits.

## Interfaces
New private read-only clones of current reference_generate and _attention_heads!,with exact inverse-source proof. Generator delegates its existing head-contraction call to the renamed identical head function,which adds readbacks only. Same publicchunk128/64,freshGenerateState258,p127,model/backend/epsilon/maxnew1/eos_ids(),and four observed/ordinary control routes. No production method replacement,activation injection or arithmetic change.

## Invariants
Capture block4only: normalized input;Q/K/Vprojections;split heads;Q/Knorm;RoPE;KVviews;actual repeatedQ/K/Vcore inputs;attention result;merged heads;optionalgateprojection;gatedmerge;outputprojection. Complete rawFloat32arrays bounded128MiB. Mark current-query row ranges separately from complete cached-key prefixes. In the cloned head loop,capture only knownbad globalquery116scores/probabilities/value outputs across all16heads;do not claim other per-headrows were observed. Allouterqueryrows andkeys remain available. Ordinarycontrols must reproduce668whole/1503chunked and match rawlogits/hidden/fullIDs/reportedlive-state digests exactly. All910source hashes before/after. No source fix from localization alone.

## Tests
Prerequisite qualified p127modeltrace:600stagerows,60960reportedrowdigests,72rereadfullfields;allrowslayers1–3exact;firstdifferenceblock4mixer,token116only,maxabs3.6135315895080566e-6. Ordinarycontrols reproducebothIDsandallreportedlive-state hashes;repaired-digest outputmutation rejected. Newtrace must independently validate both inverseclones,stage/row/headcoverage,rawSHA/types/finiteness/argmax,KVprefix consistency,observercontrols and source/environment identities. Effective rawnumericmutation required. Follow with matched-input operator replay before any new source candidate.

## Performance target
No throughput claim;readback perturbs timing.128MiBartifact cap,20GiBRAM/4GiBswap,4800MiBGPU,2400seconds,16threads,BLAS/OMP1,sharedlocks.

## Expected artifact
Raw-qualified first differing attention operation and matchedinputs,or explicit observer-control failure. No source/model/fullstress/production admission,push,merge or deployment.

## Receipt
OPEN.Preceding full-model trace independently qualifies:sourcecounterexamplereproduced,allordinarycontrolsendpointsexact,910sourcefilesunchanged. Focused attention traceunwritten/unrun.

### Focused capture running
New external helperp127_attention_trace_v1.jl passes BaseMeta.parseall error-node check. It adds only readonly outer captures and query116per-headscore/probability/value captures through two newly named private clones;each clone verifies exact inverse to source before execution. Shared-lockrunp127-attention-trace-1.log is active(session45712),ordinarycontrols still pending. Actual observed shapes confirmquery16heads×256,key2heads×256,gate4096. No production/source/precision change. Independent auditor prepared;no qualification claimed beforeterminal/control/raw audit.

### Terminal qualification
The focused trace is terminal and independently qualified by p127-attention-trace-validation-1.json. All outer inputs through actual attention Q/K/V and query116scores match; first observed difference is head1probabilities (up to4ULP), with13heads showing probability differences. Observed and ordinary controls match exactly. Source910files unchanged. p127-attention-trace-mutation-validation-1.json rejects a repaired-digest numeric mutation. This localizes the captured softmax difference, not a complete model fix.
