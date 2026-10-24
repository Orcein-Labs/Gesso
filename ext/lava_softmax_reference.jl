# Fixed per-row tree keeps checked Float32 softmax independent of query partition.
@kernel cpu=false function _lava_softmax_rowsum64_kernel!(den, @Const(E), K::Int)
    row=@index(Group, Linear)
    lane=@index(Local, Linear)
    partial=@localmem Float32 (64,)
    value=Float32(0)
    for j = lane:64:K
        value+=E[row, j]
    end
    partial[lane]=value
    @synchronize
    step=32
    while step>=1
        if lane<=step
            partial[lane]+=partial[lane+step]
        end
        @synchronize
        step=step÷2
    end
    if lane==1
        den[row, 1]=partial[1]
    end
end
function _lava_softmax_rowsum64(E)
    eltype(E)===Float32&&ndims(E)==2||error("row sum type")
    L, K=size(E)
    den=Lava.LavaArray{Float32}(undef, L, 1)
    L==0&&return den
    kernel=Lava.with_no_contraction(
        Lava.with_float_controls(
            _lava_softmax_rowsum64_kernel!(Lava.KernelAbstractions.get_backend(E)),
            (32,),
        ),
    )
    kernel(den, E, Int(K); ndrange = 64L, workgroupsize = 64)
    den
end
