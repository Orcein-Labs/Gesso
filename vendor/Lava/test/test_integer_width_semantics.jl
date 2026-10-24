module LavaIntegerWidthTests
using Test, Lava, Random
using Lava.KernelAbstractions: @kernel, @index, @Const
const KA = Lava.KernelAbstractions
const WIDTHS = (2, 3, 7, 8, 9, 11, 15, 16, 17, 23, 31, 32, 33, 53, 63, 64)
const NOPS = 15

# Native argument/result ABI; only internal register values have odd widths.
# Do not attach nsw/nuw: wrapping operations are part of the contract.
@generated function width_ops(::Val{W}, x::UInt64, y::UInt64) where {W}
    narrow(v, name) = W == 64 ? "%$name = add i64 %$v, 0" : "%$name = trunc i64 %$v to i$W"
    widen(v, name; signed=false) = W == 64 ? "%$name = add i64 %$v, 0" :
        "%$name = $(signed ? "sext" : "zext") i$W %$v to i64"
    ir = """
    define [$NOPS x i64] @entry(i64 %x, i64 %y) #0 {
      $(narrow("x", "a"))
      $(narrow("y", "b"))
      %d0 = and i$W %b, $((big(1) << (W-1))-1)
      %d = or i$W %d0, 1
      %k64 = urem i64 %y, $W
      $(narrow("k64", "k"))
      $(widen("a", "r0"))
      $(widen("a", "r1"; signed=true))
      %bit = trunc i$W %a to i1
      %r2 = zext i1 %bit to i64
      %sc = icmp slt i$W %a, %b
      %r3 = zext i1 %sc to i64
      %uc = icmp ult i$W %a, %b
      %r4 = zext i1 %uc to i64
      %add = add i$W %a, %b
      $(widen("add", "r5"))
      %sub = sub i$W %a, %b
      $(widen("sub", "r6"))
      %mul = mul i$W %a, %b
      $(widen("mul", "r7"))
      %shl = shl i$W %a, %k
      $(widen("shl", "r8"))
      %lsr = lshr i$W %a, %k
      $(widen("lsr", "r9"))
      %asr = ashr i$W %a, %k
      $(widen("asr", "r10"))
      %sd = sdiv i$W %a, %d
      $(widen("sd", "r11"))
      %sr = srem i$W %a, %d
      $(widen("sr", "r12"))
      %ud = udiv i$W %a, %d
      $(widen("ud", "r13"))
      %ur = urem i$W %a, %d
      $(widen("ur", "r14"))
    """
    for j in 0:NOPS-1
        prev = j == 0 ? "poison" : "%t$(j-1)"
        ir *= "  %t$j = insertvalue [$NOPS x i64] $prev, i64 %r$j, $j\n"
    end
    ir *= "  ret [$NOPS x i64] %t$(NOPS-1)\n}\nattributes #0 = { alwaysinline }\n"
    return :(Base.llvmcall(($ir, "entry"), NTuple{NOPS,UInt64}, Tuple{UInt64,UInt64}, x, y))
end

@kernel cpu=false function width_kernel!(out, @Const(x), @Const(y), width)
    i = @index(Global, Linear)
    if i <= length(x)
        values = width_ops(width, x[i], y[i])
        for j in 1:NOPS
            @inbounds out[j, i] = values[j]
        end
    end
end

function reference(w, x, y)
    modulus = big(1) << w
    mask = modulus - 1
    a, b = big(x) & mask, big(y) & mask
    signed(v) = v >= modulus >> 1 ? v-modulus : v
    sa, sb = signed(a), signed(b)
    d = (b & ((modulus >> 1)-1)) | 1
    k = Int(y % UInt64(w))
    values = (a, sa, a & 1, sa < sb, a < b, a+b, a-b, a*b,
              a << k, a >> k, sa >> k, div(sa,d), rem(sa,d), div(a,d), rem(a,d))
    # Signed extension has a native 64-bit result; all arithmetic outputs zext W.
    return UInt64[UInt64(big(v) & (j == 2 ? (big(1)<<64)-1 : mask))
                  for (j,v) in enumerate(values)]
end

@testset "LLVM scalar integer widths" begin
    @test_throws ErrorException Lava.spirv_int_width(65)
    rng = MersenneTwister(0x11ba5e)
    for w in WIDTHS
        @testset "i$w" begin
            mask = w == 64 ? typemax(UInt64) : (UInt64(1)<<w)-1
            sign = UInt64(1)<<(w-1)
            edges = UInt64[0,1,2,3,sign-1,sign,sign+1,mask-1,mask,typemax(UInt64)]
            x = UInt64[a for a in edges for _ in edges]
            y = UInt64[b for _ in edges for b in edges]
            append!(x,rand(rng,UInt64,256)); append!(y,rand(rng,UInt64,256))
            if w <= 7
                append!(x,UInt64[a for a in 0:Int(mask) for _ in (0,1,mask)])
                append!(y,UInt64[b for _ in 0:Int(mask) for b in (0,1,mask)])
            end
            expected = hcat((reference(w,a,b) for (a,b) in zip(x,y))...)
            cpu = hcat((collect(width_ops(Val(w),a,b)) for (a,b) in zip(x,y))...)
            @test cpu == expected
            xd,yd = Lava.LavaArray(x),Lava.LavaArray(y)
            for rep in 1:3
                out = Lava.LavaArray{UInt64}(undef,NOPS,length(x))
                width_kernel!(KA.get_backend(xd))(out,xd,yd,Val(w);ndrange=length(x),workgroupsize=64)
                @test Array(out) == expected
            end
        end
    end
end
end

module LavaIntegerWidthBoundaryTests
using Test,Lava
const LLVM=Lava.LLVM

@testset "unsupported logical integer boundaries fail closed" begin
    for (name,body,decl) in (
        ("intrinsic", "%x = call i11 @llvm.ctpop.i11(i11 3)\nret void", "declare i11 @llvm.ctpop.i11(i11)"),
        ("load", "%x = load i11, ptr null\nret void", ""),
        ("store", "store i11 3, ptr null\nret void", ""),
        ("atomic", "%x = atomicrmw add ptr null, i11 3 seq_cst\nret void", ""),
    )
        @testset "$name" begin
            LLVM.Context() do _
                mod=parse(LLVM.Module,"$decl\ndefine void @entry() {\n$body\n}")
                firstinst=first(LLVM.instructions(first(LLVM.blocks(LLVM.functions(mod)["entry"]))))
                sm=Lava.SPIRVModule();ctx=Lava.SPIRVTypeContext(sm,Lava.build_pointee_type_map(mod));state=Lava.SPIRVEmitterState(sm,ctx)
                @test_throws r"Unsupported logical-width integer boundary" Lava.emit_instruction!(state,firstinst)
            end
        end
    end
    LLVM.Context() do _
        mod=parse(LLVM.Module,"define <2 x i11> @vector(<2 x i11> %a) { ret <2 x i11> %a }")
        ty=LLVM.value_type(first(LLVM.parameters(LLVM.functions(mod)["vector"])))
        sm=Lava.SPIRVModule();ctx=Lava.SPIRVTypeContext(sm,Lava.build_pointee_type_map(mod))
        @test_throws r"Unsupported vector of logical-width integers" Lava.map_type!(ctx,ty)
    end
end
end

module LavaIntegerWidthCastTests
using Test,Lava,Random
using Lava.KernelAbstractions: @kernel,@index,@Const
const KA=Lava.KernelAbstractions
@generated function cast_pair(::Val{S},::Val{D},x::UInt64) where {S,D}
    if S<D
        body="%a = trunc i64 %x to i$S\n%s = sext i$S %a to i$D\n%u = zext i$S %a to i$D\n%r0 = zext i$D %s to i64\n%r1 = zext i$D %u to i64"
    else
        body="%a = trunc i64 %x to i$S\n%t = trunc i$S %a to i$D\n%r0 = sext i$D %t to i64\n%r1 = zext i$D %t to i64"
    end
    ir="define [2 x i64] @entry(i64 %x) #0 {\n$body\n%t0 = insertvalue [2 x i64] poison, i64 %r0, 0\n%t1 = insertvalue [2 x i64] %t0, i64 %r1, 1\nret [2 x i64] %t1\n}\nattributes #0 = { alwaysinline }"
    :(Base.llvmcall(($ir,"entry"),NTuple{2,UInt64},Tuple{UInt64},x))
end
@kernel cpu=false function cast_kernel!(out,@Const(x),s,d)
    i=@index(Global,Linear)
    if i<=length(x)
        p=cast_pair(s,d,x[i]);out[1,i]=p[1];out[2,i]=p[2]
    end
end
@testset "casts sharing a physical bucket" begin
    rng=MersenneTwister(0xca57)
    for (s,d) in ((2,7),(7,3),(9,15),(15,11),(17,31),(31,23),(33,63),(63,53),(8,16),(16,8))
        x=rand(rng,UInt64,512);append!(x,UInt64[0,1,2,3,typemax(UInt64),UInt64(1)<<(s-1),(UInt64(1)<<s)-1])
        expected=Matrix{UInt64}(undef,2,length(x));sm=(big(1)<<s)-1;dm=(big(1)<<d)-1;full=(big(1)<<64)-1
        for (i,v) in enumerate(x)
            a=big(v)&sm
            if s<d
                signed=a>=(big(1)<<(s-1)) ? a-(big(1)<<s) : a
                expected[:,i]=UInt64[UInt64(signed&dm),UInt64(a)]
            else
                a&=dm;signed=a>=(big(1)<<(d-1)) ? a-(big(1)<<d) : a
                expected[:,i]=UInt64[UInt64(signed&full),UInt64(a)]
            end
        end
        @test hcat((collect(cast_pair(Val(s),Val(d),v)) for v in x)...)==expected
        xd=Lava.LavaArray(x)
        for rep in 1:3
            out=Lava.LavaArray{UInt64}(undef,2,length(x))
            cast_kernel!(KA.get_backend(xd))(out,xd,Val(s),Val(d);ndrange=length(x),workgroupsize=64)
            @test Array(out)==expected
        end
    end
end
end

module LavaIntegerFloatCastTests
using Test, Lava, Random
using Lava.KernelAbstractions: @kernel, @index, @Const
const KA = Lava.KernelAbstractions

@generated function integer_to_float(::Val{W}, x::UInt64) where {W}
    input = W == 64 ? "%a = add i64 %x, 0" : "%a = trunc i64 %x to i$W"
    ir = """
    define [4 x double] @entry(i64 %x) #0 {
      $input
      %s = sitofp i$W %a to double
      %u = uitofp i$W %a to double
      %b = trunc i64 %x to i1
      %bs = sitofp i1 %b to double
      %bu = uitofp i1 %b to double
      %t0 = insertvalue [4 x double] poison, double %s, 0
      %t1 = insertvalue [4 x double] %t0, double %u, 1
      %t2 = insertvalue [4 x double] %t1, double %bs, 2
      %t3 = insertvalue [4 x double] %t2, double %bu, 3
      ret [4 x double] %t3
    }
    attributes #0 = { alwaysinline }
    """
    :(Base.llvmcall(($ir,"entry"),NTuple{4,Float64},Tuple{UInt64},x))
end
@generated function float_to_integer(::Val{W}, s::Float64, u::Float64) where {W}
    signed = W == 64 ? "%sx = add i64 %a, 0" : "%sx = sext i$W %a to i64"
    unsigned = W == 64 ? "%ux = add i64 %b, 0" : "%ux = zext i$W %b to i64"
    ir = """
    define [2 x i64] @entry(double %s, double %u) #0 {
      %a = fptosi double %s to i$W
      %b = fptoui double %u to i$W
      $signed
      $unsigned
      %t0 = insertvalue [2 x i64] poison, i64 %sx, 0
      %t1 = insertvalue [2 x i64] %t0, i64 %ux, 1
      ret [2 x i64] %t1
    }
    attributes #0 = { alwaysinline }
    """
    :(Base.llvmcall(($ir,"entry"),NTuple{2,UInt64},Tuple{Float64,Float64},s,u))
end
@kernel cpu=false function to_float_kernel!(out,@Const(x),w)
    i=@index(Global,Linear)
    if i<=length(x)
        p=integer_to_float(w,x[i])
        for j in 1:4;out[j,i]=p[j];end
    end
end
@kernel cpu=false function to_integer_kernel!(out,@Const(s),@Const(u),w)
    i=@index(Global,Linear)
    if i<=length(s)
        p=float_to_integer(w,s[i],u[i]);out[1,i]=p[1];out[2,i]=p[2]
    end
end
same_bits(a,b)=size(a)==size(b)&&reinterpret(UInt64,vec(a))==reinterpret(UInt64,vec(b))
@testset "logical-width integer and floating casts" begin
    rng=MersenneTwister(0xf10a7)
    for w in (2,3,7,8,9,11,15,16,17,23,31,32,33,53,63,64)
        mask=(big(1)<<w)-1;sign=big(1)<<(w-1)
        x=UInt64[0,1,2,3,UInt64(sign-1),UInt64(sign),UInt64(sign+1),UInt64(mask),typemax(UInt64)]
        append!(x,rand(rng,UInt64,32))
        expected=Matrix{Float64}(undef,4,length(x))
        for (i,v) in enumerate(x)
            a=big(v)&mask;s=a>=sign ? a-(big(1)<<w) : a
            expected[:,i]=[Float64(s),Float64(a),isodd(v) ? -1.0 : 0.0,isodd(v) ? 1.0 : 0.0]
        end
        @test same_bits(hcat((collect(integer_to_float(Val(w),v)) for v in x)...),expected)
        xd=Lava.LavaArray(x)
        s=Float64[-1.75,-1,-0.5,-0.0,0.0,0.5,1,1.75]
        u=Float64[-0.0,0.5,1,1.75,2,2.75,3,3.75]
        expected_int=hcat(([reinterpret(UInt64,Int64(trunc(a))),UInt64(trunc(b))] for (a,b) in zip(s,u))...)
        @test hcat((collect(float_to_integer(Val(w),a,b)) for (a,b) in zip(s,u))...)==expected_int
        sd,ud=Lava.LavaArray(s),Lava.LavaArray(u)
        for rep in 1:3
            out=Lava.LavaArray{Float64}(undef,4,length(x))
            to_float_kernel!(KA.get_backend(xd))(out,xd,Val(w);ndrange=length(x),workgroupsize=64)
            @test same_bits(Array(out),expected)
            ints=Lava.LavaArray{UInt64}(undef,2,length(s))
            to_integer_kernel!(KA.get_backend(sd))(ints,sd,ud,Val(w);ndrange=length(s),workgroupsize=64)
            @test Array(ints)==expected_int
        end
    end
end
end
