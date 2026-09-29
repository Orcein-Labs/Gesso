# Logging & telemetry conventions (Phase 0, Gesso_Stack.md §LXXIII).
#
# Laws this file exists to enforce (before the machinery that will enforce them
# better exists):
#
#   §LXX   Failure must be explicit. No silent representation downgrade, backend
#          switch, quantization mismatch, memory-plan violation, or kernel
#          substitution unless policy explicitly permits it. If fallback occurs,
#          record it.
#
#   §XLII  Receipts. Significant actions are auditable. Every event carries a
#          timestamp; domain code attaches structured key-value context, never
#          prose-only messages.
#
#   §XLIX  Performance observability. Gesso must explain performance. Phase 0
#          only fixes the event vocabulary; collection comes with Phase 6.
#
# Conventions for all future Gesso code:
#
#   * Log through `glog(level, event, pairs...)` — never bare `@info` in
#     library code, so events remain structured and greppable.
#   * `event` is a short snake_case symbol naming the event kind
#     (e.g. :backend_fallback, :plan_selected).
#   * Context is key-value pairs. Values should be primitives or strings that
#     survive serialization into future receipts.
#   * Nothing in Gesso logs model weights, prompts, or generated text by
#     default. Content belongs in receipts, not logs.

module Log

export glog, GessoLogConfig, min_level!, current_config
export @gfallback

using Dates

@enum LogLevel begin
    LOG_DEBUG = 0
    LOG_INFO = 1
    LOG_WARN = 2
    LOG_ERROR = 3
end

Base.@kwdef mutable struct GessoLogConfig
    min_level::LogLevel = LOG_INFO
    io::IO = stderr
    timestamps::Bool = true
end

const GLOBAL_CONFIG = GessoLogConfig()

current_config() = GLOBAL_CONFIG

"""
    min_level!(level) -> LogLevel

Set the global minimum log level. Returns the previous level.
"""
function min_level!(level::LogLevel)
    prev = GLOBAL_CONFIG.min_level
    GLOBAL_CONFIG.min_level = level
    return prev
end

const LEVEL_TAG = Dict{LogLevel, String}(
    LOG_DEBUG => "debug",
    LOG_INFO => "info",
    LOG_WARN => "warn",
    LOG_ERROR => "error",
)

"""
    glog([io], level, event; kw...) -> Nothing

Emit one structured Gesso event.

```julia
glog(LOG_WARN, :backend_fallback; requested = :lava, actual = :cuda, reason = "no device")
```

Events below the configured minimum level are dropped. `event` must be a Symbol.
"""
function glog(io::IO, level::LogLevel, event::Symbol; kw...)
    GLOBAL_CONFIG.min_level <= level || return nothing
    prefix = LEVEL_TAG[level]
    if GLOBAL_CONFIG.timestamps
        print(
            io,
            '[',
            prefix,
            "] ",
            Dates.format(now(), dateformat"yyyy-mm-dd\ HH:MM:SS"),
            " ",
        )
    else
        print(io, '[', prefix, "] ")
    end
    print(io, event)
    for (k, v) in kw
        print(io, ' ', k, '=', _render(v))
    end
    println(io)
    return nothing
end

glog(level::LogLevel, event::Symbol; kw...) = glog(GLOBAL_CONFIG.io, level, event; kw...)

_render(x) = repr(x)
_render(x::AbstractString) = x
_render(x::Symbol) = string(x)

"""
    @gfallback(allowed, actual, [kw...])

Record a policy-permitted fallback (§LXX). A fallback that is not logged through
this macro is a law violation waiting to be found. Emits a `LOG_WARN` event
(`:fallback`) recording what was requested and what was actually used, and
evaluates to the fallback value so call sites can write:

```julia
dev = lava_device_or_nothing() === nothing ? @gfallback(:lava, :cuda) : lava_device()
```
"""
macro gfallback(requested, actual)
    res = gensym(:fallback_result)
    quote
        $res = $(esc(actual))
        $(glog)($(LOG_WARN), :fallback; requested=($(esc(requested))), actual=($res))
        $res
    end
end

end # module Log

using .Log: glog, GessoLogConfig, min_level!, current_config, @gfallback
export glog, GessoLogConfig, min_level!, current_config, @gfallback
