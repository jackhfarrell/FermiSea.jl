
# TODO: we might want to add this upstream, or just make it part of the analysis or Alive
# callbacks that Trixi implements?
"""
    FlushOutputCallback(; interval=100)

This is just a lightweight callback that flushes stdout and stderr every `interval`
iterations.  I was trying to run simulations on the cluster and wanted the logs to show up
immediately in the log files, but Julia seems to aggressively buffer output
"""
function FlushOutputCallback(; interval::Int=100)
    interval > 0 || throw(ArgumentError("interval must be positive"))

    affect! = function (integrator)
        # Julia buffers stdout/stderr aggressively; flush explicitly so progress
        # appears promptly under batch schedulers and redirected output.
        flush(stdout)
        flush(stderr)
        return nothing
    end

    return DiscreteCallback(
        (u, t, integrator) -> integrator.iter % interval == 0,
        affect!;
        save_positions=(false, false))
end

"""
    MonitorCallback(; interval=100)

Backward-compatible alias for [`FlushOutputCallback`](@ref).

Historically this callback only flushed IO; it does not perform solution
analysis.
"""
MonitorCallback(; interval::Int=100) = FlushOutputCallback(; interval)
