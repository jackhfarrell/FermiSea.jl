# I made a custom logger before I was able to figure out how to get AnalysisCallback working
# and printing the things I wanted to print.  It's still here.  This logger is just a 
# simple wrapper around the normal logger from Logging.jl; the only change is that it
# flushes the output after every message, which I needed for the logs to update on the 
# SLURM-managed cluster I was using

struct FlushingLogger{L, IO_t} <: Logging.AbstractLogger
    logger::L
    io::IO_t
end

Logging.min_enabled_level(logger::FlushingLogger) = Logging.min_enabled_level(logger.logger)
Logging.shouldlog(logger::FlushingLogger, args...) = Logging.shouldlog(logger.logger, args...)
Logging.catch_exceptions(logger::FlushingLogger) = Logging.catch_exceptions(logger.logger)

function Logging.handle_message(logger::FlushingLogger, args...; kwargs...)
    Logging.handle_message(logger.logger, args...; kwargs...)
    flush(logger.io)
    return nothing
end

"""
    make_flushing_logger(io=stderr; min_level=Logging.Info)

Create a logger that forwards messages to `io` and flushes the stream after each
message.

This is useful for batch jobs where stdout or stderr is redirected to a log file
and buffered output would otherwise appear late.
"""
function make_flushing_logger(io=stderr; min_level=Logging.Info)
    return FlushingLogger(LoggingExtras.TransformerLogger(identity, Logging.SimpleLogger(io, min_level)),
                          io)
end
