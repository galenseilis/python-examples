from typing import Optional, Callable, Any, Tuple
import time
import sys

import click


def trace_function(frame: Any, event: str, arg: Any) -> Optional[Callable]:
    """
    Trace function for monitoring function calls.

    Args:
        frame (frame): The current frame being executed.
        event (str): The event type triggering the trace function.
        arg (Any): The argument associated with the event.

    Returns:
        Optional[Callable]: The trace function or None to stop tracing.
    """
    if not hasattr(trace_function, "log_initialized"):
        # Initialize log file with column titles if not already done
        with open("trace_log.txt", "w") as log_file:
            log_file.write("Timestamp,Event,Function,File,Line,Argument\n")
        trace_function.log_initialized = True

    current_time = time.time()
    file_name = frame.f_globals.get("__file__", "unknown")
    try:
        co_name = frame.f_code.co_name
    except Exception as co_name_exception:
        co_name = co_name_exception
    try:
        f_lineno = frame.f_lineno
    except Exception as f_lineno_exception:
        f_lineno = f_lineno_exception
    try:
        arg = str(arg)
    except Exception as arg_exception:
        arg = arg_exception
    finally:
        if len(str(arg)) > 80:
            arg = "LONG_ARG"
        arg = None
    log_entry = f"{current_time},{event},{co_name},{file_name},{frame.f_lineno},{arg}\n"
    with open("trace_log.txt", "a") as log_file:
        log_file.write(log_entry)
    return trace_function


@click.command()
@click.argument("target_script", type=click.Path(exists=True))
def trace(target_script: str) -> None:
    """
    Trace function to monitor the execution of a target script.

    Args:
        target_script (str): Path to the target script to be traced.

    Returns:
        None
    """
    # Set the trace function
    sys.settrace(trace_function)

    # Run the target script
    with open(target_script, "r") as script_file:
        exec(script_file.read(), {})

    # Disable the trace function
    sys.settrace(None)


if __name__ == "__main__":
    trace()
