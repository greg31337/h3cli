"""Run build children in owned process groups and propagate cancellation."""
import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import contextmanager


@contextmanager
def cancellation():
    def cancel(signum, frame):
        raise KeyboardInterrupt
    previous = {s: signal.signal(s, cancel) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stop(process):
    # A second Ctrl+C must not interrupt cleanup and strand a compiler group.
    previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        _stop(process)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _stop(process):
    # Descendants can outlive their group leader, so signal the owned group even
    # when poll() reports the direct child has already exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + (5 if sig == signal.SIGTERM else 1)
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                # On macOS a group containing only unreaped zombies can report
                # EPERM here. Poll again to reap the child before probing it.
                # Permission failures from TERM/KILL still propagate normally.
                pass
            time.sleep(.05)
        else:
            continue
        break
    process.wait()


def run(argv, *, check=False, capture_output=False, timeout=None, **kwargs):
    """subprocess.run with cancellation of the entire owned child group."""
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with cancellation():
        process = subprocess.Popen(list(map(str, argv)), start_new_session=True, **kwargs)
        try:
            # Keep a supplied stdin PIPE open as an SSH cancellation lifeline.
            if kwargs.get('stdin') == subprocess.PIPE:
                process.wait(timeout=timeout)
                stdout = stderr = None
            else:
                stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            stop(process)
            raise
        finally:
            if process.stdin:
                process.stdin.close()
    result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def supervise(argv):
    """Remote SSH command: EOF on stdin cancels work after connection loss."""
    with cancellation():
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(sys.stdin, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                if selector.select(.2) and not os.read(sys.stdin.fileno(), 4096):
                    raise KeyboardInterrupt
            return process.returncode
        except BaseException:
            stop(process)
            raise
        finally:
            selector.close()


if __name__ == '__main__':
    try:
        args = sys.argv[1:]
        if args[:1] == ['--']:
            args = args[1:]
        if not args:
            raise ValueError('A child command is required')
        sys.exit(supervise(args))
    except KeyboardInterrupt:
        sys.exit(130)
