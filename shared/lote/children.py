"""Make sure no child of the batch outlives it.

The batch launches two kinds of process that cost money while they live: the
harness of the call in progress (which is talking to Deepgram, OpenAI and
ElevenLabs) and the PoC2 worker. The orderly path closes them: `Ctrl+C` kills
the in-flight call and the `finally` shuts the worker down.

The disorderly path does not. Killing the batch mid-call with
`Stop-Process -Force` leaves **orphans**: the worker, that call's harness and
the pipeline server the harness started. The progress on disk stays intact,
but the call keeps spending credit with nobody to collect the result, and on
resume it would write its summary over that of the new call.

A `finally` cannot fix that: `Stop-Process -Force` runs nothing in the process
it kills. On Windows the only guarantee comes from the operating system: a
**Job Object** with `KILL_ON_JOB_CLOSE`. Every process placed in the job dies
when the job is closed, and the job is closed when its last handle is released,
which always happens when the batch ends, even through a `Stop-Process` or a
power cut of the process itself. Grandchildren inherit job membership, so the
PoC1 pipeline server goes down too.

Outside Windows this does nothing and says so: no campaign runs there yet, and
faking the guarantee would be worse than not having it.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes
from typing import Any

# `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`, class 9 of `SetInformationJobObject`.
EXTENDED_LIMITS_CLASS = 9
# `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`: when the job is closed, it kills whatever
# is inside. This flag is the whole point of this module.
KILL_ON_CLOSE = 0x00002000

IS_WINDOWS = sys.platform == "win32"


if IS_WINDOWS:  # pragma: no branch - the campaign runs on Windows

    class _IOCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IOCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _create_job() -> Any:
    """Create the job with the flag set, or `None` if that failed."""
    if not IS_WINDOWS:
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        return None

    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = KILL_ON_CLOSE
    was_set = kernel.SetInformationJobObject(
        job,
        EXTENDED_LIMITS_CLASS,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    )
    if not was_set:
        kernel.CloseHandle(job)
        return None
    return job


# A single job for all children, and its handle lives as long as the process:
# **releasing** it is what triggers the kill, so keeping it in a global is not
# carelessness but the mechanism. It is created the first time it is needed.
_JOB: Any = None
_JOB_ATTEMPTED = False


def batch_job() -> Any:
    global _JOB, _JOB_ATTEMPTED
    if not _JOB_ATTEMPTED:
        _JOB_ATTEMPTED = True
        _JOB = _create_job()
    return _JOB


def adopt(process: subprocess.Popen[Any]) -> bool:
    """Put the process (and its descendants) into the batch's job.

    Returns whether it was adopted. A `False` is no reason not to run the
    call: it means that if the batch dies badly, cleanup will have to be done
    by hand, which is exactly how things were before this module.
    """
    job = batch_job()
    if job is None:
        return False
    handler_fn = getattr(process, "_handle", None)
    if handler_fn is None:
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    return bool(kernel.AssignProcessToJobObject(job, int(handler_fn)))
