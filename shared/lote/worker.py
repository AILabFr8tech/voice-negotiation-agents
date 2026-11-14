"""The PoC2 worker, as seen from the batch.

PoC1 takes care of itself: its harness starts the pipeline server in a
subprocess and kills it on hang-up. PoC2 does not: its agent is a worker
registered with LiveKit that lives in another terminal and stays up between
calls, because registering and preloading Silero costs considerably more than
a call.

For an unattended batch that leaves two loose ends: that the worker is up when
PoC2's turn comes, and that if it dies mid-campaign someone brings it back up.
This class is that someone.

**It changes nothing about how the worker runs.** It launches exactly the same
command one types by hand (`python -m poc2_livekit.app.agent start`), with the
same interpreter. If the batch runs with `--no-worker`, not even that: it
assumes one is already up, which is how every call has been run to date.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from shared.lote.children import adopt

# What `livekit.agents` prints when the worker has registered and can receive
# dispatch (`worker.py`, "registered worker"). Waiting to see it is the
# difference between a first PoC2 call that fails from a race and one that does
# not; without it the harness would dispatch to a worker that does not exist yet.
READY_SIGNAL: str = "registered worker"

# Loading the plugins and preloading Silero before registering takes a while.
# Two minutes is comfortable on a cold machine and still a clear failure if it
# does not happen.
STARTUP_CAP_S: float = 120.0


class PoC2Worker:
    """Keeps a PoC2 worker up for as long as the batch lasts."""

    def __init__(
        self,
        *,
        interpreter: Path,
        root: Path,
        log_file_path: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.interpreter = interpreter
        self.root = root
        self.log_file_path = log_file_path
        self.environment = environment
        self.process: subprocess.Popen[bytes] | None = None
        self.onsets = 0
        self._log = None

    # -- Lifecycle --

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, *, tope_seg: float = STARTUP_CAP_S) -> bool:
        """Launch the worker and wait for it to register. Returns whether it did.

        It raises no exception when it fails: a worker that does not start must
        leave the PoC2 calls marked as failures and let the PoC1 calls run, not
        bring down the whole campaign.
        """
        self.halt()
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Opened in append mode: a restart mid-campaign must not erase the trace
        # of why the previous worker died.
        self._log = open(self.log_file_path, "ab")
        self.onsets += 1
        self._log.write(
            f"\n=== startup {self.onsets} — {time.strftime('%Y-%m-%d %H:%M:%S')}"
            " ===\n".encode()
        )
        self._log.flush()
        self.process = subprocess.Popen(  # noqa: S603 - our own arguments
            [str(self.interpreter), "-m", "poc2_livekit.app.agent", "start"],
            cwd=str(self.root),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=self.environment,
        )
        # It must not outlive the batch: an orphan worker stays registered and
        # the next campaign would dispatch to two. See `shared/lote/children.py`.
        adopt(self.process)
        return self.wait_until_registered(tope_seg=tope_seg)

    def wait_until_registered(self, *, tope_seg: float) -> bool:
        """Watch the log until the signal shows up, or until the worker dies."""
        limit = time.monotonic() + tope_seg
        while time.monotonic() < limit:
            if not self.alive:
                return False
            if self.was_recorded():
                return True
            time.sleep(0.5)
        return False

    def was_recorded(self) -> bool:
        """Whether the registration signal has already appeared in this startup's log."""
        try:
            raw_value = self.log_file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        # Only the segment of the current startup: a `registered worker` from a
        # previous startup would accept a worker that has just died.
        mark = f"=== startup {self.onsets} —"
        position = raw_value.rfind(mark)
        segment = raw_value[position:] if position >= 0 else raw_value
        return READY_SIGNAL in segment

    def ensure(self, *, tope_seg: float = STARTUP_CAP_S) -> bool:
        """Bring the worker up if it is not. This is what the batch calls."""
        if self.alive:
            return True
        return self.start(tope_seg=tope_seg)

    def halt(self) -> None:
        """Shut it down. Idempotent: also called when it was never started."""
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        if self._log is not None:
            try:
                self._log.close()
            finally:
                self._log = None

    def __enter__(self) -> "PoC2Worker":
        return self

    def __exit__(self, *_: object) -> None:
        self.halt()
