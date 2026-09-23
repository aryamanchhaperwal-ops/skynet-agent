"""Sandbox + trial context for experiment procedures.

The sandbox is the *capability boundary* of the experimentation engine
(blueprint "Sandboxed" stage). A trial procedure receives a
:class:`TrialContext` — the only object it can act through — and the
context refuses anything outside the declared boundary:

- **No source access**: procedures are registry artifacts; the context
  exposes data and services explicitly handed to the experiment, never
  the filesystem of the running Skynet, environment secrets, or module
  globals.
- **Bounded**: every trial carries a hard wall-clock timeout enforced by
  the runner via ``asyncio.wait_for``.
- **Scratch space only**: ``scratch_dir`` is a per-experiment temp
  directory the trial may write; it is deleted when the sandbox closes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


class SandboxError(RuntimeError):
    """Raised when a trial attempts something outside its boundary."""


class TrialContext:
    """The read-mostly world a single trial procedure may see.

    ``services`` is the explicitly-injected map of capabilities the
    experiment declared (e.g. a search provider, an intelligence service).
    Nothing else about the host process is reachable through it — no env
    vars, no config objects, no module imports of the agent's own code.
    """

    def __init__(
        self,
        *,
        experiment_id: str,
        arm: str,
        trial_index: int,
        seed: int | None,
        params: dict[str, Any],
        services: dict[str, Any] | None = None,
        scratch_dir: Path | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.arm = arm
        self.trial_index = trial_index
        self.seed = seed
        self.params = dict(params)
        self.services: dict[str, Any] = dict(services or {})
        self._scratch_dir = scratch_dir

    # -- sandboxed resources -------------------------------------------------

    @property
    def scratch_dir(self) -> Path:
        """Per-experiment scratch directory (created on first access)."""
        if self._scratch_dir is None:
            raise SandboxError("this sandbox exposes no scratch directory")
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        return self._scratch_dir

    def service(self, name: str) -> Any:
        """Fetch an explicitly injected service; unknown names raise."""
        if name not in self.services:
            raise SandboxError(
                f"service {name!r} was not injected into this experiment; "
                "experiments may only use explicitly declared capabilities"
            )
        return self.services[name]

    # -- deliberately absent --------------------------------------------------
    # (documented, enforced by not existing: os.environ, database sessions,
    #  the action registry, the memory manager, settings objects, file paths
    #  outside scratch_dir, network credentials.)


class Sandbox:
    """Lightweight isolation boundary for one experiment.

    Creates a scratch temp directory, hands every trial a
    :class:`TrialContext`, and cleans up on close. Heavier isolation
    (subprocess, container) implements the same interface later without
    touching the runner.
    """

    def __init__(self, *, experiment_id: str, with_scratch: bool = True) -> None:
        self._experiment_id = experiment_id
        self._scratch: Path | None = None
        self._with_scratch = with_scratch
        if with_scratch:
            root = Path(tempfile.gettempdir()) / "skynet-sandbox"
            root.mkdir(parents=True, exist_ok=True)
            self._scratch = Path(tempfile.mkdtemp(prefix=f"{experiment_id[:12]}-", dir=root))

    def context_for(
        self,
        *,
        arm: str,
        trial_index: int,
        seed: int | None,
        params: dict[str, Any],
        services: dict[str, Any] | None = None,
    ) -> TrialContext:
        return TrialContext(
            experiment_id=self._experiment_id,
            arm=arm,
            trial_index=trial_index,
            seed=seed,
            params=params,
            services=services,
            scratch_dir=self._scratch,
        )

    def close(self) -> None:
        """Remove the scratch directory (idempotent)."""
        if self._scratch is not None and self._scratch.exists():
            import shutil

            shutil.rmtree(self._scratch, ignore_errors=True)
        self._scratch = None

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["Sandbox", "SandboxError", "TrialContext"]
