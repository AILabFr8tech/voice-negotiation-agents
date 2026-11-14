"""How the run ended and whether it has to be paid for again.

The harness does **not** decide whether a call goes into the comparison (that is done by
`shared/metrics/`), but it does decide whether it must be repeated, which is a money
decision and has to be written down with its reason. It is the same decision in the 3
PoCs, so it lives here: if each harness had its own minimum, the three would populate
their percentiles with calls of different lengths and the p50 would stop comparing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - only for the type
    from shared.carrier_simulator.observed import Observed

# Turns a run has to reach to be usable. Below
# this the conversation did not get to negotiate anything and the run is
# repeated; above it, it counts even if the script was not acted out in full.
#
# The agent rushes to close, and not always on the same turn. A p50 that
# mixes 8-turn calls with 5-turn calls does not describe the same thing, so the
# coverage is always reported and not only when it fails.
MIN_VALID_TURNS: Final[int] = 5

# Why a run ended. They go into the JSON and the report verbatim.
END_COMPLETE: Final[str] = "guion_completo"
END_CLOSED_BY_AGENT: Final[str] = "cerrada_por_agente"
END_TIMEOUT: Final[str] = "vencimiento_de_la_espera"
END_AGENT_MUTE: Final[str] = "agente_sin_audio"
END_INCOMPLETE: Final[str] = "incompleta"


@dataclass(frozen=True)
class RunResult:
    """The run's two end conditions, and what to do with the run."""

    motivo: str
    reached: int
    of_script: int
    valid: bool
    retry: bool
    detail: str
    cut_at: str = ""
    # Turns with agent text and not a single frame with speech: the TTS did not sound
    # on that turn. They do not go into the percentile and count as a technical failure.
    no_audio: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.reached == self.of_script and self.of_script > 0

    @property
    def measurable(self) -> int:
        """Reached turns that were also heard. It is the base of the percentile."""
        return self.reached - len(self.no_audio)

    def as_json(self) -> dict[str, Any]:
        return {
            "motivo": self.motivo,
            "turns_reached": self.reached,
            "script_turns": self.of_script,
            "cerrada_por_agente": self.motivo == END_CLOSED_BY_AGENT,
            "cut_at": self.cut_at,
            "valid": self.valid,
            "retry": self.retry,
            "min_turns": MIN_VALID_TURNS,
            "turns_without_agent_audio": list(self.no_audio),
            "measurable_turns": self.measurable,
            "detail": self.detail,
        }


def evaluate_run(observed: "Observed") -> RunResult:
    """What happened with the run and whether it has to be repeated.

    The order of the rules matters, and the first one is the least obvious: **if the
    agent produced text but not a single frame with speech was heard, the run does not
    count, however complete it looks**. That is the TTS failure that leaves the
    carrier listening to silence while the worker's JSON fills up with
    perfect dialogue; without this check it would be reported as a good call.

    That rule is about the whole run, and that is why **it is not enough on its own**: a partial
    TTS failure (audio on 6 of 8 turns) passes it cleanly, and the 2 mute
    turns would slip into the percentile with the latency the worker computed over
    audio nobody heard. Mute turns are counted one by one
    (`ObservedTurn.text_without_audio`), they leave the count of measurable turns, and the
    minimum is required over **those**, not over the reached ones.

    Then, the turn rule: >= `MIN_VALID_TURNS` counts even if
    the script was not acted out in full. The agent closing at T6 is not a
    harness failure; it is agent behavior, and it is recorded as such instead
    of discarding the run.
    """
    scope = observed.turns_reached()
    reached, of_script = scope["reached"], scope["of_script"]
    no_audio = tuple(t.step.turn for t in observed.turns_without_audio())

    if observed.agent_fragments() and observed.t_agent_first_speech is None:
        return RunResult(
            motivo=END_AGENT_MUTE,
            reached=reached,
            of_script=of_script,
            valid=False,
            retry=True,
            cut_at=scope["cut_at"],
            no_audio=no_audio,
            detail=(
                "el agente generó texto pero no salió una sola trama con voz en "
                "toda la llamada: el TTS no sonó. La transcripción del worker se "
                "ve completa y la llamada no lo está"
            ),
        )

    if observed.was_closed_by_agent:
        motivo = END_CLOSED_BY_AGENT
        detail = (
            f"el agente cerró la negociación y borró la sala en {scope['cut_at'] or 'el último turno'}"
            f" ({observed.close_reason or 'sin motivo'})"
        )
    elif reached == of_script and of_script:
        motivo, detail = END_COMPLETE, "el guion se actuó completo"
    elif scope["motivo"]:
        motivo, detail = END_TIMEOUT, scope["motivo"]
    else:
        motivo, detail = END_INCOMPLETE, "la corrida no llegó al final del guion"

    # The minimum is required over the **measurable** turns, not over the
    # reached ones: a turn the agent answered in writing that was not heard does not
    # contribute latency, so counting it toward the minimum would count it wrongly
    # twice: it is excluded from the percentile and yet it would have vouched for the run.
    measurable = reached - len(no_audio)
    valid = measurable >= MIN_VALID_TURNS
    if no_audio:
        detail += (
            f"; {len(no_audio)} turno(s) con texto y sin audio "
            f"({', '.join(no_audio)}): fuera del percentil, a confiabilidad"
        )
    return RunResult(
        motivo=motivo,
        reached=reached,
        of_script=of_script,
        valid=valid,
        retry=not valid,
        cut_at=scope["cut_at"],
        no_audio=no_audio,
        detail=(
            detail
            if valid
            else f"{detail}; con {measurable} turno(s) medible(s) no llega al "
            f"mínimo de {MIN_VALID_TURNS} y hay que repetirla"
        ),
    )
