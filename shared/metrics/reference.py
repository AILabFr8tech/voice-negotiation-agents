"""WER reference file: generated pre-filled, corrected by hand.

The WER needs to know **what was really said**. A machine cannot make that up:
the person who acted out the script knows it. Nor do they need to write it from
scratch, precisely because they were following a script: what is done here is
to seed the file with the scenario's lines and let the person **correct only
where they improvised**. From typing a whole transcript (15-20 min per call) to
correcting deviations (2-3 min).

**The reference is the acted script, not a recording.** That method does not
depend on the voice framework, so it is identical in the 3 PoCs — which is
precisely what makes WERs comparable across architectures.

## The format

Plain text, line-oriented, editable in any editor:

    yo:  what you said               <- THE ONLY thing edited; feeds the WER
    stt: what the STT understood     <- context only, not edited, not measured

Everything else (`#` comments, `##` headers, free notes) is ignored by the
parser, so it can be annotated freely without breaking the computation.

## The bias to avoid

The `stt:` line is shown alongside because it helps locate oneself in the audio,
but copying the `stt:` into the `yo:` line produces WER 0 and destroys the
measurement. The correction is made **against the audio**, not against the STT.
The file header says so too, where it will be read.

## Why WER is computed over the full channel

The STT splits spontaneous speech into fragments ("Sí, claro que sí." / "Este," /
"mañana,"), so there is no 1-to-1 correspondence between script turn and STT
fragment. Matching turns by hand would be manual work — exactly what is being
eliminated. Instead, the number that counts is the **full channel** one: all
the `yo:` lines against all of the carrier's STT text. The grouping by turns
this module does is only so the file reads comfortably; per-turn WER is
reported as indicative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from shared.metrics.scenarios import Scenario, ScriptTurn

# Line prefixes. `yo:` is the only one that feeds the computation.
REFERENCE_PREFIX: Final[str] = "yo:"
HYPOTHESIS_PREFIX: Final[str] = "stt:"

# Separator between STT fragments within a single `stt:` line.
FRAGMENT_SEPARATOR: Final[str] = " | "

NOTES_HEADER: Final[str] = "## Notas manuales"
FREE_HEADER: Final[str] = "## Notas libres"
LEFTOVERS_HEADER: Final[str] = "## Fragmentos del STT sin turno del guion"

# The only two numbers the person has to enter by hand.
SCORE_FIELDS: Final[tuple[str, ...]] = ("naturalness", "turn_taking")
# Optional rubric counts; if left empty they are not made up.
COUNT_FIELDS: Final[tuple[str, ...]] = (
    "unwanted_interruptions",
    "false_end_of_turn",
)
METADATA_FIELDS: Final[tuple[str, ...]] = (
    "poc",
    "scenario",
    "repetition",
    "call_sid",
    "script",
    # Where the WER hypothesis comes from. By default `json`: the `stt:` lines
    # of a hand-corrected file are a copy for reading, and if someone edited
    # them by mistake they must not contaminate the measurement. A file
    # **derived** from the clips (§2.10) sets `file`, because there the `stt:`
    # lines are not a copy: they carry the exclusions —dead tail and pairing—
    # and taking the whole JSON would bring back the text of the excluded turns
    # as insertions.
    "hypothesis",
)

JSON_HYPOTHESIS: Final[str] = "json"
FILE_HYPOTHESIS: Final[str] = "file"

# Gap between carrier fragments that, with no agent turns in between, is taken
# as a turn change. Used only as a fallback: if the transcript carries the
# agent's turns, the cut is made there, which is the real signal that the turn
# changed.
TURN_GAP_S: Final[float] = 4.0

CARRIER_ROLE: Final[str] = "carrier"
AGENT_ROLE: Final[str] = "agent"

_CLAVE_VALOR: Final[re.Pattern[str]] = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$")


@dataclass
class ManualNotes:
    """What the person fills in by hand. None of this can be automated."""

    naturalness: int | None = None
    turn_taking: int | None = None
    unwanted_interruptions: int | None = None
    false_end_of_turn: int | None = None
    free: str = ""

    @property
    def completed(self) -> bool:
        """True if both 1-5 scores, the mandatory ones, are present."""
        return self.naturalness is not None and self.turn_taking is not None

    def absent(self) -> list[str]:
        return [c for c in SCORE_FIELDS if getattr(self, c) is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "naturalness": self.naturalness,
            "turn_taking": self.turn_taking,
            "unwanted_interruptions": self.unwanted_interruptions,
            "false_end_of_turn": self.false_end_of_turn,
            "free": self.free,
            "completed": self.completed,
        }


@dataclass
class ReferenceTurn:
    """One turn of the file: what was said and what the STT understood."""

    id: str
    reference: str = ""
    hypothesis: str = ""
    when: str = ""

    @property
    def empty(self) -> bool:
        return not self.reference.strip() and not self.hypothesis.strip()


@dataclass
class ReferenceFile:
    """Parsed content of a reference file."""

    metadata: dict[str, str] = field(default_factory=dict)
    turns: list[ReferenceTurn] = field(default_factory=list)
    notes: ManualNotes = field(default_factory=ManualNotes)
    ruta: Path | None = None

    @property
    def reference_text(self) -> str:
        """All the `yo:` lines joined: the carrier's full channel."""
        return " ".join(t.reference.strip() for t in self.turns if t.reference.strip())

    @property
    def hypothesis_text(self) -> str:
        """Everything the STT understood, including what fell into no turn."""
        return " ".join(t.hypothesis.strip() for t in self.turns if t.hypothesis.strip())

    def integer(self, clave: str) -> int | None:
        raw_value = self.metadata.get(clave, "")
        try:
            return int(str(raw_value).strip())
        except (TypeError, ValueError):
            return None


# --------------------------------------------------------------------------
# Grouping the transcript into turns
# --------------------------------------------------------------------------


def group_carrier_turns(
    transcript: Sequence[dict[str, Any]],
    *,
    hole: float = TURN_GAP_S,
) -> list[str]:
    """Group the STT fragments into conversational turns.

    Primary criterion: **an agent turn closes the carrier's turn.** It is the
    real signal of a turn change, and it is available since the pipeline also
    annotates what the agent says.

    Fallback, when the transcript carries no agent turns (calls recorded before
    the annotator was instrumented): cut on time gaps. It is worse —spontaneous
    speech leaves gaps of several seconds within a single turn— but the grouping
    only affects how the file reads, not the full-channel WER.
    """
    has_agent = any(t.get("role") == AGENT_ROLE for t in transcript)

    groups: list[list[str]] = []
    actual: list[str] = []
    ultimo_t: float | None = None

    for row in transcript:
        role = row.get("role")
        text = str(row.get("text", "")).strip()

        if role == AGENT_ROLE:
            if has_agent and actual:
                groups.append(actual)
                actual = []
                ultimo_t = None
            continue

        if role != CARRIER_ROLE or not text:
            continue

        try:
            t = float(row.get("t"))
        except (TypeError, ValueError):
            t = ultimo_t if ultimo_t is not None else 0.0

        if not has_agent and actual and ultimo_t is not None and t - ultimo_t > hole:
            groups.append(actual)
            actual = []

        actual.append(text)
        ultimo_t = t

    if actual:
        groups.append(actual)

    return [FRAGMENT_SEPARATOR.join(g) for g in groups]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

_INSTRUCTIONS: Final[str] = """\
# ============================================================================
# REFERENCIA DE TRANSCRIPCIÓN — para calcular el WER
# ============================================================================
#
# QUÉ HACER (2-3 minutos, es todo el trabajo manual que pide este archivo):
#
#   1. Ten a la mano el guion del escenario que actuaste: la referencia eres tú.
#   2. Las líneas `yo:` ya vienen con la línea del guion. Corrige SOLO donde te
#      desviaste: si dijiste la línea tal cual, no toques nada.
#   3. Si improvisaste algo que no está en el guion, agrégalo como otra línea
#      `yo:` en el turno que corresponda.
#   4. Si un turno del guion nunca lo dijiste, deja su línea `yo:` vacía.
#   5. Llena `naturalidad` y `turn_taking` (1-5) hasta abajo.
#
#   Todo lo numérico —WER, CER, error de montos, percentiles— se calcula solo.
#
# OJO CON EL SESGO:
#   Las líneas `stt:` son SOLO CONTEXTO, para que te ubiques en la conversación.
#   No las edites y no las copies a `yo:`. Si copias el STT, el WER sale 0 y la
#   medición no vale nada. Corrige contra lo que DIJISTE, nunca contra el STT.
#   No hay grabación a la cual volver: esta disciplina es la única defensa que
#   tiene la medición, así que llena el archivo con la llamada fresca.
#
# FORMATO:
#   `yo:`  -> alimenta el cálculo. Es lo único que se edita.
#   `stt:` -> lo que entendió el STT. Ignorado por el cálculo.
#   `#`    -> comentario, ignorado. Anota con libertad.
#   Los montos escríbelos como los dijiste ("dieciocho mil" o "18,000"): las dos
#   formas cuentan igual, la normalización las vuelve equivalentes.
# ============================================================================
"""


def _commented_line(label: str, text: str) -> str:
    return f"#   {label}: {text}"


def _turn_block(
    script_turn: ScriptTurn | None,
    hypothesis: str,
    *,
    turn_id: str,
) -> list[str]:
    """The lines of one turn of the file."""
    text_lines: list[str] = []

    when = script_turn.when if script_turn else ""
    header = f"## {turn_id}" + (f" — {when}" if when else "")
    text_lines.append(header)

    if script_turn:
        if script_turn.note:
            text_lines.append(_commented_line("note", script_turn.note))
        if script_turn.silence_s is not None:
            text_lines.append(
                _commented_line(
                    "silence", f"{script_turn.silence_s:g} s marcados por el guion"
                )
            )
        if script_turn.hangs_up:
            text_lines.append(_commented_line("action", "cuelgas sin decir nada más"))
        for alternate in script_turn.alternates:
            text_lines.append(_commented_line("alt", alternate))
        if script_turn.is_silence:
            text_lines.append(
                "#   este turno es silencio: si el STT transcribió algo, cuenta "
                "como alucinación"
            )

    linea_yo = script_turn.linea if script_turn else ""
    text_lines.append(f"{REFERENCE_PREFIX} {linea_yo}".rstrip())
    text_lines.append(f"{HYPOTHESIS_PREFIX} {hypothesis}".rstrip())
    return text_lines


def generate(
    *,
    scenario: Scenario | None,
    transcript: Sequence[dict[str, Any]] = (),
    poc: int | str = "",
    repetition: int | str = "",
    call_sid: str = "",
    script_version: str = "",
) -> str:
    """Build the reference file's text, ready to write and edit.

    `scenario` can be None (a call outside the script, like the warm-up one):
    then there are no seeded lines and the file is left with the STT fragments
    to transcribe by hand.
    """
    groups = group_carrier_turns(transcript)
    script_turns = list(scenario.turns) if scenario else []

    parts: list[str] = [_INSTRUCTIONS]

    if scenario:
        parts.append(f"# Escenario {scenario.label}")
        if scenario.mide:
            parts.append(f"#   mide: {scenario.mide}")
        if scenario.expected_result:
            parts.append(f"#   resultado esperado: {scenario.expected_result}")
        parts.append("")

    parts.append(f"poc: {poc}")
    parts.append(f"scenario: {scenario.id if scenario else ''}")
    parts.append(f"repetition: {repetition}")
    parts.append(f"call_sid: {call_sid}")
    parts.append(f"script: {script_version}")
    parts.append("")

    # The script turns set the order; the STT groups are placed in order on the
    # turns that do have a spoken line. A silence turn does not consume a
    # group: if the STT transcribed something during the silence, that text
    # falls into the following turn or into the leftovers, and either way it
    # is visible.
    group_index = 0
    if script_turns:
        for turn in script_turns:
            if turn.is_silence:
                hypothesis = ""
            else:
                hypothesis = groups[group_index] if group_index < len(groups) else ""
                if group_index < len(groups):
                    group_index += 1
            parts.extend(_turn_block(turn, hypothesis, turn_id=turn.id))
            parts.append("")
    else:
        for i, group in enumerate(groups):
            parts.extend(_turn_block(None, group, turn_id=f"T{i + 1}"))
            parts.append("")
        group_index = len(groups)

    extras = groups[group_index:]
    if extras:
        parts.append(LEFTOVERS_HEADER)
        parts.append(
            "#   El STT produjo más turnos que el guion: improvisaste, o partió "
            "un turno en dos."
        )
        parts.append("#   Reparte estas líneas a mano en los turnos de arriba, o")
        parts.append("#   escribe aquí lo que dijiste — cuenta igual para el WER.")
        for i, group in enumerate(extras):
            parts.extend(
                _turn_block(None, group, turn_id=f"extra{i + 1}")
            )
            parts.append("")

    parts.append(NOTES_HEADER)
    parts.append("#   naturalidad y turn_taking: 1-5, rúbrica en el guion.")
    parts.append("#   Los dos conteos son opcionales; vacío = no anotado.")
    for campo in SCORE_FIELDS:
        parts.append(f"{campo}: ")
    for campo in COUNT_FIELDS:
        parts.append(f"{campo}: ")
    parts.append("")
    parts.append(FREE_HEADER)
    parts.append("# Lo que quieras, en varias líneas. No se parsea.")
    parts.append("")

    return "\n".join(parts) + "\n"


def path_to_reference(folder: Path | str, call_sid: str) -> Path:
    """Where a call's reference file lives.

    Next to the call's JSON, in `grabaciones/`, which is gitignored: the
    transcript is transcribed speech and must not enter the repo (LFPDPPP).
    """
    return Path(folder) / f"reference_{call_sid or 'no_sid'}.txt"


def write(
    content: str,
    ruta: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the file. Does not overwrite corrections already made without permission.

    It is the safeguard that really matters in this workflow: accidentally
    regenerating an already corrected file would erase the 3 minutes of manual
    work the file exists to capture.
    """
    destination = Path(ruta)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass overwrite=True to replace it "
            "(you would lose the hand corrections)"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _to_int(raw_value: str) -> int | None:
    try:
        return int(str(raw_value).strip())
    except (TypeError, ValueError):
        return None


def parse(text: str, *, ruta: Path | None = None) -> ReferenceFile:
    """Parse the content of a reference file.

    Tolerant on purpose: any line that is not `yo:`, `stt:`, a `##` header or a
    known `key: value` is ignored. That way the person can annotate whatever
    they want without breaking the computation.
    """
    file = ReferenceFile(ruta=ruta)
    current_turn: ReferenceTurn | None = None
    in_notes = False
    in_free = False
    free: list[str] = []

    def close_turn() -> None:
        nonlocal current_turn
        if current_turn is not None and not current_turn.empty:
            file.turns.append(current_turn)
        current_turn = None

    for raw_line in text.splitlines():
        linea = raw_line.strip()

        if in_free:
            # Everything below the free-notes header is free text, except the
            # template comments.
            if linea and not linea.startswith("#"):
                free.append(linea)
            continue

        # Order matters: a `##` header also starts with `#`, so if comment
        # discarding comes first it swallows every header, the turns collapse
        # into one and the notes section never opens.
        if linea.startswith("##"):
            title = linea.lstrip("#").strip()
            if linea.startswith(FREE_HEADER):
                close_turn()
                in_free = True
                in_notes = False
                continue
            if linea.startswith(NOTES_HEADER):
                close_turn()
                in_notes = True
                continue
            close_turn()
            in_notes = False
            turn_id = title.split("—")[0].strip() or f"T{len(file.turns) + 1}"
            current_turn = ReferenceTurn(
                id=turn_id,
                when=title.split("—", 1)[1].strip() if "—" in title else "",
            )
            continue

        if not linea or linea.startswith("#"):
            continue

        low = linea.lower()

        if low.startswith(REFERENCE_PREFIX):
            valor = linea[len(REFERENCE_PREFIX) :].strip()
            if current_turn is None:
                current_turn = ReferenceTurn(id=f"T{len(file.turns) + 1}")
            # Several `yo:` lines in one turn are concatenated: it is the
            # natural way to add an improvisation without reordering the file.
            current_turn.reference = (
                f"{current_turn.reference} {valor}".strip()
                if current_turn.reference
                else valor
            )
            continue

        if low.startswith(HYPOTHESIS_PREFIX):
            valor = linea[len(HYPOTHESIS_PREFIX) :].strip()
            # The visual fragment separator is undone: it was for reading, not
            # for measuring. The spaces it leaves when removed are collapsed.
            valor = " ".join(
                valor.replace(FRAGMENT_SEPARATOR.strip(), " ").split()
            )
            if current_turn is None:
                current_turn = ReferenceTurn(id=f"T{len(file.turns) + 1}")
            current_turn.hypothesis = (
                f"{current_turn.hypothesis} {valor}".strip()
                if current_turn.hypothesis
                else valor
            )
            continue

        paired = _CLAVE_VALOR.match(linea)
        if paired:
            clave, valor = paired.group(1).lower(), paired.group(2).strip()
            if in_notes and clave in (*SCORE_FIELDS, *COUNT_FIELDS):
                setattr(file.notes, clave, _to_int(valor))
            elif clave in METADATA_FIELDS:
                file.metadata[clave] = valor

    close_turn()
    file.notes.free = "\n".join(free).strip()
    return file


def leer(ruta: Path | str) -> ReferenceFile:
    """Read and parse a reference file from disk."""
    destination = Path(ruta)
    return parse(destination.read_text(encoding="utf-8"), ruta=destination)


def search(folder: Path | str, call_sid: str) -> Path | None:
    """A call's reference file, if it already exists."""
    candidate = path_to_reference(folder, call_sid)
    return candidate if candidate.is_file() else None


def validate(file: ReferenceFile) -> list[str]:
    """Warnings about a reference file. Empty = ready to compute.

    The warning that matters most is the literal-copy one: it is the only error
    in this workflow that produces a nice, false number instead of a visible
    error.
    """
    warning_list: list[str] = []

    if not file.reference_text.strip():
        warning_list.append(
            "sin líneas `yo:` con texto: el archivo no se corrigió, no hay WER"
        )
        return warning_list

    absent = file.notes.absent()
    if absent:
        warning_list.append(f"faltan las puntuaciones manuales: {', '.join(absent)}")

    for campo in SCORE_FIELDS:
        valor = getattr(file.notes, campo)
        if valor is not None and not 1 <= valor <= 5:
            warning_list.append(f"{campo}={valor} está fuera del rango 1-5 de la rúbrica")

    identical = [
        t.id
        for t in file.turns
        if t.reference.strip()
        and t.hypothesis.strip()
        and t.reference.strip() == t.hypothesis.strip()
    ]
    if identical:
        warning_list.append(
            f"{len(identical)} turno(s) con `yo:` idéntico al `stt:` "
            f"({', '.join(identical[:5])}): revisa que no se haya copiado el STT "
            "en vez de transcribir el audio"
        )

    return warning_list


def turns_with_text(
    turns: Iterable[ReferenceTurn],
) -> list[ReferenceTurn]:
    """Turns that contribute something to the computation."""
    return [t for t in turns if t.reference.strip() or t.hypothesis.strip()]
