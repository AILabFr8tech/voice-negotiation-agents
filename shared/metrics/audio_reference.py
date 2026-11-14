"""The WER reference corrections that came from the audio.

The reference that comes from the **script text** (`clips_reference.generate`
takes `linea_reproducida` from the log) cannot be validated against the
manifest's `texto` field, which is what `verify_against_manifest` does: **both
are the same copied script**, and neither looks at the audio. What was really
said in each clip came from transcribing the audio, and whatever differs from
the script is corrected here.

The full verification is in `docs/respaldo_reference_de_audio.md`. This module
is the part that had to live in the repo: **the corrections as data**, so that
the WER can be recomputed without repeating it.

## Why the corrections belong to the clip and not to the run

The stimulus is the same `.ulaw` in all three PoCs, so if a word is not in the
audio, it is not there for any of them. Correcting by `clip_id` is what keeps
the correction **symmetric by construction**: there is no way to fix the
reference for one PoC and not for the others.

## The three tests, and which one rules

1. **Geometric** — the published clip is a contiguous slice of the raw
   recording (verified on all 49 to ±0.2 ms), so the question is whether the
   last word, timestamped by Whisper in the raw recording, falls inside the
   window the published clip keeps.
2. **Agreement across the three PoCs** — if all three engines miss the same
   final word, that is a sign it is not there.
3. **Human listening**, with the admission rule fixed before listening.

**The ear rules, and `e2_t3` is the reason it is written that way.** See
`CORRECTIONS["e2_t3"]`: it is the only clip where the geometry said the word
fit and the ear said it did not. The ear won, and that makes the geometric test
necessary but **not sufficient** — its false negative was caught by the
agreement test, which was the strongest signal in the bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping


@dataclass(frozen=True)
class Correction:
    """A reference line the script asked for and the audio does not say."""

    clip_id: str
    script: str
    audio: str
    motivo: str
    test: str
    """Which of the three tests decided it. See the module docstring."""


CORRECTIONS: Final[Mapping[str, Correction]] = {
    c.clip_id: c
    for c in (
        Correction(
            clip_id="e5_t7",
            script="Órale, pues vaya que sí. Quedamos en 18,500 con todo incluido.",
            audio="Órale, pues vaya que sí. Quedamos en 18,500.",
            motivo="el recorte se llevó «con todo incluido»",
            test="geometría: «incluido» acaba 1,140 ms pasado el fin del publicado; «con» a −200",
        ),
        Correction(
            clip_id="e3_t6",
            script="Bueno… le bajo a veinticinco mil, pero de ahí no me muevo.",
            audio="Bueno… le bajo a veinticinco mil, pero de ahí…",
            motivo="el recorte se llevó «no me muevo»",
            test="geometría: «no» a −10 ms, «me» a −170, «muevo» a −510",
        ),
        Correction(
            clip_id="e2_t6",
            script=(
                "Ándele, es una buena ruta y se la hago rápida. "
                "Diecisiete… no, dieciocho mil."
            ),
            audio=(
                "Ándele, es una buena ruta y se la hago rápida. "
                "Diecisiete… no, dieciocho."
            ),
            motivo="el recorte se llevó «mil»: el monto cambia de 18,000 a 18",
            test="geometría: «mil» a −210 ms; «18» cabe con +190",
        ),
        Correction(
            clip_id="e2_t5_alt1",
            script=(
                "Está bien lo de la compensación, pero el número "
                "tiene que ser dieciocho mil."
            ),
            audio=(
                "Está bien lo de la compensación, pero el número "
                "tiene que ser dieciocho."
            ),
            motivo="el recorte se llevó «mil»: el monto cambia de 18,000 a 18",
            test="geometría: «.000» a −270 ms; «18» cabe con +130",
        ),
        Correction(
            clip_id="e1_t3",
            script="Ajá. ¿Y para cuándo la necesitan en Laredo?",
            audio="Ajá. ¿Y para cuándo la necesita en Laredo?",
            motivo="desvío al actuar, no recorte. El único de los 49",
            test="lo confirmó quien grabó el clip",
        ),
        # --- the sixth, from 23/09/2026 -------------------------------------
        Correction(
            clip_id="e2_t3",
            script="¿A cómo andan pagando esa ruta ustedes normalmente?",
            audio="¿A cómo andan pagando esa ruta ustedes?",
            motivo="el recorte se llevó «normalmente»",
            test=(
                "escucha directa de quien grabó el clip. **La geometría dio "
                "margen POSITIVO y se equivocó**: es el único falso negativo de "
                "esa prueba en los 49, y lo atrapó la concordancia — la última "
                "palabra falla en el 100% de PoC1, el 100% de PoC2 y el 71% de "
                "PoC3, la señal más fuerte del banco"
            ),
        ),
    )
}


def clip_reference(clip_id: str, script_line: str) -> str:
    """The reference used for the WER: the corrected one if there is one.

    `script_line` is `linea_reproducida` from the log. It is passed in whole
    instead of being read from here because scenarios with branches play
    variants the base script does not contain, and the log is what knows which
    one was played.
    """
    correction = CORRECTIONS.get(clip_id)
    return correction.audio if correction is not None else script_line


def corrected_clips() -> frozenset[str]:
    return frozenset(CORRECTIONS)
