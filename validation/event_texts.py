"""What the agent said in each turn with an interruption event, for reading.

The harness metrics say **when** the agent went quiet and **when** it spoke
again, not **what** it said when it came back. This listing gives, per event, the
carrier's line, the clip played over the agent ("ajá", "espéreme…") and
everything the agent said in that turn, together with what the harness measured.
That way one can see without listening whether, after a backchannel, the agent
continued the same sentence or generated a new message (repeats, rephrases or
answers the "ajá").

**The agent's text** is split into before and after the event in the campaigns
run since 6 October 2026; in earlier ones it covers the whole turn, because the
log did not store the time of each word. In PoC1 and PoC2 it is what was
synthesized (it may run a few words ahead of what had actually been played before
the cut); in PoC3 it is the transcript written by the model itself. For the fine
detail, the audio cut of the same event (`validation.interruption_cuts`) has the
same name as given here.

    uv run --project poc3_gpt_live python -m validation.event_texts \\
        --campana diagnostico/lote/interrupciones_completa --include-failed \\
        --output diagnostico/recortes_interrupciones_completa/textos.md

Rates nothing: it is reading material.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

DEFAULT_KINDS: Final[tuple[str, ...]] = ("apoyo", "interrupcion")


def _ms(x: Any) -> str:
    return "—" if x in (None, "") else f"{float(x):,.0f} ms"


def what_it_did(ev: dict[str, Any]) -> str:
    """What the harness measured, in one line."""
    if not ev.get("fired"):
        return "no se disparó"
    if ev.get("stopped") is None:
        return "no hablaba en la referencia"
    if not ev["stopped"]:
        return "siguió hablando (no se calló)"
    text = f"se calló a los {_ms(ev.get('stop_time_ms'))}"
    if ev.get("endless_pause"):
        return text + f"; no volvió a hablar antes de la línea siguiente (silencio ≥ {_ms(ev.get('pause_duration_ms'))})"
    if ev.get("speaks_again_ms") is not None:
        return text + (f"; pausa de {_ms(ev.get('pause_duration_ms'))}, volvió a hablar"
                        f" {_ms(ev.get('speaks_again_ms'))} después del clip")
    return text


def entries_of(campana: Path, *, include_failed: bool, kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    valid_items = ("ok", "roja") if include_failed else ("ok",)
    output: list[dict[str, Any]] = []
    for linea in (campana / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        p = json.loads(linea)
        if p.get("estado") not in valid_items:
            continue
        ruta = campana / "summaries" / f"{p['clave']}.json"
        if not ruta.is_file():
            continue
        summary = json.loads(ruta.read_text(encoding="utf-8"))
        event_log = Path((summary.get("artifacts") or {}).get("event_log") or "")
        if not event_log.is_file():
            continue
        for t in json.loads(event_log.read_text(encoding="utf-8")).get("turns") or []:
            ev = t.get("event")
            if not ev or ev.get("tipo") not in kinds:
                continue
            output.append({
                "poc": p.get("poc"), "estado": p.get("estado"), "clave": p["clave"],
                "turn": t.get("turn"), "tipo": ev["tipo"],
                "cut": f"poc{p.get('poc')}/{p['clave']}_{t.get('turn')}_{ev['tipo']}.wav",
                "linea": t.get("played_line") or "",
                "clip": ev.get("text") or "",
                "agent": (t.get("agent") or "").strip(),
                # Since 6 October 2026 the harness splits the text at the event.
                "antes": ev.get("agent_before_event"),
                "after": ev.get("agent_after_event"),
                "what_it_did": what_it_did(ev),
            })
    return output


def a_markdown(entries: list[dict[str, Any]]) -> str:
    parts = ["# Lo que dijo el agente en cada turno con evento", "",
              "El texto del agente es de todo el turno (antes y después del evento). "
              "El recorte de audio indicado tiene el evento en el segundo 2.", ""]
    for poc in sorted({str(e["poc"]) for e in entries}):
        for tipo in DEFAULT_KINDS + ("control",):
            of = [e for e in entries if str(e["poc"]) == poc and e["tipo"] == tipo]
            if not of:
                continue
            parts += [f"## PoC{poc} · {tipo} ({len(of)})", ""]
            for e in of:
                failed = " · llamada roja" if e["estado"] == "roja" else ""
                parts += [
                    f"### {e['clave']} {e['turn']}{failed}",
                    f"- Recorte: `{e['cut']}`",
                    f"- Arnés: {e['what_it_did']}",
                    f"- Transportista: {e['linea'] or '(silencio)'}",
                    f"- Encima del agente: {e['clip'] or '(nada: control)'}",
                ]
                if e.get("antes") is not None or e.get("after") is not None:
                    parts += [
                        f"- Agente antes del evento: {(e.get('antes') or '').strip() or '(nada)'}",
                        f"- Agente después del evento: {(e.get('after') or '').strip() or '(nada)'}",
                        "",
                    ]
                else:
                    parts += [f"- Agente: {e['agent'] or '(sin texto)'}", ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--campana", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True, help="archivo .md")
    p.add_argument("--include-failed", action="store_true")
    p.add_argument("--kinds", default=",".join(DEFAULT_KINDS),
                   help="clases de evento, separadas por coma (apoyo,interrupcion,control)")
    args = p.parse_args(argv)
    kinds = tuple(t.strip() for t in args.kinds.split(",") if t.strip())
    entries: list[dict[str, Any]] = []
    for c in args.campana:
        entries.extend(entries_of(c, include_failed=args.include_failed, kinds=kinds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(a_markdown(entries), encoding="utf-8")
    print(f"{len(entries)} eventos escritos en {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
