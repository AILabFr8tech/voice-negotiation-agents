"""Preparación de los clips del escenario de interrupciones.

Los mismos pasos que los 49 clips del guion (`shared.audio.cli preparar`): el
mismo recorte de silencio, la misma calibración de nivel, la misma bajada a
8 kHz y μ-law. Cambian dos cosas, y por eso vive aparte:

- **La lista sale de `shared/test_scenarios/interrupciones.yaml`**, no del guion.
  El inventario de `shared.audio.cli` es el de los 5 escenarios, y meter aquí
  estos clips cambiaría su manifiesto y sus huellas, que son la evidencia de
  que las tres PoC reprodujeron el mismo estímulo.
- **El manifiesto guarda también el inicio de la voz** (`speech_onset_s`).
  En un clip del guion importa dónde termina la voz, porque de ahí sale el cero
  de la latencia. En una interrupción importa dónde empieza: el tiempo que
  tarda el agente en callarse se cuenta desde que el transportista empezó a
  hablarle encima. Se mide con el mismo umbral que el fin de voz.

Uso, desde la raíz del repo:

    python -m shared.audio.interrupciones preparar
    python -m shared.audio.interrupciones verificar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml

from shared.audio.clips import EmptyClip, write_artifacts, prepare_clip
from shared.audio.level import load_calibration, windows_with_speech
from shared.audio.resampling import TELEPHONY_SAMPLE_RATE
from shared.audio.wav import AudioNotSupported

BASE_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "test_scenarios"
CATALOG_PATH: Final[Path] = BASE_PATH / "interrupciones.yaml"
DEFAULT_RAW_PATH: Final[Path] = BASE_PATH / "raw_interruption_clips"
RUTA_CLIPS_DEFAULT: Final[Path] = BASE_PATH / "interruption_clips"
MANIFEST_NAME: Final[str] = "manifest.json"

KINDS: Final[tuple[str, ...]] = ("interrupcion", "apoyo")


@dataclass(frozen=True)
class InterruptionClip:
    """Una línea del catálogo: qué dice y qué se espera del agente."""

    id: str
    tipo: str
    text: str


@dataclass(frozen=True)
class InterruptionEntry:
    """Lo que el manifiesto guarda de cada clip preparado."""

    id: str
    tipo: str
    text: str
    source: str
    source_sha256: str
    sha256_ulaw: str
    origin_sample_rate: int
    duration_s: float
    speech_onset_s: float
    end_of_speech_s: float
    speech_drop_db: float
    speech_threshold_dbfs: float
    pico_dbfs: float
    speech_rms_dbfs: float
    gain_db: float
    start_cut_ms: float
    end_cut_ms: float
    provisional_calibration: bool
    warning_list: tuple[str, ...]


def load_catalog(ruta: Path = CATALOG_PATH) -> list[InterruptionClip]:
    data = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    clips = [InterruptionClip(**c) for c in data["clips"]]
    ids = [c.id for c in clips]
    if len(set(ids)) != len(ids):
        raise ValueError("el catálogo de interrupciones repite un id")
    for c in clips:
        if c.tipo not in KINDS:
            raise ValueError(f"{c.id}: tipo {c.tipo!r}; los válidos son {KINDS}")
    return clips


def speech_onset(pcm8k: Any, *, umbral_dbfs: float) -> float:
    """Segundo del clip en que empieza la voz, con el umbral del fin de voz."""
    with_voice, per_window, _ = windows_with_speech(pcm8k, TELEPHONY_SAMPLE_RATE, umbral_dbfs=umbral_dbfs)
    if with_voice.size == 0:
        return 0.0
    return round(int(with_voice[0]) * per_window / TELEPHONY_SAMPLE_RATE, 4)


def prepare(
    origin: Path = DEFAULT_RAW_PATH, destination: Path = RUTA_CLIPS_DEFAULT
) -> tuple[list[InterruptionEntry], list[str]]:
    """Degrada los clips del catálogo y escribe su manifiesto. `(entradas, avisos)`."""
    calibration = load_calibration()
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[InterruptionEntry] = []
    warning_list: list[str] = []
    for c in load_catalog():
        ruta = origin / f"{c.id}.wav"
        if not ruta.is_file():
            warning_list.append(f"{c.id}: falta {ruta.name}")
            continue
        try:
            clip = prepare_clip(ruta, calibration=calibration)
        except (AudioNotSupported, EmptyClip, ValueError) as error:
            warning_list.append(f"{c.id}: {error}")
            continue
        write_artifacts(clip, destination / c.id)
        own_ones = tuple(clip.warning_list())
        warning_list.extend(f"{c.id}: {a}" for a in own_ones)
        entries.append(InterruptionEntry(
            id=c.id, tipo=c.tipo, text=c.text,
            source=ruta.name,
            source_sha256=hashlib.sha256(ruta.read_bytes()).hexdigest(),
            sha256_ulaw=hashlib.sha256(clip.ulaw).hexdigest(),
            origin_sample_rate=clip.origin_sample_rate,
            duration_s=round(clip.duration_s, 3),
            speech_onset_s=speech_onset(clip.pcm8k, umbral_dbfs=clip.speech_threshold_dbfs),
            end_of_speech_s=clip.end_of_speech_s,
            speech_drop_db=clip.speech_drop_db,
            speech_threshold_dbfs=clip.speech_threshold_dbfs,
            pico_dbfs=round(clip.pico_dbfs, 1),
            speech_rms_dbfs=clip.speech_rms_dbfs,
            gain_db=clip.gain_db,
            start_cut_ms=clip.start_cut_ms,
            end_cut_ms=clip.end_cut_ms,
            provisional_calibration=clip.provisional_calibration,
            warning_list=own_ones,
        ))
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "catalog": CATALOG_PATH.name,
        "clips": [asdict(e) for e in entries],
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return entries, warning_list


def load_manifest(destination: Path = RUTA_CLIPS_DEFAULT) -> dict[str, dict[str, Any]]:
    """`{id: entrada}` del manifiesto preparado."""
    ruta = destination / MANIFEST_NAME
    if not ruta.is_file():
        raise FileNotFoundError(
            f"no hay manifiesto en {ruta}: corre `python -m shared.audio.interrupciones preparar`"
        )
    data = json.loads(ruta.read_text(encoding="utf-8"))
    return {e["id"]: e for e in data["clips"]}


def verify(destination: Path = RUTA_CLIPS_DEFAULT) -> list[str]:
    """Problemas entre el catálogo, el manifiesto y los `.ulaw` de disco."""
    problems: list[str] = []
    manifest = load_manifest(destination)
    for c in load_catalog():
        e = manifest.get(c.id)
        if e is None:
            problems.append(f"{c.id}: no está en el manifiesto")
            continue
        if e["text"] != c.text or e["tipo"] != c.tipo:
            problems.append(f"{c.id}: el catálogo cambió desde que se preparó")
        ruta = destination / f"{c.id}.ulaw"
        if not ruta.is_file():
            problems.append(f"{c.id}: falta {ruta.name}")
        elif hashlib.sha256(ruta.read_bytes()).hexdigest() != e["sha256_ulaw"]:
            problems.append(f"{c.id}: el .ulaw no coincide con su huella")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subs = p.add_subparsers(dest="command", required=True)
    pp = subs.add_parser("prepare", help="degrada los clips y escribe el manifiesto")
    pp.add_argument("--origin", type=Path, default=DEFAULT_RAW_PATH)
    pp.add_argument("--destination", type=Path, default=RUTA_CLIPS_DEFAULT)
    pv = subs.add_parser("verify", help="re-hashea contra el manifiesto")
    pv.add_argument("--destination", type=Path, default=RUTA_CLIPS_DEFAULT)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.command == "prepare":
        entries, warning_list = prepare(args.origin, args.destination)
        print(f"{'id':<8} {'tipo':<13} {'dura':>6} {'voz de':>7} {'a':>6}  texto")
        for e in entries:
            print(f"{e.id:<8} {e.tipo:<13} {e.duration_s:>5.2f}s {e.speech_onset_s:>6.2f}s "
                  f"{e.end_of_speech_s:>5.2f}s  {e.text}")
        missing_items = len(load_catalog()) - len(entries)
        print(f"\n{len(entries)} clip(s) preparados en {args.destination}"
              + (f" · faltan {missing_items}" if missing_items else ""))
        for a in warning_list:
            print(f"  [!] {a}")
        return 1 if missing_items else 0
    problems = verify(args.destination)
    for pr in problems:
        print(f"  [!] {pr}")
    print("todo coincide" if not problems else f"{len(problems)} problema(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
