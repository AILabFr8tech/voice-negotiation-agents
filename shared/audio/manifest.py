"""El manifiesto: qué clips hacen falta, cuáles hay, y si siguen siendo los mismos.

`*.wav` está ignorado en todo el repo (pesa y puede traer datos personales —
LFPDPPP), así que **los clips no se versionan y este archivo sí**. Es la única
evidencia commiteable de qué audio reprodujeron las tres PoC, y es lo que
permite afirmar en el reporte que la comparación no cambió de entrada entre
arquitecturas. Sin él, esa afirmación sería un recuerdo.

Cuatro cosas se verifican, y todas han fallado alguna vez en proyectos así:

1. **Que estén todos.** El inventario sale del guion, no de una lista a mano:
   49 clips hoy (5 escenarios × 8 turnos + 9 alternativas).
2. **Que el audio no haya cambiado.** Re-hashea los artefactos en disco.
3. **Que el guion no se haya movido debajo de los clips.** Si alguien edita una
   línea en `escenarios.yaml`, el clip grabado deja de decir lo que el guion
   dice, y el WER se calcularía contra una referencia que nadie pronunció. Esa
   deriva es silenciosa por naturaleza; aquí es un error con nombre.
4. **Que el nivel siga siendo el vigente.** El objetivo de calibración se
   re-mide cuando haya grabaciones reales de PoC1 (ver `nivel.py`); un clip
   preparado contra el objetivo viejo suena distinto que uno nuevo, y mezclarlos
   haría que el WER dependiera de cuándo se grabó cada línea.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from shared.audio.level import TOLERANCE_DB, Calibration, load_calibration
from shared.metrics.scenarios import Script, load_script

MANIFEST_VERSION: Final[str] = "1"

MANIFEST_NAME: Final[str] = "manifest.json"

# Etiqueta de la variante base de un turno. Las alternativas del guion se
# numeran desde 1 en el orden en que aparecen en el YAML.
BASE_VARIANT: Final[str] = "base"


@dataclass(frozen=True)
class ExpectedClip:
    """Una línea del guion que hay que grabar, con el nombre que debe llevar."""

    id: str
    scenario: int
    turn: str
    variant: str
    text: str

    @property
    def wav_name(self) -> str:
        return f"{self.id}.wav"

    @property
    def is_alternative(self) -> bool:
        return self.variant != BASE_VARIANT


def expected_inventory(script: Script | None = None) -> tuple[ExpectedClip, ...]:
    """Todos los clips que el guion pide, en orden de escenario y turno.

    Incluye las `alternativas`: son las ramas del árbol de decisión de los
    escenarios 2, 4 y 5, y si no están grabadas el arnés se queda sin qué
    reproducir justo cuando el agente toma el camino menos esperado.
    """
    script = script or load_script()
    expected_items: list[ExpectedClip] = []
    for scenario in script.scenarios:
        for turn in scenario.turns:
            if turn.is_silence:
                # Un turno de silencio no se graba: el arnés lo produce con
                # ruido de confort, que es más fiel que un WAV de silencio.
                continue
            base = f"e{scenario.id}_{turn.id.lower()}"
            expected_items.append(
                ExpectedClip(
                    id=base,
                    scenario=scenario.id,
                    turn=turn.id,
                    variant=BASE_VARIANT,
                    text=turn.linea,
                )
            )
            for numero, alternate in enumerate(turn.alternates, start=1):
                expected_items.append(
                    ExpectedClip(
                        id=f"{base}_alt{numero}",
                        scenario=scenario.id,
                        turn=turn.id,
                        variant=f"alt{numero}",
                        text=alternate,
                    )
                )
    return tuple(expected_items)


@dataclass(frozen=True)
class ManifestEntry:
    """Un clip ya preparado, con su identidad criptográfica."""

    id: str
    scenario: int
    turn: str
    variant: str
    text: str
    # --- de dónde salió ---
    source: str
    source_sha256: str
    origin_sample_rate: int
    # --- lo que se reproduce ---
    sha256_pcm8k: str
    sha256_ulaw: str
    duration_s: float
    samples: int
    pico_dbfs: float
    rms_dbfs: float
    # --- cómo se preparó ---
    resampling_engine: str
    start_cut_ms: float
    end_cut_ms: float
    trim_threshold_dbfs: float
    # --- los dos ceros del turno ---
    # `fin_de_voz_seg` es dónde deja de haber voz dentro del clip; el fin del
    # archivo es `duracion_seg`. Los dos hacen falta y significan cosas
    # distintas: el del archivo es un hecho físico (esos bytes salieron), el de
    # voz es una medición que depende del umbral — que por eso va al lado y no
    # se puede separar del número. Vive aquí, en el manifiesto, y no en el arnés
    # de una PoC: es una propiedad del clip, y las 3 reproducen el mismo clip.
    end_of_speech_s: float
    cola_seg: float
    speech_drop_db: float
    speech_threshold_dbfs: float
    # --- calibración de nivel (ver nivel.py) ---
    target_dbfs: float
    # El criterio con el que se midió ese objetivo. Va junto porque el mismo
    # audio mide hasta 9 dB distinto según qué ventanas cuenten como voz: un
    # objetivo sin su caída no es reproducible.
    drop_db: float
    speech_rms_dbfs: float
    source_speech_rms_dbfs: float
    gain_db: float
    provisional_calibration: bool
    calibration_source: str
    warning_list: tuple[str, ...] = ()

    @property
    def deviation_db(self) -> float:
        return round(self.speech_rms_dbfs - self.target_dbfs, 2)

    def as_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["warning_list"] = list(self.warning_list)
        return data


@dataclass
class Manifest:
    """El inventario preparado, con el contexto para poder auditarlo después."""

    generated: str = ""
    version: str = MANIFEST_VERSION
    script_version: str = ""
    entries: list[ManifestEntry] = field(default_factory=list)

    def by_id(self) -> dict[str, ManifestEntry]:
        return {entry.id: entry for entry in self.entries}

    def as_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated": self.generated or datetime.now(timezone.utc).isoformat(),
            "script_version": self.script_version,
            "clips": [entry.as_json() for entry in self.entries],
        }

    def write(self, ruta: Path | str) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps(self.as_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ruta


def load_manifest(ruta: Path | str) -> Manifest:
    """Lee un manifiesto escrito por `escribir`. Falla claro si está incompleto."""
    data = json.loads(Path(ruta).read_text(encoding="utf-8"))
    entries = []
    for raw_value in data.get("clips", []):
        fields = dict(raw_value)
        fields["warning_list"] = tuple(fields.get("warning_list") or ())
        try:
            entries.append(ManifestEntry(**fields))
        except TypeError as error:
            raise ValueError(
                f"el manifiesto {Path(ruta).name} tiene un clip cuyos campos no "
                f"cuadran con los de este código ({error}). ¿Lo escribió otra "
                "versión? Vuelve a correr `cli preparar`"
            ) from error
    return Manifest(
        generated=str(data.get("generated", "")),
        version=str(data.get("version", "")),
        script_version=str(data.get("script_version", "")),
        entries=entries,
    )


@dataclass(frozen=True)
class Problem:
    """Algo que impide usar los clips tal como están."""

    clip: str
    tipo: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.tipo}] {self.clip}: {self.detail}"


def _sha256_of(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def verify(
    manifest: Manifest,
    folder: Path | str,
    *,
    script: Script | None = None,
    calibration: Calibration | None = None,
    require_complete: bool = True,
) -> list[Problem]:
    """Compara manifiesto, disco, guion y calibración. Lista vacía = todo cuadra.

    `require_complete=False` sirve para revisar una grabación a medias sin que
    los 40 clips que faltan tapen el problema real del que sí está.
    """
    folder = Path(folder)
    calibration = calibration or load_calibration()
    problems: list[Problem] = []
    entries = manifest.by_id()
    expected_items = {clip.id: clip for clip in expected_inventory(script)}

    for clip_id, expected in expected_items.items():
        entry = entries.get(clip_id)
        if entry is None:
            if require_complete:
                problems.append(
                    Problem(clip_id, "missing", f"sin grabar — «{expected.text}»")
                )
            continue
        if entry.text.strip() != expected.text.strip():
            problems.append(
                Problem(
                    clip_id,
                    "script_drift",
                    "el guion cambió desde que se grabó el clip.\n"
                    f"        clip:  «{entry.text}»\n"
                    f"        script: «{expected.text}»",
                )
            )

    for clip_id in entries:
        if clip_id not in expected_items:
            problems.append(
                Problem(clip_id, "sobra", "no corresponde a ningún turno del guion")
            )

    for clip_id, entry in entries.items():
        # El objetivo de nivel cambia cuando se re-mide contra una grabación
        # real (ver nivel.py). Un clip preparado contra el objetivo viejo suena
        # distinto que uno nuevo, y mezclarlos haría que el WER dependiera de
        # cuándo se grabó cada línea. Hay que regenerar, no seguir.
        if (
            abs(entry.target_dbfs - calibration.target_dbfs) > 0.01
            or abs(entry.drop_db - calibration.drop_db) > 0.01
        ):
            problems.append(
                Problem(
                    clip_id,
                    "different_target",
                    f"se preparó contra {entry.target_dbfs:.1f} dBFS "
                    f"(caída {entry.drop_db:.0f} dB) y la calibración vigente "
                    f"pide {calibration.target_dbfs:.1f} dBFS "
                    f"(caída {calibration.drop_db:.0f} dB). "
                    "Vuelve a correr `cli preparar`",
                )
            )
        elif abs(entry.deviation_db) > TOLERANCE_DB:
            problems.append(
                Problem(
                    clip_id,
                    "level_out_of_tolerance",
                    f"voz a {entry.speech_rms_dbfs:.1f} dBFS, {entry.deviation_db:+.1f} "
                    f"dB del objetivo (tolerancia ±{TOLERANCE_DB:.0f} dB)",
                )
            )

        ruta_wav = folder / f"{clip_id}.wav"
        ruta_ulaw = folder / f"{clip_id}.ulaw"
        if not ruta_wav.exists():
            problems.append(Problem(clip_id, "no_file", f"no está {ruta_wav.name}"))
        if not ruta_ulaw.exists():
            problems.append(Problem(clip_id, "no_file", f"no está {ruta_ulaw.name}"))
            continue
        # Se re-hashea el `.ulaw` y no el `.wav`: el WAV lleva cabecera, y dos
        # escrituras del mismo audio con cabeceras distintas darían hashes
        # distintos sin que el audio haya cambiado. El `.ulaw` son los bytes
        # desnudos, y el PCM se deriva de ellos por construcción (`clips.py`).
        actual = _sha256_of(ruta_ulaw)
        if actual != entry.sha256_ulaw:
            problems.append(
                Problem(
                    clip_id,
                    "hash_differs",
                    f"el audio en disco no es el del manifiesto "
                    f"({actual[:12]}… vs {entry.sha256_ulaw[:12]}…)",
                )
            )

    return problems
