"""Simulated carrier: acts out the 5 scenarios against each PoC's agent.

It is the harness without its transport. It plays the pre-built clips
(`shared/audio/`) turn by turn over a continuous comfort-noise floor,
observes the agent's speech and text, decides whether the run is usable, and
writes down what was played, what happened and what it cost.

What is **not** here, because it belongs to each PoC: how the audio is published
(the room's `AudioSource` in PoC2, the Media Streams WebSocket in PoC1 and PoC3),
how the transcript is read, and how the turn's real zero is published to the
worker. Each harness inherits from `Player` and `Observed` and fills those gaps.

Why it lives in `shared/` and not in each PoC: the rule for measurable turns,
the clip zero and the log format all shape how the metrics are read.
Three copies would diverge, and the comparison would stop comparing.
"""

from __future__ import annotations

from shared.carrier_simulator.verbal_backchannel import (
    BACKCHANNELS,
    MAX_WORDS,
    is_backchannel,
)
from shared.carrier_simulator.audio import (
    MS_PER_FRAME,
    COMFORT_NOISE_PEAK,
    split_into_frames,
    rms_pcm16,
    comfort_noise_frame,
)
from shared.carrier_simulator.auditoria import (
    AGENT_CHANNEL,
    HARNESS_CHANNEL,
    AUDIT_PREFIX,
    AuditRecorder,
    UnauditableSampleRate,
    output_sample_rate,
)
from shared.carrier_simulator.cero import (
    ZERO_PROTOCOL,
    MARK_KIND,
    SUMMARY_KIND,
    PROTOCOL_VERSION,
    ZeroEmitter,
    ZeroMark,
    Envelope,
    encode_mark,
    encode_summary,
    decode,
)
from shared.carrier_simulator.cruce import (
    LAST_LINE_TAIL_S,
    PAIRING_TOLERANCE_S,
    EstimatedZero,
    Line,
    PairedTurn,
    group,
    pair,
    lines_summary,
)
from shared.carrier_simulator.run import (
    END_AGENT_MUTE,
    END_CLOSED_BY_AGENT,
    END_COMPLETE,
    END_INCOMPLETE,
    END_TIMEOUT,
    MIN_VALID_TURNS,
    RunResult,
    evaluate_run,
)
from shared.carrier_simulator.speech_onset import (
    TurnStart,
    compute as start_latency,
    rows_per_turn,
)
from shared.carrier_simulator.script import (
    RUTA_CLIPS_DEFAULT,
    HarnessFailure,
    ScriptStep,
    load_clip,
    load_scenario,
    negotiation_state,
)
from shared.carrier_simulator.observed import (
    STREAK_HANGOVER_S,
    RMS_WITH_SPEECH,
    TextFragment,
    Observed,
    SpeechStreak,
    ObservedTurn,
    text_from_deltas,
)
from shared.carrier_simulator.report import (
    CostContext,
    ReportContext,
    Tariffs,
    event_log_de_la_corrida,
    print_cost_warning,
    print_report,
    audio_frame_line,
    published_frame_line,
)
from shared.carrier_simulator.summary import (
    VERSION as SUMMARY_VERSION,
    write_summary,
    measure_return_path,
    measure_overlap,
    quote_summary,
    call_summary,
    run_wer,
)
from shared.carrier_simulator.pacing import FramePacing
from shared.carrier_simulator.player import (
    MAX_AGENT_WAIT_S,
    WAIT_AFTER_CLIP_S,
    MIN_SUBSTANTIVE_SPEECH_S,
    AGENT_SILENCE_S,
    BACKCHANNEL_CAP_S,
    AgentWait,
    Player,
    wait_until_agent_is_silent,
    wait_for_agent_turn,
    play_script,
)

__all__ = [
    "BACKCHANNELS",
    "event_log_de_la_corrida",
    "AGENT_CHANNEL",
    "HARNESS_CHANNEL",
    "load_clip",
    "load_scenario",
    "CostContext",
    "ReportContext",
    "is_backchannel",
    "MAX_AGENT_WAIT_S",
    "WAIT_AFTER_CLIP_S",
    "AgentWait",
    "wait_until_agent_is_silent",
    "wait_for_agent_turn",
    "negotiation_state",
    "evaluate_run",
    "HarnessFailure",
    "END_AGENT_MUTE",
    "END_CLOSED_BY_AGENT",
    "END_COMPLETE",
    "END_INCOMPLETE",
    "END_TIMEOUT",
    "TextFragment",
    "AuditRecorder",
    "STREAK_HANGOVER_S",
    "print_cost_warning",
    "TurnStart",
    "start_latency",
    "rows_per_turn",
    "print_report",
    "audio_frame_line",
    "published_frame_line",
    "MAX_WORDS",
    "MIN_VALID_TURNS",
    "MIN_SUBSTANTIVE_SPEECH_S",
    "MS_PER_FRAME",
    "Observed",
    "split_into_frames",
    "ScriptStep",
    "COMFORT_NOISE_PEAK",
    "AUDIT_PREFIX",
    "ZERO_PROTOCOL",
    "MARK_KIND",
    "SUMMARY_KIND",
    "PROTOCOL_VERSION",
    "ZeroEmitter",
    "ZeroMark",
    "Sobre",
    "EstimatedZero",
    "PAIRING_TOLERANCE_S",
    "PairedTurn",
    "pair",
    "LAST_LINE_TAIL_S",
    "Line",
    "group",
    "lines_summary",
    "encode_mark",
    "encode_summary",
    "decode",
    "SpeechStreak",
    "FramePacing",
    "play_script",
    "Player",
    "RunResult",
    "write_summary",
    "measure_return_path",
    "measure_overlap",
    "quote_summary",
    "call_summary",
    "SUMMARY_VERSION",
    "run_wer",
    "RMS_WITH_SPEECH",
    "rms_pcm16",
    "RUTA_CLIPS_DEFAULT",
    "AGENT_SILENCE_S",
    "Tariffs",
    "output_sample_rate",
    "UnauditableSampleRate",
    "text_from_deltas",
    "BACKCHANNEL_CAP_S",
    "comfort_noise_frame",
    "ObservedTurn",
]
