# PoC2 — LiveKit Agents

Segunda de las 3 PoC de la comparativa: el mismo pipeline STT → LLM → TTS
sobre LiveKit Agents, con las 4 funciones de negociación byte por byte iguales a
las que ve el LLM en PoC1. Las mediciones del reporte se corrieron con el arnés
([`app/arnes.py`](app/arnes.py)), que corre la sala del agente con clips
pregrabados y sin telefonía.

## Qué existe

| Archivo | Qué hace |
|---|---|
| [`app/config.py`](app/config.py) | Lee el `.env` de la raíz. Valida por niveles y hace cumplir el aislamiento del despachador. |
| [`app/tools.py`](app/tools.py) | Las 4 funciones, con **el mismo schema JSON que PoC1**. |
| [`app/latencia.py`](app/latencia.py) | Captura de latencia sobre `ChatMessage.metrics`, más los dos diagnósticos. |
| [`app/fin_de_turno.py`](app/fin_de_turno.py) | Por qué terminó cada turno: por confianza del detector o por vencimiento. |
| [`app/marcas.py`](app/marcas.py) | El cero real de cada turno, del arnés al worker. |
| [`app/agente.py`](app/agente.py) | Servicios, `AgentSession`, worker y persistencia. |
| [`app/arnes.py`](app/arnes.py) | Arnés: corre la sala del agente con clips pregrabados, sin telefonía. |
| [`app/humo_sala.py`](app/humo_sala.py) | Una sesión real del agente sin telefonía, para validar el camino del arnés. |
| [`app/diagnostico_banda.py`](app/diagnostico_banda.py) · [`app/diagnostico_cero.py`](app/diagnostico_cero.py) · [`app/estadistica.py`](app/estadistica.py) | Diagnósticos de la banda telefónica y del cero, y sus pruebas estadísticas. |
| [`app/llamada.py`](app/llamada.py) | Dispatch + llamada saliente por el trunk. El único módulo que gasta telefonía. |
| [`app/smoke.py`](app/smoke.py) | Humo, niveles 0 y 1. |
| [`CAPTURA_METRICAS.md`](CAPTURA_METRICAS.md) | Qué evento de LiveKit llena cada campo de `shared/metrics`. Lado de implementación de [`docs/CONTRATO_METRICAS.md`](../docs/CONTRATO_METRICAS.md). |

Los datos del viaje, el estado de la llamada y la interpolación del prompt son
comunes a las 3 PoC y viven en `shared/` (`negotiation_engine/trip.py`,
`negotiation_engine/session.py`, `system_prompt/prompt.py`).

Ninguna prueba necesita red, llaves ni teléfono.

```bash
# Estado del entorno, sin tocar la red
uv run --project poc2_livekit python -m poc2_livekit.app.config

# Humo nivel 0: sin red, sin costo
uv run --project poc2_livekit python -m poc2_livekit.app.smoke

# Pruebas (desde la raíz, que es donde está pytest.ini)
uv run --project poc2_livekit pytest poc2_livekit
```

## Qué se reutiliza de `shared/` — sin cambios

| De `shared/` | Qué se usa | Cambios para PoC2 |
|---|---|---|
| [`negotiation_engine/`](../shared/negotiation_engine/) | Validación de precios, concesiones decrecientes, catálogo de compensaciones, contrato de cierre | **ninguno** |
| [`system_prompt/v0.md`](../shared/system_prompt/v0.md) | El prompt completo, con su apertura corta | **ninguno** — congelado e idéntico en las 3 PoC |
| [`metrics/`](../shared/metrics/) | Todo el cálculo: percentiles, WER/CER, error de montos, reporte Markdown + CSV, la CLI | **ninguno** — lo único que cambia entre PoC es la *captura* |
| [`test_scenarios/escenarios.yaml`](../shared/test_scenarios/) | Los 5 escenarios y sus líneas como dato | **ninguno** |

Y la configuración congelada del `.env`: `gpt-4.1-mini`, `nova-3`/`multi`,
`eleven_flash_v2_5` y la misma voz de ElevenLabs. El humo nivel 0 **falla** si
alguno se separa de PoC1.

### Qué es específico de LiveKit

Solo la capa que toca el framework: config, ensamblado de la sesión, registro de
las funciones, captura de latencia y marcador. Nada de eso contiene reglas de
negociación.

## Arquitectura de esta PoC

| Capa | Tecnología |
|---|---|
| Telefonía | LiveKit SIP, trunk **saliente** sobre Twilio (`SIP_OUTBOUND_TRUNK_ID` en `.env`) |
| Medios | WebRTC de LiveKit; el participante SIP entra a la sala |
| STT | Deepgram `nova-3` / `multi` |
| LLM | `gpt-4.1-mini` |
| TTS | ElevenLabs `eleven_flash_v2_5`, misma voz que PoC1 |
| Fin de turno | `inference.TurnDetector`, el detector **de audio**, versión `v1-mini` (local) fijada en `.env` |
| VAD | Silero con los defaults de LiveKit (`min_silence_duration = 0.55 s`) |
| Negociación | `shared/negotiation_engine` vía tool-calling |

## Pipecat → LiveKit, componente por componente

| Función | PoC1 (Pipecat) | PoC2 (LiveKit) |
|---|---|---|
| Originar la llamada | REST de Twilio (`outbound-api`) con `url` de TwiML por llamada | `CreateSIPParticipantRequest` contra la API de LiveKit |
| Recibir el audio | Servidor FastAPI + WebSocket de Media Streams + **túnel ngrok** | **Nada**: el participante SIP entra a la sala y el worker ya está dentro |
| Ensamblado | `Pipeline([transporte.input(), stt, llm, tts, transporte.output(), ...])` | `AgentSession(stt=, llm=, tts=, vad=, turn_handling=)` + una clase `Agent` |
| Fin de turno | `LocalSmartTurnAnalyzerV3`, ONNX **local**, en `UserTurnStrategies` | `inference.TurnDetector()` en `TurnHandlingOptions`; `v1-mini` corre **local** |
| Tool calling | `FunctionSchema` + `registrar_funciones(llm)` | `function_tool(..., raw_schema=...)`, **mismo JSON** |
| Etiqueta de corrida | query del TwiML → `<Parameter>` → `customParameters` | `metadata` del dispatch → `ctx.job.metadata` |
| Latencia | 3 `SondaDeLatencia` leyendo frames | `ChatMessage.metrics` vía `conversation_item_added` |
| Probar sin teléfono | humo nivel 1: pipeline de texto propio + WAV | `AgentSession` **sin sala**, en modo texto |
| Colgar al cerrar | `task.stop_when_done()` → `EndFrame` → colgado por REST | `session.drain()` → `aclose()` → `delete_room()` |
| Evidencia para el WER | transcripción del STT en el JSON, referencia contra el guion actuado | **igual**: no depende del framework |

Dos cosas que se ven bien para LiveKit y conviene decir con cuidado en el
reporte: **desaparecen tres piezas de infraestructura** (túnel, servidor de
webhooks y serializador de audio telefónico) y **hay una forma nativa de probar
sin teléfono**. Son ventajas de operación, no de latencia; van en la sección de
esfuerzo de integración.

Y una que se ve mal y también hay que decir: el detector de turnos de PoC2 **no
publica su probabilidad al código**, así que un falso fin de turno no se puede
diagnosticar después — en PoC1 sí, leyendo el JSON. Ver
[`CAPTURA_METRICAS.md`](CAPTURA_METRICAS.md).

Sobre dónde corre el detector: PoC2 usa `v1-mini`, **local en CPU**, porque todo
PoC2 corre local y nada está desplegado en LiveKit Cloud. Así queda local contra
el Smart Turn v3 en proceso de PoC1, sin un salto de red que el resto de la
arquitectura no tiene. Es un confound menos en la comparación de latencia.

## Aislamiento del proyecto Dispatcher-Voice

El proyecto de LiveKit es el del **agente despachador real, que está en uso**.
PoC2 comparte URL, llaves, trunk y número. Lo que no comparte:

| | Despachador | PoC2 |
|---|---|---|
| Nombre del agente | `outbound-dispatcher-test` (`AGENT_NAME`) | `spots-poc2-negociador` (`LIVEKIT_AGENT_NAME`) |
| Salas | las suyas | prefijo `spots-poc2-…` |

**El mecanismo de aislamiento es el nombre del agente.** Un worker registrado con
`agent_name` no recibe dispatch automático: solo entra a una sala cuando alguien
lo pide por nombre con `CreateAgentDispatch`. Mientras los nombres sean
distintos, ningún job del despachador puede aterrizar en el agente de PoC2.

Reglas, no negociables:

1. **Nunca leer `AGENT_NAME`.** `config.py` usa `LIVEKIT_AGENT_NAME` y revienta
   (`ColisionConElDespachador`) si alguien le pone el valor del despachador. Está
   cubierto por pruebas.
2. **No crear, editar ni borrar dispatch rules, trunks ni reglas de entrada.** El
   trunk se referencia por id; su configuración no se toca. Eso incluye no
   activar grabación a nivel de trunk.
3. **Solo saliente.** PoC2 no registra ninguna regla de entrada, así que no puede
   contestar llamadas dirigidas al número del despachador.
4. **`record=False` en la sesión.** El `record=True` de LiveKit sube audio,
   transcripción y trazas a la observabilidad del proyecto — que es el del
   despachador.
5. **El worker corre local**, desde la terminal. No se despliega nada al cloud.

## Evidencia: PoC2 no graba audio

**No hay capa de grabación, y es a propósito.** El WER de la comparativa se
calcula contra el **guion actuado**, no contra un archivo de audio: las líneas
`stt:` del JSON de cada llamada son la hipótesis. En las llamadas telefónicas
de agosto la referencia eran las líneas `yo:`, corregidas a mano por quien
actuó; con el arnés la referencia se deriva del clip reproducido
([`docs/CONTRATO_METRICAS.md`](../docs/CONTRATO_METRICAS.md) §2.10). Los dos
métodos son idénticos en las 3 PoC, que es lo que hace comparables los números.

Consecuencia para PoC2, y hay que decirla en el reporte: sin audio, un falso fin
de turno **no se puede diagnosticar después**. En PoC1 queda la probabilidad del
detector por turno en el JSON (`veredictos_fin_de_turno`); en PoC2 el detector
acústico no la expone al código, así que la única oportunidad de juzgarlo es
**en vivo, durante la llamada**. Por eso `naturalidad` y `turn_taking` se
puntúan al colgar y no después. Ver [`CAPTURA_METRICAS.md`](CAPTURA_METRICAS.md)
y [`docs/CONTRATO_METRICAS.md`](../docs/CONTRATO_METRICAS.md).

## Entorno

```bash
uv sync --project poc2_livekit    # ya ejecutado: hay uv.lock
```

Python 3.12 (livekit-agents 1.6.8 acepta 3.10–3.14; se fija 3.12 para no cambiar
de versión de Python entre arquitecturas).

Nota sobre los extras: **no se instala `turn-detector`**. Ese es el plugin del
detector **por texto** (`MultilingualModel`), deprecado y con eliminación
anunciada para la 2.0. El de audio viene dentro de `livekit-agents`.

## Cómo se corre una llamada

Dos terminales. El worker se queda arriba entre llamadas; solo el marcador se
vuelve a correr.

```bash
# Terminal 1 — el worker (no marca, no gasta)
uv run --project poc2_livekit python -m poc2_livekit.app.agente

# Terminal 2 — una llamada
uv run --project poc2_livekit python -m poc2_livekit.app.llamada \
    --escenario 2 --repeticion 1 --acepto-costo
```

Sin `--acepto-costo` el marcador solo imprime la estimación y sale.

| Bandera | Para qué |
|---|---|
| `--a +52...` | marcar a otro número (default: `TWILIO_NUMERO_DESTINO`) |
| `--escenario` / `--repeticion` | etiqueta de la corrida; viaja en el `metadata` del dispatch |
| `--limite 300` | baja el tope duro de duración (default 600 s) |

### Corrida de campaña vs. corrida de depuración

Lo que separa las dos es `--escenario`: con él, el JSON entra al consolidado y a
la decisión técnica; sin él, no.

Por eso el número de escenario se valida contra
[`shared/test_scenarios/`](../shared/test_scenarios/) **antes de leer la
configuración y antes de tocar la red**. Un typo produciría una corrida
etiquetada como campaña que no corresponde a ningún guion, y eso se descubriría
al consolidar, tarde:

```
$ python -m poc2_livekit.app.llamada --escenario 9 --repeticion 1
[rechazada] no existe el escenario 9; disponibles: 1, 2, 3, 4, 5
```

El código de salida es `2`, distinto del `1` de "falta algo en el `.env`", para
poder guionizar las 15 corridas y distinguir los dos fallos.

Al colgar, la corrida queda en `poc2_livekit/grabaciones/llamada_<sala>.json`
(gitignoreado), con la misma forma que el JSON de PoC1 más un bloque
`livekit_extra` con los diagnósticos.

## Plan de implementación

- [x] **Inc. 1 — scaffolding.** Estructura, dependencias, config, contrato de métricas.
- [x] **Inc. 2 — negociación sin voz.** Prompt, sesión, las 4 funciones y humo nivel 0.
- [x] **Inc. 3 — voz sin teléfono.** `AgentSession`, captura de latencia, humo nivel 1.
- [x] **Inc. 4 — la llamada.** Tabla de costos aprobada
      ([`docs/costos_poc2.md`](../docs/costos_poc2.md)) y llamada de diagnóstico.
- [x] **Inc. 5 — las 15 corridas telefónicas** (5 escenarios × 3 repeticiones,
      agosto de 2026; [`docs/metricas/`](../docs/metricas/)). Las mediciones del
      reporte salen después de la campaña del arnés.

## Pendientes abiertos

- [ ] **Confirmar los renglones de LiveKit Cloud** en la tabla de costos: son los
      únicos sin verificar.
- [ ] **Aviso de grabación y consentimiento** (REPEP, LFPDPPP, CRT). Mismo estado
      que PoC1: las llamadas de la comparativa van a un celular propio; es
      bloqueante absoluto antes de marcarle a un transportista real.

## Antes de la primera llamada real

1. Tabla de costos aprobada — [`docs/costos_poc2.md`](../docs/costos_poc2.md).
2. Aviso de grabación y consentimiento.
3. **Confirmar el aislamiento en vivo**: que el worker de PoC2 esté arriba con su
   `agent_name` y que el despachador siga recibiendo sus llamadas con
   normalidad. El chequeo de configuración es necesario pero no suficiente.
