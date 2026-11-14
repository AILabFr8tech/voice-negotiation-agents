# Captura de métricas en PoC2 — qué evento de LiveKit llena cada campo

Este archivo es el **lado de implementación** de
[`docs/CONTRATO_METRICAS.md`](../docs/CONTRATO_METRICAS.md). Aquel documento fija
qué se iguala entre las 3 PoC y qué no; este dice, campo por campo, de dónde sale
el número en LiveKit Agents.

`shared/metrics/` **calcula**; cada PoC **mide**. El contrato es
[`MuestraDeTurno`](../shared/metrics/latencia.py) y PoC2 tiene que llenarlo con el
mismo significado que PoC1, o los percentiles no son comparables.

**Verificado contra livekit-agents 1.6.8 instalado**, leyendo el paquete — no
contra la documentación. Cada afirmación lleva su estado: **directo** (mapeo
inmediato), **resuelto** (había una duda y se cerró leyendo el SDK), **a
confirmar** (solo se puede cerrar con una llamada) o **no expuesto** (LiveKit no
lo da y no se va a inventar).

---

## Turn detection: `inference.TurnDetector`, el de audio

PoC2 usa el audio turn detector. El porqué está en
[`docs/CONTRATO_METRICAS.md` §2.1](../docs/CONTRATO_METRICAS.md) y no se repite
aquí. Lo que sí va aquí es el detalle de API, ya verificado contra la
documentación:

| | Audio turn detector | Text turn detector |
|---|---|---|
| API | `livekit.agents.inference.TurnDetector` | `MultilingualModel` (plugin `turn-detector`) |
| Instalación | viene dentro de `livekit-agents` | extra `turn-detector` — **no se instala** |
| Decide sobre | el audio: contenido **y** prosodia | el texto que devuelve el STT |
| Estado | recomendado | **deprecado, se elimina en la 2.0** |
| Parámetros | `version`, `unlikely_threshold` | — |
| Endpointing | `min_delay` / `max_delay` en `TurnHandlingOptions` de `AgentSession` (0.3 / 2.5 s con el detector de audio) | — |
| VAD | exige `min_silence_duration ≥ 0.25 s`; el default de Silero (0.55 s) ya cumple | — |

**Hallazgo que no estaba en el contrato y sí importa: una de las versiones corre
remota.** `v1` es el modelo completo, servido en LiveKit Inference; `v1-mini`
corre local en CPU. Verificado en `inference/eot/detector.py`:

```python
auto = not is_given(version)
resolved_version = (
    version if is_given(version)
    else ("v1" if (utils.is_hosted() or utils.is_dev_mode()) else "v1-mini")
)
```

O sea: **sin `version=` explícito, la arquitectura del detector dependería de
cómo se lanzó el worker** — `dev`/`console` ponen `LIVEKIT_DEV_MODE=1` y dan el
modelo remoto; `start` da el local. Eso no es una base defendible para un
benchmark, así que PoC2 lo fija en `LIVEKIT_TURN_DETECTOR_VERSION` (default
`v1-mini`) y `config.py` rechaza cualquier otro valor.

Ojo con un detalle del SDK que muerde: con `version` explícito el chequeo de
credenciales solo corre en la rama `v1`. Un valor que no sea exactamente `"v1"`
—incluido un typo o una cadena vacía— cae a `turn-detector-v1-mini` **sin
avisar**. Por eso `config.py` valida contra la lista de versiones válidas antes
de construir nada, y el JSON registra la versión que corrió de verdad.

**Se elige `v1-mini`** (decisión del 7 de agosto de 2026, tras la llamada 1.1):
PoC2 corre entero en local, con el worker lanzado desde la terminal y nada
desplegado en LiveKit Cloud. Pedir `v1` metería un round trip a LiveKit
Inference dentro del camino de turn-taking que ni el resto de la arquitectura ni
PoC1 pagan —Smart Turn v3 corre como ONNX en proceso—, y podría explicar por sí
solo una diferencia de latencia. Con `v1-mini` queda **local contra local**: un
confound menos, no una concesión.

Ojo con una cosa más: el detector **puede cambiar de versión durante la
llamada**. `_fall_back_to_local` reemplaza el transporte remoto por el local si
el cloud falla (`base.py`) y muta `_model` en caliente. Con `v1-mini` ese modo
de falla no aplica —ya se corre local— pero el JSON sigue registrando la versión
resuelta **al iniciar y al terminar**, porque esa misma vigilancia atrapa el
caso de que el `.env` y lo que corrió no coincidan.

### Qué pasa si el detector se degrada

`ColectorDeLatencia.degradacion_del_detector()` la detecta en dos formas —cambió
de versión a media llamada, o arrancó en una versión distinta de la pedida— y
`guardar_corrida` escribe entonces la llave `excluir_de_pooled` al nivel de
arriba del JSON.

`shared/metrics/` la respeta al consolidar: esas llamadas **no votan en los
percentiles agregados**, porque mezclarían dos arquitecturas de turn-taking
dentro de la misma muestra. Lo que **no** pasa es que se descarten:

- siguen en la tabla de detalle por llamada, con su p50 propio;
- su WER y sus notas manuales siguen contando;
- el consolidado dice cuántas fueron y por qué.

Que el fallback ocurra es en sí un dato de confiabilidad de LiveKit —PoC1 no
puede tener ese modo de falla, porque su detector corre en proceso— así que
esconderlo sería perder un resultado.

La llave es **opcional y genérica**: PoC1 y PoC3 nunca la escriben, y sin ella el
consolidado se comporta exactamente como antes. Hay pruebas de las dos mitades
en `tests/test_exclusion_pooled.py`.

---

## Mapeo campo por campo

El orden real de los eventos en un turno es el mismo que ya se corrigió en PoC1
(el fin de turno llega **después** de la transcripción, no antes):

```
fin de habla del transportista   <- cero del turno
  -> transcripción final          (STT)
  -> fin de turno                 (el detector declara que terminó)
  -> primer token del LLM
  -> primer byte de audio del TTS <- se cierra la muestra
```

En LiveKit 1.6.8 las métricas por turno viven en `ChatMessage.metrics`, que se lee
por el evento `conversation_item_added` (`ev.item.metrics`) o recorriendo
`session.history`. **`metrics_collected` está deprecado** y no se usa.

| Campo del esquema | PoC1 (Pipecat) | PoC2 (LiveKit) | Estado |
|---|---|---|---|
| cero del turno | `VADUserStoppedSpeakingFrame` **menos `stop_secs`** | `stopped_speaking_at` del mensaje del usuario, **sin restarle nada** | **resuelto** — ver A |
| `fin_de_turno_ms` | `UserStoppedSpeakingFrame` − cero | `end_of_turn_delay` | directo |
| `stt_ms` | `TranscriptionFrame` − cero | `transcription_delay` | directo — ver B |
| `llm_ttft_ms` | primer `LLMTextFrame` − fin de turno | `llm_node_ttft` | directo — ver C |
| `tts_ttfb_ms` | primer `TTSAudioRawFrame` − primer token | `tts_node_ttfb` | directo |
| `turno_completo_ms` | primer audio del TTS − cero | `e2e_latency`, tal cual | **resuelto** — ver D |
| `interrumpido` | el usuario empieza a hablar con el agente hablando | `agent_state_changed` = `speaking` + `user_state_changed` → `speaking` | directo — ver E |
| `es_apertura` | `colector.iniciar()` al conectar el Media Stream | **no hay campo**: el saludo no tiene mensaje de usuario previo | **no expuesto** — ver F |
| `veredictos_fin_de_turno` | `TurnMetricsData.probability` del Smart Turn | — | **no expuesto** — ver G |
| `notas` | lo llena `MuestraDeTurno.desde_eventos` | igual | directo |

### A. El cero honesto del turno — RESUELTO en las dos PoC: no se resta nada

Es el punto que más podía sesgar la comparativa, así que va primero.

La regla del contrato es que **el cero cae donde la persona de verdad dejó de
hablar**, no donde el VAD lo confirmó. PoC1 lo consigue restando su `stop_secs`
(0.2 s) al instante del frame. La duda era si PoC2 tenía que restar sus 0.55 s.

**No hay que restar nada: LiveKit ya lo hace internamente.** En
`livekit/agents/voice/audio_recognition.py`, rama `VADEventType.END_OF_SPEECH`:

```python
speech_end_time = time.time() - ev.silence_duration - ev.inference_duration
self._last_speaking_time = speech_end_time
```

`_last_speaking_time` es exactamente lo que `_compute_end_of_turn_metrics`
publica como `stopped_speaking_at`, y de ahí cuelgan `transcription_delay`,
`end_of_turn_delay` y `e2e_latency`. Los dos frameworks anclan en el mismo
evento físico, cada uno por su cuenta.

Las dos equivocaciones posibles eran grandes, opuestas, e invisibles en el
reporte:

- No restar cuando tocaba → PoC2 saldría hasta 550 ms **más rápida** de lo real.
- Restar cuando no tocaba (lo que se estuvo a punto de hacer) → PoC2 saldría
  hasta 550 ms **más lenta**.

**Y no se deja en la palabra de nadie.** `ColectorDeLatencia.veredicto_del_cero()`
re-verifica la conclusión con los datos de cada llamada, y el resultado va al
JSON en `livekit_extra.veredicto_del_cero`:

- Si el cero está en el fin de habla real, `fin_de_turno_ms` tiene que incluir la
  espera de silencio del VAD (0.55 s) más el `min_delay` de endpointing (0.3 s):
  del orden de **850 ms o más**.
- Si estuviera en la confirmación del VAD, esa espera ya habría transcurrido y el
  delta rondaría los **300 ms**.

Son órdenes de magnitud distintos, así que una sola llamada decide. Si algún día
sale `anclado_en_confirmacion_del_vad`, el SDK cambió de criterio y hay que
restar `MIN_SILENCE_DURATION` y recalcular toda la comparativa.

#### A.1 Y con el arnés, el cero se mide de frente (PASO D)

El veredicto de arriba es un argumento **por magnitud**: concluye de dónde
cuelga el cero mirando si `fin_de_turno_ms` alcanza los 850 ms. Vale, pero no
mide el error; solo descarta que sea de medio segundo.

El arnés sí lo mide. Publica sus ceros por el canal de datos **confiable**
(`arnes.cero_del_turno`) y el worker los contrasta contra su propio
`stopped_speaking_at`. Detalle en [`app/marcas.py`](app/marcas.py).

##### Son DOS ceros, y significan cosas distintas

| Cero | Qué es | Naturaleza |
|---|---|---|
| `fin_del_clip` | el último sample salió por la pista | **hecho físico** — esos bytes se empujaron |
| `fin_de_voz_medido` | dónde dejó de haber voz *dentro* del clip | **medición** — depende de un umbral |

De ahí salen los dos sesgos:

```
sesgo_de_clip_ms = stopped_speaking_at - fin del archivo     (control)
sesgo_de_voz_ms  = stopped_speaking_at - última voz          (la medida)
```

**El umbral del segundo es la caída de 25 dB bajo el pico** — el mismo
`CAIDA_PARA_NIVEL_DB` con el que se mide el nivel de voz, y **no** el permisivo
de 35 dB del recorte, que existe para no morder fricativas finales. Viaja en
cada marca y en cada fila del JSON (`caida_db`, `umbral_voz_dbfs`), porque el
mismo audio mide hasta 9 dB distinto según el criterio: el número no es
reproducible sin él. Si en `criterio_del_fin_de_voz` aparece más de un valor, los
turnos se midieron con criterios distintos y no se comparan entre sí.

##### Por qué no basta con el del archivo

Ese fue el primer diseño y produjo un número falso. Corrida del 15 de septiembre
de 2026, escenario 1: el turno 6 salió con **−707 ms** de sesgo, que como
latencia es imposible — diría que LiveKit detectó el fin de voz antes de que la
voz terminara. La causa es que `e1_t6` arrastra **1,289 ms** de respiración y
cola de sala después de la última sílaba, y el cero estaba fechado en el fin del
*archivo*. Sobre los 7 turnos medidos, la cola correlaciona con el sesgo a
**r = −0.986 con pendiente −0.9 ms por ms**: no es una asociación, es una
identidad.

Y **no se corrige con el margen de recorte**. Son 30 ms fijos contra colas
reales de 50 a 1,289 ms: una cota inferior, no una corrección. Por eso no se
publica — un campo así se acaba usando como si corrigiera. La cola se **mide**
por clip y vive en el manifiesto.

##### Vive en `shared/audio`, no en el arnés de PoC2

El fin de voz es una propiedad **del clip**, y las 3 PoC reproducen el mismo
clip. Se mide una sola vez al preparar (`nivel.fin_de_voz`) y se guarda en el
manifiesto junto al hash; el arnés lo lee de ahí y no lo re-mide. Si cada PoC lo
midiera por su cuenta, bastaría una diferencia de versión para que las tres
compararan contra ceros distintos, y el sesgo dejaría de ser común. `cli
preparar` avisa por clip cuando la cola pasa de 250 ms.

##### ✅ CERRADA (17 de septiembre de 2026): el cero NO está corrido, y por qué lo parecía

> **Lo que sigue en este apartado se conserva como está** porque el recorrido
> importa —es el ejemplo de cómo un discriminante indirecto manda a buscar el
> problema donde no está— pero **la conclusión ya no está abierta**. Resumen,
> con las dos mediciones que la cierran, las dos en banco local y sin gastar
> (`python -m poc2_livekit.app.diagnostico_cero`):
>
> 1. **La resta sí ocurre, y es de 576 ms.** `silence_duration` mide 576.0 ms
>    exactos en los 6 clips del banco —18 ventanas de Silero de 32 ms, no los
>    550 nominales— y el SDK la resta entera.
> 2. **Sin transporte, el sesgo sale en 170 ms.** Si los 576 no se restaran,
>    ninguna medición podría salir por debajo de 576.
>
> El error estaba en la vara: el sesgo nunca fue `min_silence_duration + algo`,
> sino **desajuste de criterio + camino del audio**, y comparar contra 550 no
> tenía sentido. Los 620-735 ms se descomponen en 170 de criterio, 20 de la cola
> de la `AudioSource`, 68 del SFU y ~403 de Opus más jitter buffer.
> **No hay que restar nada a la latencia de PoC2.** Detalle en
> `docs/CONTRATO_METRICAS.md` §2.2.2 y §2.2.3.
>
> Corrección de paso, porque abajo se cita mal: el piso de 576 ms de
> `fin_de_turno_ms` **no** es "550 + 27 ms de `inference_duration` implícito".
> `inference_duration` llega en **0.0** — el plugin de Silero no lo puebla en el
> evento `END_OF_SPEECH`, es el default del dataclass en
> `livekit/agents/vad.py:58` — así que los 576 son pura cuantización de ventana.
> La conclusión de ese apartado (que el mecanismo propuesto no explica el
> desplazamiento) no cambia; el número al que se le puso nombre, sí.

**La medición directa contradice lo que `veredicto_del_cero()` concluye por
magnitud, y si se confirma es una corrección sobre TODA la latencia de PoC2 en
la comparativa.** Va aquí arriba y no en una nota al pie porque nada de lo que
sigue en este documento sería correcto si se confirma.

El veredicto por magnitud dice `anclado_en_fin_de_habla_real`: LiveKit ya
descuenta la espera del VAD, no hay que restar nada. La medición del arnés dice
otra cosa. Corrida del escenario 1 del 15 de septiembre de 2026, 7 turnos:

```
caída del umbral    mediana del sesgo de voz
      20 dB               734.5 ms
      25 dB               660.9 ms
      30 dB               620.9 ms
      35 dB          (circular — ver abajo)
```

`min_silence_duration` es **550 ms**. El sesgo no baja de 620 en ningún punto
del rango defendible.

**Lo importante es que el resultado es robusto al criterio, no dependiente de
él.** Sería fácil descartarlo como "eligió el umbral que le convenía": no lo es.
Mover la caída de 20 a 30 dB —todo el rango con sentido— mueve la mediana
113 ms, y los tres valores quedan por encima de la espera del VAD. Para llegar a
550 hay que irse a 35 dB, y **ese número es circular**: el recorte ya cortó el
clip a 35 dB, así que medir ahí devuelve dónde cortamos, no dónde termina la
voz. Se ve en el dato — a 35 dB la cola mide exactamente 30 ms en los 49 clips,
que son los 30 ms del margen de recorte.

Si el cero está de verdad en la confirmación del VAD y no en el fin de habla
real, hay que restar `MIN_SILENCE_DURATION` a todas las marcas de PoC2 y
recalcular la comparativa entera. Medio segundo sobre p50, p95 y p99.

**Con n=7 de una sola llamada esto no concluye**, y hay una razón concreta para
no forzarlo: entre dos corridas del mismo escenario el mismo día la mediana se
movió ~150 ms, que es del orden del margen sobre 550. La incertidumbre por
corrida compite con el efecto.

###### La segunda corrida lo midió, y es peor: el sesgo tiene un desplazamiento POR LLAMADA

Corrida de verificación del 15 de septiembre de 2026, 18:39, mismo escenario 1,
mismos clips (recortados, pero con el fin de voz medido idéntico al milisegundo)
y 6 turnos emparejados. Los dos rangos **no se tocan**:

```
corrida   n    min     mediana    max     ¿algún turno bajo 550?
16:41     7   579.3     660.9    752.7    ninguno
18:39     6   307.5     388.7    468.2    todos
```

Clip por clip, contra la corrida anterior:

```
clip      ayer     hoy    Δ sesgo    Δ cola
e1_t2    660.9   467.4    -193.5       0.0
e1_t3    607.7   348.4    -259.3       0.0
e1_t4    579.3   358.7    -220.6       0.0
e1_t5    634.6   307.5    -327.1       0.0
e1_t6    752.7   468.2    -284.5   -1,059.0
e1_t7    712.9   418.6    -294.3     -340.0
```

**No es el recorte.** Los cuatro clips cuya cola no se movió **ni un
milisegundo** bajaron entre 194 y 327 ms, igual que los dos a los que sí se les
cortó cola. Recortar 1,059 ms de respiración de `e1_t6` movió su sesgo lo mismo
que no tocar `e1_t5`. Eso además confirma, de paso, que Silero ya encontraba el
fin de la voz e ignoraba la cola débil — el recorte era correcto por otras
razones, pero no tocaba esta medición.

Lo que se movió es **toda la corrida a la vez**: −263 ms de media, con una
dispersión entre clips (194–327, rango 133 ms) del mismo orden que la
dispersión **dentro** de cada corrida (RIC de ~100 ms). O sea: el sesgo tiene un
desplazamiento constante por llamada que es **mayor que su variación por turno**.

La hipótesis que se planteó entonces fue la carga: la segunda corrida tenía
`stt_ms` de 335 a 902 ms de mediana y el detector de fin de turno se había ido
del camino rápido al lento (`fin_de_turno_ms` de 577.6 a 2,481.4 de mediana).
Con la fórmula de LiveKit —`speech_end_time = time.time() - silence_duration -
inference_duration`— más carga significaría `inference_duration` más largo, que
fecha el cero **más temprano**, que es la dirección observada. **Esa hipótesis
se cruzó contra los datos y quedó descartada**; los dos números de arriba, de
hecho, resultaron ser consecuencia del desplazamiento y no su causa. Dos
apartados más abajo.

**Tres consecuencias, y son de método:**

1. **Ninguna llamada suelta decide esto.** Una dice que el sesgo supera
   `min_silence_duration` en los 7 turnos; la otra dice que no lo alcanza en
   ninguno de los 6. Las dos están bien medidas.
2. **La n efectiva de la campaña son las llamadas, no los turnos.** Juntar
   15 × ~7 turnos como 105 muestras independientes subestimaría el error: la
   varianza entre llamadas domina a la de dentro. El contraste se hace sobre la
   mediana **por llamada**, con n=15.
3. **El desplazamiento no correlaciona con la carga.** Se cruzó contra ella y no
   aparece: ni dentro de cada corrida ni entre corridas. El apartado siguiente
   tiene el cruce y por qué los dos proxies obvios no servían para hacerlo.

**La magnitud se resuelve con la campaña**, no con más análisis sobre estas
corridas: cada llamada del arnés publica su bloque `arnes`, y con 15 llamadas el
contraste tiene la n para acotar cuánto se mueve el desplazamiento por llamada.
Hasta entonces, los números de latencia de PoC2 se citan como están y **este
párrafo se cita junto a ellos**.

###### Se cruzó contra la carga y el mecanismo propuesto queda descartado

El cruce se hizo offline sobre las tres corridas del arnés, sin gastar nada.
**El resultado no confirma la hipótesis: la descarta** — y de paso destapa que
los dos proxies con los que se iba a hacer no servían.

**Los dos proxies estaban contaminados por construcción.** `stt_ms` y
`fin_de_turno_ms` son `transcription_delay` y `end_of_turn_delay`, y el SDK los
calcula desde la misma ancla cuyo sesgo se quería explicar
(`voice/audio_recognition.py`, `_compute_end_of_turn_metrics`):

```python
stopped_speaking_at = last_speaking_time
transcription_delay = max(last_final_transcript_time - last_speaking_time, 0)
end_of_turn_delay   = max(now - last_speaking_time, 0)
```

Si el ancla se corre δ hacia atrás, el sesgo baja δ y los dos delays suben δ sin
que haya pasado nada físico. Una correlación negativa ahí es aritmética, no
evidencia — y aparece: dentro de la corrida de las 16:41 el sesgo da r = −0.80
(p = 0.03) contra `stt_ms` y r = −0.81 (p = 0.03) contra `fin_de_turno_ms`.
**Esos dos números no significan nada**, y son exactamente los que habrían
cerrado el caso en falso.

**El piso de `fin_de_turno_ms` se puede medir de frente, sin correlación.** El
SDK duerme hasta `last_speaking_time + endpointing_delay`
(`audio_recognition.py`, ~1678), así que `end_of_turn_delay =
max(endpointing_delay, lo ya transcurrido)`. Con `min_delay` de 0.3 s contra
`min_silence_duration` de 0.55 s, en la rama de confianza el sueño sale
negativo: no duerme, y lo que publica **es** la espera de silencio del VAD. Ese
es el piso de la distribución:

```
corrida   piso de fin_de_turno_ms   sobre los 550 nominales
15:15            577.5 ms                   +27.5 ms
16:41            576.0 ms                   +26.0 ms
18:39            577.0 ms                   +27.0 ms
```

> **Corregido el 17 de septiembre de 2026.** Esa tercera columna decía
> `inference_duration implícito`, y **ese no es su nombre**. `inference_duration`
> llega en `0.0` en el evento `END_OF_SPEECH` (el plugin de Silero no lo puebla;
> `livekit/agents/vad.py:58`), así que no puede explicar nada de este piso. Los
> ~27 ms son **cuantización de ventana**: Silero acumula el silencio en ventanas
> de 512 muestras a 16 kHz = 32 ms, y el primer múltiplo que alcanza los 550 es
> 18 × 32 = **576 ms**. Medido directamente en 6 clips, `silence_duration` da
> 576.0 exactos. El piso no es "550 + inferencia", es 576 y nada más.
>
> **La conclusión de este apartado no cambia** —el mecanismo propuesto no explica
> el desplazamiento entre corridas— y de hecho se refuerza: 576 es una constante
> aritmética, no un tiempo de cómputo, así que no podía moverse con la carga ni
> en principio.

Los 8 turnos de confianza de las tres corridas caen entre **576.0 y 580.6 ms**:
4.6 ms de rango total. El piso **no se movió**, mientras el sesgo se desplazaba
273 ms entre dos de esas mismas corridas. El mecanismo propuesto está acotado a
±5 ms y no alcanza para explicar nada.

**Contra un proxy de carga limpio tampoco hay relación.** `llm_ttft_ms` y
`tts_ttfb_ms` vienen del mensaje del asistente y no cuelgan del ancla. Pareando
por clip entre las dos corridas de criterio 25 dB, Δsesgo contra Δ`llm_ttft_ms`
da **r = +0.06** con n=6. Y las medidas independientes van al revés de la
hipótesis: la corrida de las 18:39 —la del sesgo *más chico*— tuvo el TTS **más
rápido** (223 contra 244 ms de mediana), y el transporte del canal de datos no
se movió en ninguna de las tres (66.2, 67.9 y 68.5 ms de mediana).

**El `stt_ms` que "subió" era el cero moviéndose.** De 669 a 894 ms de media
entre las dos corridas: +224 ms contra un cero que se adelantó 273 ms, la misma
cantidad con el signo cambiado, mordida por el `max(..., 0)` del SDK — que
recorta a cero 2 de 7 turnos en una corrida y 1 de 6 en la otra. **`stt_ms` en
PoC2 no sirve como indicador de carga**: está anclado en un cero móvil y truncado
por abajo. Eso no toca el contrato —`transcription_delay` se sigue publicando tal
cual, ver el apartado B— pero sí prohíbe leerlo como "qué tan ocupado estaba el
STT".

**Qué queda entonces.** Descontados `silence_duration` e `inference_duration`,
que son constantes, `stopped_speaking_at` es el instante en que el último audio
con voz **llegó al worker**. O sea que `sesgo_de_voz` mide el retardo del camino
de audio —encoder Opus, jitter buffer, ventana de Silero— y lo que cambia entre
llamadas es ese retardo, no el detector. Como el canal de datos no se movió, no
es la red: la sospecha es la profundidad del jitter buffer, que se adapta y puede
asentarse distinto en cada llamada. **No está probado**, y con tres llamadas no
se va a probar. Lo que sí quedó probado es cuál mecanismo **no** es.

###### `turno_completo_ms` subestima lo que el transportista espera

`e2e_latency`, de donde sale `turno_completo_ms`, es `started_speaking_at −
stopped_speaking_at` (`voice/agent_activity.py:3404`). Los dos extremos son
instantes **del worker**: cuándo le llegó el audio del transportista y cuándo
empezó a emitir el suyo. Lo que el transportista espera va de su boca a su oído,
y eso tiene dos tramos que la medida no ve:

- el de ida, que es exactamente `sesgo_de_voz`: **395 a 667 ms** de mediana
  según la llamada;
- el de vuelta, del worker al transportista, que el arnés no mide y que no hay
  razón para suponer más chico.

O sea que **`turno_completo_ms` es una cota inferior de la espera real**, corta
por al menos 400–700 ms, y la falta varía ~273 ms entre llamadas. No es defecto
de la instrumentación —es lo que el SDK define, y PoC1 y PoC3 también miden de
puertas adentro—, pero **el p50 de PoC2 no se cita como "lo que el transportista
espera"**. Para comparar frameworks entre sí sirve; para describir la
experiencia del transportista, no.

Vale la pena notar que el mecanismo descartado, de haber existido, habría ido en
el sentido contrario: un `inference_duration` que crece bajo carga fecha el cero
más temprano y por lo tanto **infla** `turno_completo_ms`. La subestimación no
viene de ahí; viene del tramo de transporte, que la medida deja fuera por
definición.

**Y aplica a las 15 llamadas de la campaña, donde no había arnés que lo viera.**
Ahí el audio no entra por una pista local sino por el trunk SIP desde Twilio, así
que el tramo de ida incluye PSTN, Twilio y la ingesta de LiveKit: el sesgo es
**al menos** tan grande como los 395–667 ms medidos aquí, y con toda probabilidad
mayor. El número no se transfiere, la estructura sí. Ninguna de esas 15 llamadas
tiene con qué medirlo, y el reporte tiene que decirlo en vez de presentar su p50
como la espera del transportista.

##### Qué contiene el ~473 ms observado en PoC2, y qué no

Descontada la cola, el sesgo de voz quedó en **~400 ms** sobre los 5 clips de
cola limpia (intercepto del ajuste: 473 ms). Lleva dentro **encoder Opus, jitter
buffer y la ventana de Silero**.

**No es red.** El canal de datos de este mismo módulo viaja por el mismo camino
—mismo SFU, misma sesión— y llegó en **68 ms de mediana** (63–88 ms). Ese número
se publica en `reloj.transporte_de_datos_ms` justo para poder descartar la red
sin discutirlo. Lo que el contraste **no** hace es separar los tres sumandos
restantes; para eso haría falta otra medida.

##### Tres cosas que lo mantienen honesto

- **Los turnos se emparejan por cercanía en el tiempo, no por número.** El arnés
  numera clips y LiveKit numera muestras, y los dos conteos no coinciden: el
  saludo del agente es una muestra sin clip, y LiveKit puede fusionar dos clips
  en un turno (pasó el 15 de septiembre: 8 clips → 7 muestras). Emparejar por
  índice desalinearía todo en silencio, y un sesgo calculado contra el turno
  equivocado no se ve mal — se ve como un hallazgo. Lo que no case dentro de 2 s
  queda sin emparejar y se reporta como tal.
- **Los relojes se verifican.** Restar marcas de dos procesos solo vale si
  comparten reloj. Cada marca viaja con el `host` del arnés; si no coincide con
  el del worker, el bloque marca `sesgo_interpretable: false` y no resume nada.
  No se intenta corregir: una corrección sin sincronización real sería inventar.
- **La cobertura es verificable.** Al colgar, el arnés manda un resumen con
  todas las marcas repetidas y el total que declara haber mandado. Así una marca
  perdida llega en la segunda oportunidad, y el worker puede decir "3 de 5" en
  vez de publicar 3 y callarse. **Cuando el agente cierra la sala él mismo, ese
  resumen no sale** —publicar contra una sala borrada solo da `engine is
  closed`—, así que `marcas_declaradas_por_el_arnes` queda en `null` y la
  cobertura no se puede cerrar. Las marcas en caliente sí llegaron; lo que se
  pierde es la verificación. Pasó en la corrida del 15 de septiembre a las
  18:39, y el bloque lo dice en vez de afirmar que llegaron todas.

##### El turno mudo: texto del agente sin una sola trama de voz

`agente_sin_audio` mira la **corrida entera** y por eso se le escapa la falla
que más engaña: si el TTS falla en 2 de 8 turnos, basta que uno suene para que
la regla no se dispare, y esos dos turnos entran al percentil con la latencia
que el worker calculó sobre audio que nadie oyó. El JSON se ve completo y la
llamada no lo está.

Se registra **por turno**: el arnés marca el turno en el que el agente produjo
texto y él no oyó una sola trama con voz, y en el resumen manda la lista al
worker. Es lo único de esto que el worker no puede deducir solo — él ve su TTS
pedir audio, no si ese audio sonó del otro lado.

- El turno queda marcado en `arnes.por_turno` con `audio_del_agente: false` y
  `excluir_del_percentil: true`, y listado en `arnes.confiabilidad`. **No se
  borra**: que ocurriera es el hallazgo.
- El **mínimo de turnos** de la corrida se exige sobre los medibles, no sobre
  los alcanzados. Un turno mudo no aporta latencia, así que contarlo hacia el
  mínimo sería contarlo dos veces mal: se excluye del percentil y aun así habría
  avalado la corrida.
- El **sesgo del cero sí se mide** en un turno mudo: `stopped_speaking_at`
  depende del VAD sobre **nuestro** audio, no de que el TTS del agente sonara.
- Si el resumen no llegó, `confiabilidad.declarado` es `false` y la nota lo
  dice: una lista vacía ahí **no es evidencia** de que todos los turnos se
  oyeron.

Va en `livekit_extra.arnes` y la llave **solo aparece si llegó alguna marca**:
en las 15 llamadas de la campaña no hay arnés, y un bloque vacío ahí invitaría a
leerlo como captura fallida. `latencias` no cambia y `veredictos_fin_de_turno`
sigue vacío por contrato.

**Y el mismo discriminante se aplicó a PoC1**, sobre sus 13 llamadas ya
corridas: 0 de 166 turnos de régimen caen por debajo de su `stop_secs` (mínimo
371 ms contra un umbral de 200 ms). Los dos ceros alinean sobre el mismo evento
físico, así que el desglose por etapa se compara directo. Los números y el
residual conocido están en
[`docs/CONTRATO_METRICAS.md` §2.2.1](../docs/CONTRATO_METRICAS.md).

### B. `transcription_delay` puede ser 0 legítimamente

LiveKit documenta que vale 0 cuando la transcripción ya estaba lista. En este
esquema `0.0` y `None` **no** son lo mismo: `None` es "no se midió" y se excluye
del percentil; `0.0` es una medición real. Un cero de LiveKit se escribe como
`0.0`, nunca se convierte a `None`.

Un matiz que sí importa al leer los percentiles: LiveKit **clampea a cero**, no
reporta negativos —`max(last_final_transcript_time - last_speaking_time, 0)` en
`_compute_end_of_turn_metrics`—, así que un `0.0` puede ser "ya estaba listo" o
"salió negativo y se aplastó". No se pueden distinguir. PoC1 sí los distingue:
manda el negativo a `None` con el motivo en `notas`. Es una asimetría menor —solo
afecta a la etapa de STT, no a `turno_completo_ms`— pero conviene tenerla
presente si el p50 de `stt_ms` de PoC2 sale sospechosamente bajo.

### C. `llm_ttft_ms` no está anclado igual

PoC1 mide desde el **fin de turno** hasta el primer token, así que incluye lo que
tarde el pipeline en arrancar el LLM. `llm_node_ttft` se mide **dentro** del nodo
LLM, desde que el nodo arranca. Es la misma etapa conceptual, pero PoC2 puede
salir sistemáticamente algo más bajo.

No se corrige inventando un anclaje: se mide el residual (verificación 4) y, si es
material, se reporta como nota del desglose. `turno_completo_ms` —la métrica
principal— no se ve afectada, porque va de punta a punta.

### D. `e2e_latency` — RESUELTO: se usa tal cual

PoC1 cierra la muestra en el **primer byte de audio que produce el TTS**
(`TTSAudioRawFrame`), antes del transporte. La duda era si `e2e_latency` de
LiveKit llegaba más lejos, hasta que el audio suena, y había que restarle
`playback_latency` para emparejarlas.

No hace falta. En el SDK, `e2e_latency = started_speaking_at −
user.stopped_speaking_at` (`voice/agent_activity.py`), y `started_speaking_at` se
estampa cuando se resuelve `first_frame_fut` del `AudioOutput`: el primer marco
de audio **forwarded**. Es el mismo punto del pipeline que PoC1.

Y `playback_latency` está documentado en `llm/chat_context.py` como *"near-zero
for the default room output"* — solo se vuelve significativo cuando hay un avatar
remoto en la cadena, que no es nuestro caso. Se guarda igual en los extras, para
poder demostrar que fue ~0 en vez de suponerlo.

Que los dos extremos sean del worker es lo que hace comparable a PoC2 con PoC1, y
es también lo que impide leer el resultado como la espera del transportista:
los dos tramos de transporte quedan fuera por definición. Está medido y explicado
en [`turno_completo_ms` subestima lo que el transportista espera](#turno_completo_ms-subestima-lo-que-el-transportista-espera).

### E. `interrumpido` se define por el criterio de PoC1, no por el del SDK

El campo significa **barge-in**: el transportista habló encima del agente. No
significa "el SDK decidió cortar la respuesta". Así que el criterio se
reimplementa igual que en PoC1 —el usuario empieza a hablar mientras el estado del
agente es `speaking`— y lo que el SDK aporte de más (`InterruptionMetrics`,
`agent_false_interruption`) se guarda como extra.

Importa porque `interrumpido` decide a qué cubeta va la muestra, y con ~8 turnos
por llamada mover una muestra de cubeta mueve percentiles visiblemente.

### F. El turno de apertura hay que instrumentarlo a mano

En una llamada saliente el agente habla primero, así que su saludo no tiene
mensaje de usuario antes y **no existe un `e2e_latency` para él**. Es el mismo
problema que PoC1 resolvió con `colector.iniciar()`.

En PoC2: `t0` = la llamada fue contestada (participante SIP conectado), `t1` =
primer audio del agente. Se construye con `MuestraDeTurno.desde_eventos(0, ...)` y
`es_apertura=True`, para que caiga en la cubeta `apertura` y no ensucie el p50 de
régimen. Es código propio, no un campo del SDK — igual que en PoC1.

### G. Los veredictos de fin de turno no existen en PoC2

El audio turn detector no publica su probabilidad: solo acepta
`unlikely_threshold` como entrada. `veredictos_fin_de_turno` queda `()`.

`shared/metrics/` ya lo contempla (no entra en ningún percentil), así que no hay
que cambiar nada ahí. Lo que sí cambia es **cómo se calibra el turn-taking en
PoC2**: con las perillas (`min_delay`, `max_delay`, `unlikely_threshold`) y con
las notas de la rúbrica tomadas **en vivo** — no leyendo veredictos del JSON,
que es como se hace en PoC1. Y como PoC2 no graba, no hay a qué volver después:
la nota se toma durante la llamada o no se toma.
Por qué eso es un resultado y no un hueco:
[`docs/CONTRATO_METRICAS.md` §2.3](../docs/CONTRATO_METRICAS.md).

---

## Lo que LiveKit da de más

`on_user_turn_completed_delay`, `playback_latency`, `started_speaking_at`,
`e2e_latency` crudo, `InterruptionMetrics`, y la versión del turn detector que
acabó corriendo.

**No entran en `latencias`.** Van en una llave aparte del JSON por llamada
(`livekit_extra`), por dos razones:

1. `MuestraDeTurno.desde_dict` **ignora las claves que no conoce**. Meterlas
   dentro de `latencias` las tiraría en silencio, dejando la falsa sensación de
   que se están midiendo.
2. Las columnas del CSV son fijas y se cruzan entre las 3 PoC. Una columna que
   solo existe en PoC2 no es comparable con nada.

Sirven para diagnóstico y para explicar diferencias, no para el reporte.

### `fragmentos_usuario`: por qué se guardan TODOS los fragmentos del turno

Una intervención del transportista suele llegar partida en varios
`ChatMessage`. Cada uno es un commit de fin de turno que no disparó respuesta
—LiveKit marca `skip_reply`, o clasifica el habla como backchannel
(`audio_recognition.py`, `_EndOfTurnInfo`)— y **cada uno trae sus propias
`started_speaking_at` / `stopped_speaking_at`**.

Por eso `livekit_extra.por_turno[i]` lleva dos cosas distintas:

| Llave | Qué es | Para qué |
|---|---|---|
| `usuario` | el **último** fragmento anclado | es el cero del turno, del que cuelgan `fin_de_turno_ms` y `stt_ms` |
| `fragmentos_usuario` | **todos** los fragmentos, en orden y crudos | reconstruir el silencio *dentro* del turno, restando consecutivos |

El silencio intra-turno sale de `f[n+1].started_speaking_at −
f[n].stopped_speaking_at`. **Es captura, no métrica**: hoy no se calcula nada y
`MuestraDeTurno` sale idéntica. Se guarda ahora porque después no se
reconstruye — es justo el dato que a PoC1 le falta, y por el que su escenario 5
no se puede releer.

Dos detalles que importan al leerlo:

- Los fragmentos **sin ancla** también se guardan, crudos. Si se tiraran, el
  hueco entre los que sobrevivieron se leería como una sola pausa cuando en
  realidad hubo habla adentro. Se reconocen porque les falta la llave.
- Si la llamada se corta con el transportista a media intervención, esos
  fragmentos se vuelcan con `cerrar_pendientes()` en una entrada con
  `"indice": null` y `"sin_respuesta_del_agente": true`. No tiene muestra
  asociada, a propósito: el agente nunca contestó, así que no hay nada que medir.

Cuando llegue el momento de calcularlo (post-hoc, tras las 15 llamadas), va
etiquetado como **métrica exclusiva de PoC2 y covariable del WER** —explica los
cortes del agente en la autocorrección a media frase— **no** como comparación de
turn-taking contra PoC1. Es el espejo de `veredictos_fin_de_turno`, con el mismo
trato que ya fija [`docs/CONTRATO_METRICAS.md` §2.3](../docs/CONTRATO_METRICAS.md).

---

## Qué queda para la llamada de diagnóstico

Lo verificable sin llamada ya está cerrado contra el paquete instalado:

- [x] **Nombres de `ChatMessage.metrics`** — confirmados en `llm/chat_context.py`
      (`MetricsReport`).
- [x] **Dónde ancla `stopped_speaking_at`** — en el fin de habla real (punto A).
- [x] **Alcance de `e2e_latency`** — mismo punto que PoC1 (punto D).
- [x] **Selección de versión del detector** — fijada en `.env`, ya no depende de
      cómo se lance el worker.
- [x] **Endpointing del STT** — PoC1 y PoC2 corren cada uno los defaults de su
      plugin. No coinciden; se anota como asimetría en vez de emparejarse a mano.

Lo que **solo** una llamada real puede cerrar, y que el JSON ya trae listo para
leer:

1. **El veredicto del cero**, en `livekit_extra.veredicto_del_cero`. Tiene que
   salir `anclado_en_fin_de_habla_real`. Si sale lo otro, hay que restar y
   recalcular.
2. **Qué versión del detector corrió de verdad**, en
   `livekit_extra.turn_detector`. Se registra al empezar y al terminar. Con
   `v1-mini` no aplica el fallback en caliente, pero el registro atrapa el caso
   de que el `.env` y lo que corrió no coincidan — que es justo lo que pasó al
   descubrir que la campaña iba a correr sobre otro detector del documentado.
3. **Residual del desglose:** `e2e_latency − (end_of_turn_delay + llm_node_ttft +
   tts_node_ttfb)`. Lo que quede sin atribuir es encolado entre etapas. Todos los
   sumandos están en `livekit_extra.por_turno`.
4. **¿El texto del agente es lo pronunciado o lo generado?** PoC1 anota después
   del TTS a propósito, para no registrar como dicho un monto que el LLM generó y
   una interrupción se comió. LiveKit sustituye el contenido por el transcript
   **sincronizado** cuando hay interrupción y el TTS provee alineación
   (ElevenLabs la provee, `sync_alignment=True` por default), así que debería
   coincidir — pero hay que verlo con un barge-in real.

---

## Por qué cerró cada turno: confianza o vencimiento

`fin_de_turno_ms` es **bimodal**, y sin saber por qué un p50 mezcla dos
fenómenos distintos. En las 15 llamadas de la campaña, 103 turnos de régimen:

| `fin_de_turno_ms` | turnos | |
|---|---|---|
| 500-749 ms | 90 (87.4%) | el detector estuvo seguro |
| 750-1499 ms | 4 (3.9%) | |
| 2250-2749 ms | 9 (8.7%) | el detector **no** se convenció |

No son dos poblaciones de llamadas: son dos caminos del mismo código.
`audio_recognition.py::_bounce_eou_task` arranca con la espera corta y solo la
alarga cuando la probabilidad del detector no alcanza su umbral:

```python
endpointing_delay = self._endpointing.min_delay          # 0.3 s
...
if end_of_turn_probability < unlikely_threshold:
    endpointing_delay = self._endpointing.max_delay      # 2.5 s
```

Medido en vivo el 14 de septiembre de 2026 (umbral **0.35**, `min_delay` 0.3 s,
`max_delay` 2.5 s), la correlación es exacta:

| turno | `fin_de_turno_ms` | motivo | probabilidad | espera usada |
|---|---|---|---|---|
| 1 | 576.0 | confianza | 0.499 | 0.3 s |
| 2 | 1,111.5 | confianza | 0.471 | 0.3 s |
| 3 | 2,497.8 | **vencimiento** | 0.295 | 2.5 s |
| 4 | 1,041.0 | confianza | 0.361 | 0.3 s |
| 5 | 2,494.1 | **vencimiento** | 0.298 | 2.5 s |

O sea que el modo lento **no es ruido de medición**: es el detector diciendo
"esta persona no terminó de hablar" y equivocándose. Cuál de los dos caminos
tomó cada turno queda en `livekit_extra.por_turno[i].fin_de_turno`, y el conteo
agregado en `livekit_extra.fin_de_turno`.

**Se captura del log, no de una API.** LiveKit no expone la probabilidad al
código: no hay evento ni callback, y `EOTInferenceMetrics` solo trae duraciones
y además la emite el camino remoto, que PoC2 no usa. Lo único que publica es un
`logger.debug("eot prediction", extra={…})`. Capturarlo con un filtro de logging
es **observar**, no parchar — no se toca el SDK, ni la decisión, ni ninguna
constante congelada. Detalle en [`app/fin_de_turno.py`](app/fin_de_turno.py).

`veredictos_fin_de_turno` **se queda vacío**: la asimetría declarada en
`docs/CONTRATO_METRICAS.md` sigue en pie y el contrato no se toca, así que las
15 llamadas ya corridas siguen siendo comparables. Se puede apagar con
`POC2_CAPTURAR_FIN_DE_TURNO=0`.

---

## La forma del JSON por llamada

Idéntica a la de PoC1, para que
`python -m shared.metrics.cli calcular --carpeta poc2_livekit/grabaciones`
funcione sin tocar nada:

```json
{
  "call_sid": "<id de la llamada>",
  "poc": 2,
  "escenario": 2,
  "repeticion": 1,
  "resumen": { "...": "resumen de la sesión de negociación, sin precio_maximo" },
  "transcripcion": [
    {"t": 0.0, "rol": "agente", "texto": "..."},
    {"t": 4.2, "rol": "transportista", "texto": "..."}
  ],
  "latencias": [
    {"indice": 0, "turno_completo_ms": 2100.5, "fin_de_turno_ms": null,
     "stt_ms": null, "llm_ttft_ms": 900.0, "tts_ttfb_ms": 810.5,
     "interrumpido": false, "es_apertura": true,
     "veredictos_fin_de_turno": [], "notas": []}
  ],
  "livekit_extra": { "...": "playback_latency, e2e crudo, versión del detector" },
  "fallos_tecnicos": [
    {"t": 66.4, "tipo": "llm_error", "recuperable": true, "proveedor": "openai",
     "modelo": "gpt-4.1-mini", "detalle": "Request timed out."}
  ]
}
```

Cuatro detalles que no son cosméticos:

- **`poc`, `escenario` y `repeticion` al nivel de arriba.** Es donde
  `shared.metrics.llamadas.identificar()` los busca primero, antes de caer al
  archivo de referencia. En PoC1 viajan por el TwiML; en PoC2 viajan en el
  `metadata` del dispatch del agente (ver el [README](README.md)).
- **`rol` vale `agente` o `transportista`**, los mismos literales que usa
  [`shared/metrics/referencia.py`](../shared/metrics/referencia.py).
- **El archivo se llama `llamada_<id>.json`**, que es el glob de `listar_llamadas`.
- **`fallos_tecnicos` es la familia "confiabilidad" del contrato**, y se llena
  desde el evento `error` de la `AgentSession`. Se guardan **todos**, incluidos
  los recuperables: un reintento que salió bien no rompió la llamada, pero le
  costó segundos de silencio al transportista y ensució la latencia de ese
  turno. La muestra afectada lleva además una nota en `notas` — no se borra ni
  se anula, porque el número que midió LiveKit sigue siendo cierto; lo que deja
  de ser cierto es que describa al framework. El caso que lo motivó, con sus
  40.4 s, está en [`docs/CONTRATO_METRICAS.md` §1 y §2.4](../docs/CONTRATO_METRICAS.md).

El CSV y el reporte de PoC2 salen con las **mismas columnas** que PoC1 —sin nada
del resultado de la negociación— por la decisión del 5 de agosto de 2026
([`docs/CONTRATO_METRICAS.md` §4](../docs/CONTRATO_METRICAS.md)). El motor corre
como vehículo de la conversación; no aparece en los entregables.
