# Contrato de métricas — qué se iguala entre las 3 PoC y qué no

Este documento existe para que el comparativo sea defendible. Una comparación
entre frameworks de voz solo significa algo si queda explícito **qué se mantuvo
idéntico** (para que la diferencia medida sea del framework) y **qué se dejó
distinto a propósito** (porque igualarlo mediría una configuración que nadie
usaría en producción).

Las asimetrías no son defectos de la medición. Son parte del resultado, y por
eso se documentan aquí antes de correr, no después de ver los números.

**Antes de citar cualquier percentil de este estudio, leer §2.9.** Tres de las
distribuciones medidas resultaron bimodales, y en las tres el p50 describía un
turno que no existe.

---

## 1. Lo que se congela idéntico en las 3

Si alguna PoC corriera con algo distinto de esta lista, la comparación mediría
el prompt o el modelo en vez del framework de voz.

| Qué | Valor | Dónde vive |
|---|---|---|
| System prompt | `v0.md`, sin una coma de diferencia | `shared/system_prompt/v0.md` |
| LLM (PoC1 y PoC2) | `gpt-4.1-mini` | `OPENAI_MODEL` en `.env` |
| TTS | ElevenLabs Flash v2.5, misma voz | config de cada PoC |
| Guion de pruebas | 5 escenarios × 3 repeticiones, mismas líneas | `shared/test_scenarios/escenarios.yaml` |
| Datos del viaje | mismos origen, destino, carga y configuración | `app/trip.py` de cada PoC |
| Telefonía | Twilio, mismo número, mismo destino — **solo en las 15 llamadas de agosto**; las campañas de arnés corren sin telefonía, por WebSocket local | `.env` |
| Cálculo de métricas | el mismo módulo, sin ramas por PoC | `shared/metrics/` |
| Referencia del WER | el **guion actuado** en las llamadas telefónicas; el **guion reproducido** en las del arnés, derivado del clip y sin corrección manual (**§2.10**) | `shared/metrics/referencia.py`, `referencia_de_clips.py` |
| Timeouts y reintentos del LLM | 5 s al primer byte, 5 s de conexión, 4 intentos, 0 s de espera entre ellos | `shared/llm_transport/` |
| Ritmo con que el arnés entrega el audio | tramas de 20 ms contra deadline absoluto, con el temporizador del sistema en 1 ms | `shared/carrier_simulator/reproductor.py` |

**Por qué los timeouts del LLM entraron a esta tabla (7 de agosto de 2026).**
En la segunda llamada real de PoC2 una petición al LLM se atoró —la API la
aceptó y no mandó un solo byte— y el transportista se quedó **40.4 s** en
silencio. Eso no es una propiedad de LiveKit: es el proveedor del LLM
atorándose. Pero los defaults de los dos frameworks son distintos **y los dos
malos**, así que el mismo atorón habría costado cosas distintas en cada PoC y la
diferencia se habría leído como diferencia de framework:

| | PoC1 — pipecat (default) | PoC2 — LiveKit (default) |
|---|---|---|
| Tope del intento | ninguno propio: el del SDK de OpenAI, **600 s** | 5 s de lectura (~10 s por intento: cabecera y primer token) |
| Reintentos | **ninguno** (`retry_on_timeout=False`): turno perdido | 4 intentos |
| Espera entre intentos | — | 2 s (≈4 s muertos por atorón) |

Lo que se iguala es lo que se controla: el tope del primer intento, que haya
reintento en vez de turno perdido, y que no haya espera muerta. **El mecanismo
de reintento no se puede igualar** y se declara en §2.4.

**Por qué el ritmo del arnés entró a esta tabla (16 de septiembre de 2026).**
Con las llamadas manuales fuera del reporte, el estímulo de las 3 PoC lo entrega
el arnés, y el ritmo con que lo entrega es tan parte del estímulo como el audio.
Vive en `shared/` y no en cada PoC precisamente por eso: si PoC1 publicara parejo
y PoC2 a ráfagas, la diferencia entraría en sus latencias sin forma de separarla
después. Se fijó antes de correr ninguna campaña con arnés, que es cuando el
cambio es gratis. **No cambiarlo a media campaña**; si hubiera que hacerlo, se
cambia en las 3 y se repiten las llamadas ya hechas.

**Definición de "instante de envío", que es la que hace comparables las tres.**
Es **el instante en que el reproductor suelta la trama a su transporte**, tomado
en el bucle compartido (`Reproductor._bombear`) en la línea inmediatamente
anterior a `await publicar(...)`. No se toma después, y no es un matiz: lo que
hay después de ese punto es trabajo propio de cada transporte —armar el JSON y
el base64 y empujarlo al WebSocket en PoC1, construir el `AudioFrame` y
entregarlo a la `AudioSource` en PoC2— y en PoC2 `capture_frame` además
**bloquea** cuando su cola está llena, así que el instante quedaría gobernado
por el drenaje de un búfer de LiveKit en vez de por el reloj del arnés. Fecharlo
ahí metería el costo del transporte dentro del cero, que es justo lo que la
comparativa quiere medir aparte. Lo que el transporte tarda desde ese punto no
desaparece: aparece medido en el sesgo del cero.

**El cero cuelga del instante medido, no del deadline.** El reproductor guarda
los dos por trama (`shared/carrier_simulator/ritmo.py`) y fecha el inicio y el
fin del clip con el real. Importa porque el cero ancla en **una sola trama** —la
última del clip— así que su desvío no se promedia con nada: entraría entero en
la latencia de ese turno.

**El reloj.** El deadline se compara contra `time.perf_counter()`
(QueryPerformanceCounter, 100 ns) y **no** contra `time.monotonic()`, que en
Windows es `GetTickCount64` con 15.625 ms de granularidad: la resta salía
cuantizada y el bucle publicaba a ráfagas. Además se pide al sistema un
temporizador de 1 ms mientras dura la reproducción
(`resolucion_de_reloj_alta()`). Medido sobre 600 tramas, solo por cambiar de
reloj: σ del intervalo 6.7 → 0.68 ms y desvío máximo 27 → 2.6 ms. `time.time()`
no sirve para esto (misma granularidad y no es monótono) y solo se usa como
ancla única para traducir al eje de pared.

**Declaración, con los números que hay que citar.** Sobre 6,000 tramas y 120 s
de publicación continua contra un servidor en otro proceso:

| | |
|---|---|
| intervalo medio | 20.000 ms |
| duración real vs. nominal | 119.9790 s vs. 119.9800 s (**sin deriva acumulada**) |
| \|desvío\| contra el deadline: p50 / p90 / p95 | 1.03 / 1.97 / 2.21 ms |
| \|desvío\|: p99 / p99.9 / máximo | 11.1 / 18.2 / **112.4** ms |
| tramas con \|desvío\| > 10 ms | 62 de 6,000 (1.03%) |

Hay que leer esas dos filas como lo que son. El **típico** es de 1 a 2 ms, y la
cola es corta pero real: una trama de cada cien sale con más de 10 ms de retraso
y hubo una con 112 ms. Como el cero se fecha con el instante medido, **ese
desvío no entra en el cero** — entra en la regularidad con que el agente recibe
el audio, que es otra cosa y se vigila en el humo. Lo que queda dentro del cero
es solo el tramo que no se puede medir desde aquí: lo que el transporte tarda
entre que se suelta la trama y que el audio llega al agente, y eso se cuantifica
en el sesgo del cero, no se supone.

Es **común a las tres PoC** por construcción, así que no favorece a ninguna
arquitectura, y está por debajo del jitter de una llamada telefónica real, que
es lo que el arnés sustituye.

Se probó volver a dormir hasta alcanzar el deadline exacto y **empeora** (desvío
máximo 41 ms): cada espera corta cuesta una vuelta entera del bucle de eventos.
No insistir por ahí.

**El prompt v0 no cambia por decisiones de turn detection.** Lo que se ajusta
por framework es el componente de detección de turnos, nunca el texto que ve el
LLM.

**Ninguna PoC graba audio, y eso es parte de lo congelado.** La referencia del
WER la escribe quien actuó el escenario: el archivo se genera pre-rellenado con
las líneas del guion (`yo:`) junto a lo que entendió el STT (`stt:`), y solo se
corrigen las desviaciones. Como el método no toca el framework de voz, los WER
de las 3 arquitecturas son comparables entre sí. La contrapartida está en §2.3:
sin audio, lo que no se anotó en vivo no se recupera.

## 2. Lo que NO se iguala a propósito

### 2.1 La detección de fin de turno

Cada framework corre **su propio detector, con los defaults que recomienda su
fabricante**. Forzar una configuración común sería medir a un framework fuera de
cómo se despliega, y produciría un número que nadie podría reproducir en
producción.

| | PoC1 — Pipecat | PoC2 — LiveKit Agents | PoC3 — GPT-Live alpha |
|---|---|---|---|
| Detector | Smart Turn v3 local (ONNX, acústico) | `TurnDetector` — audio turn detector (acústico) | interno del modelo full-duplex |
| ¿Dónde corre? | **local**, en proceso | **`v1-mini`, local** en CPU (fijado en `.env`, ver abajo) | remoto, dentro del servicio del alpha |
| VAD | Silero, `stop_secs = 0.2 s` | Silero, `min_silence_duration = 0.55 s` (default de LiveKit) | VAD del servidor |
| Ventana de endpointing | 0.2 s + veredicto del modelo | 0.3 – 2.5 s | interna, no expuesta |
| ¿Expone la probabilidad al código? | **sí** (`TurnMetricsData`) | no | no |
| `veredictos_fin_de_turno` en el JSON | lleno | vacío | vacío |

**Decisión de PoC2 — se usa el audio turn detector, no el de texto.** LiveKit
ofrece dos: el acústico (`TurnDetector`, nuevo, el recomendado) y uno de texto
(`MultilingualModel`, **deprecado**, marcado para eliminarse en la 2.0). Se elige
el acústico por dos razones que se sostienen en el reporte:

1. Es el **análogo correcto** del Smart Turn v3 de Pipecat. Los dos deciden sobre
   la señal de audio, así que la comparación es entre mecanismos del mismo tipo.
   Comparar un detector acústico contra uno de texto mediría dos cosas distintas.
2. Es lo que un desarrollador elegiría hoy para un agente nuevo. Correr la
   comparativa sobre un componente deprecado sería indefendible.

**La versión del detector se fija a mano, no se deja en automático.** Sin
`version=`, `TurnDetector` resuelve `"v1" if (is_hosted() or is_dev_mode()) else
"v1-mini"` — es decir, la arquitectura dependería de **cómo se lanzó el worker**
(`dev` daría el modelo remoto, `start` el local). Eso es una base indefendible
para un benchmark, así que PoC2 lo fija en `.env`.

**Decisión del 7 de agosto de 2026: `LIVEKIT_TURN_DETECTOR_VERSION=v1-mini`,
el modelo local.** El motivo es de coherencia con lo que de verdad se está
midiendo: **PoC2 corre entero en local**, con el worker lanzado desde la
terminal y nada desplegado en LiveKit Cloud. Pedir `v1` metería un round trip a
LiveKit Inference en el camino de turn-taking que ninguna otra parte de la
arquitectura paga, y que PoC1 tampoco tiene porque su Smart Turn v3 corre en
proceso.

Con `v1-mini` los dos detectores acústicos quedan **local contra local**, así
que la diferencia de latencia que se mida es del mecanismo de detección y no de
dónde vive el modelo. Es un confound menos, no una concesión.

Lo que esto **no** significa es que la asimetría desaparezca: sigue viva en §2.3,
donde Pipecat expone la probabilidad del detector y LiveKit no. Esa es la
diferencia que el reporte tiene que sostener, y es de observabilidad, no de
despliegue.

Si algún día PoC2 se despliega en LiveKit Cloud, `v1` pasa a ser el valor
correcto — pero entonces hay que repetir las 15 llamadas, no cambiarlo a media
campaña.

**El fallback en caliente sigue vigilado.** El detector cae solo del modelo
remoto al local si el servicio de inferencia falla (`_fall_back_to_local`), y lo
hace en caliente: con `v1` una llamada podría empezar midiendo una cosa y
terminar midiendo otra. Con `v1-mini` ese modo de falla no aplica —ya se corre
local— pero la vigilancia se queda puesta, porque también atrapa el caso de que
el `.env` y lo que corrió no coincidan.

**Regla: toda llamada cuyo detector no sea el pedido, o cambie a media sesión,
queda fuera de la poza de percentiles.** El JSON la marca con
`excluir_de_pooled` y `shared/metrics/` la saca de los agregados, pero **no la
descarta**: sigue en el detalle por llamada, su WER sigue contando, y el
consolidado dice cuántas fueron y por qué.

Cada llamada registra en su JSON qué versión corrió de verdad, al empezar y al
terminar.

> **Revisado al escafoldar PoC2 (5 de agosto de 2026).** Los detalles de LiveKit
> —los dos detectores, el default de 0.55 s del VAD, la ventana de 0.3-2.5 s (que
> vive en `TurnHandlingOptions` de la sesión), el mínimo de 250 ms que exige el
> detector acústico y la eliminación del de texto en la 2.0— se confirmaron contra
> la documentación de LiveKit Agents 1.6.8. Lo que se corrigió de esta tabla fue la
> fila nueva: **el detector de LiveKit puede correr remoto**, y PoC1 corre local.
> (El 7 de agosto de 2026 se cerró esa fila en `v1-mini`, local — ver arriba.)
>
> **Actualización del 6 de agosto de 2026: ya está verificado contra el SDK
> instalado** (livekit-agents 1.6.8). Se resolvieron leyendo el paquete el
> anclaje del cero, el mapeo de las cinco etapas, la selección de versión del
> detector y el alcance de `e2e_latency`. Lo único que queda para la llamada de
> diagnóstico está en
> [`poc2_livekit/CAPTURA_METRICAS.md`](../poc2_livekit/CAPTURA_METRICAS.md), y el
> propio JSON de cada llamada trae el veredicto.

### 2.2 Cómo sobrevive la comparación de latencia a esa asimetría

Hay un efecto real que hay que nombrar antes de leer los p50: **LiveKit espera
0.55 s de silencio donde Pipecat espera 0.2 s.** Son ~350 ms de diferencia que
existen antes de que ninguna de las dos haga nada, y que no dicen nada sobre la
calidad del framework.

Se maneja con dos cosas que ya están en el contrato de captura:

- **El cero honesto.** El turno se mide desde que la persona **de verdad dejó de
  hablar**, no desde que el VAD lo confirmó. PoC1 lo consigue restando su
  `stop_secs` (0.2 s) al instante del frame.

  **PoC2 no resta nada, porque LiveKit ya lo hace.** Verificado en el SDK
  instalado (`livekit/agents/voice/audio_recognition.py`, rama
  `VADEventType.END_OF_SPEECH`):

  ```python
  speech_end_time = time.time() - ev.silence_duration - ev.inference_duration
  self._last_speaking_time = speech_end_time
  ```

  Ese `_last_speaking_time` es lo que se publica como `stopped_speaking_at`, y de
  ahí cuelgan `transcription_delay`, `end_of_turn_delay` y `e2e_latency`. O sea:
  los dos frameworks anclan en el mismo evento físico, cada uno por su cuenta.

  Restar 0.55 "para emparejar" habría inflado **toda** la latencia de PoC2 en
  medio segundo — tanto error como no restar cuando toca, solo que en la
  dirección contraria. La afirmación no se deja en la lectura del código: el
  colector la vuelve a verificar **con los datos de cada llamada**
  (`veredicto_del_cero`, en `poc2_livekit/app/latencia.py`), así que si el SDK
  cambia de criterio en una versión futura, el JSON lo grita.

### 2.2.1 El cero de PoC1, verificado contra sus propias llamadas

Antes de consolidar hay que comprobar que el cero de PoC1 cae en el mismo evento
físico, o el desglose por etapa no sería comparable. Se aplicó el **mismo
discriminante** que usa PoC2, sobre las 13 llamadas ya corridas (6 de agosto de
2026, 172 turnos de régimen, 166 con `fin_de_turno_ms`):

| | valor |
|---|---|
| `fin_de_turno_ms` mínimo | 371 ms |
| p25 / mediana / p75 | 419 / 434 / 482 ms |
| turnos por debajo de `stop_secs` (200 ms) | **0 de 166 (0.0%)** |

Si el cero estuviera en la confirmación del VAD, esos 200 ms de silencio ya
habrían transcurrido y habría turnos por debajo de la marca. No hay ninguno, y
el mínimo la dobla. **Veredicto: `anclado_en_fin_de_habla_real`.**

Confirmado además en el código: pipecat puebla el frame con el valor real del
analizador (`processors/audio/vad_processor.py`:
`stop_secs=_controller._vad_analyzer.params.stop_secs`), que es lo que
`app/latencia.py` resta. Si ese campo llegara en `0.0`, la resta no ocurriría y
todo esto se caería — por eso se verifica y no se supone.

**Los dos ceros alinean, así que el desglose por etapa se compara directo** y no
hace falta degradar el reporte a solo `turno_completo_ms`.

Queda un residual conocido, y se documenta en vez de corregirse. Es de unas
decenas de milisegundos: contra una mediana de `turno_completo_ms` de 2,606 ms
en esas mismas 13 llamadas es ~1%, un orden de magnitud por debajo de la etapa
más chica que se compara (`stt_ms`, mediana 387 ms), y sin efecto sobre ninguna
conclusión. De dónde sale, en el apartado que sigue.

> **Corregido el 17 de septiembre de 2026.** Este párrafo decía que el residual
> era que "LiveKit resta además `ev.inference_duration` y pipecat no". **Es
> falso**: `inference_duration` llega en `0.0` en el evento `END_OF_SPEECH` — el
> plugin de Silero no lo puebla y es el default del dataclass
> (`livekit/agents/vad.py:58`), medido en 6 clips. La resta de LiveKit es solo
> `silence_duration`. También afirmaba que las latencias de PoC1 salen
> marginalmente **cortas**; el signo no está establecido, porque los dos efectos
> que lo componen van en direcciones contrarias. Ver abajo.

#### 2.2.1.a La asimetría de mecanismo: quién corrige el cero y con qué número

Los dos ceros caen en el mismo evento físico, pero **no llegan ahí por el mismo
camino**, y esto hay que declararlo antes de comparar el desglose por etapa:

| | PoC1 (pipecat) | PoC2 (LiveKit) |
|---|---|---|
| Quién corrige | **la aplicación** (`app/latencia.py`) | **el SDK** (`audio_recognition.py:1425`) |
| Qué resta | `stop_secs`, el valor **nominal** del parámetro | `ev.silence_duration`, el silencio **realmente acumulado** |
| Cuánto es | 200 ms nominales | 576 ms medidos |
| Ventana del VAD | 512 muestras a 16 kHz = 32 ms | 512 muestras a 16 kHz = 32 ms |
| Espera real | `round(0.2/0.032)` = **6 ventanas = 192 ms** | 18 ventanas = **576 ms** |
| ¿La resta coincide con la espera? | **no**: resta 200, esperó 192 | **sí, por construcción** |

**La diferencia de fondo: LiveKit resta lo que midió y pipecat resta lo que se
configuró.** El VAD solo puede notar el silencio en frontera de ventana, así que
la espera efectiva siempre es un múltiplo de 32 ms. LiveKit publica ese múltiplo
y se lo descuenta, con lo que la cuantización se cancela sola. PoC1 descuenta la
constante del parámetro, que no es múltiplo de 32 ms, y le queda el desajuste.

De ahí salen los dos efectos del residual, **y tienen signos contrarios**:

- PoC1 resta 200 ms donde el VAD esperó 192: sobre-resta 8 ms, el cero cae
  **antes** del fin de habla real y la latencia sale ~8 ms **larga**.
- El fin de habla real cae en algún punto *dentro* de la ventana que abrió el
  silencio, hasta 32 ms antes de que se cuente: el cero cae **después** y la
  latencia sale hasta 32 ms **corta**.

Se cancelan parcialmente y ninguno domina, así que **el signo del residual no
está determinado analíticamente** y no se va a afirmar. Lo que sí hay ahora es
una medición que los incluye a los dos: §2.2.4, con el arnés.

**Por qué no se "arregla" restando 192 en vez de 200.** Porque cambiaría el
instrumento a media comparativa, con 13 llamadas de PoC1 ya corridas, para ganar
8 ms sobre una mediana de 2,606 — y porque la asimetría es real y es
información: **que un framework te descuente la espera medida y el otro te
obligue a descontar la nominal es una diferencia de lo que cada uno te da
hecho**, que es justo lo que esta comparativa mide. Se declara, no se empareja.

> **Hallazgo aparte, para PoC1 y no para el contrato:** de los 172 turnos de
> régimen, solo **87 tienen `turno_completo_ms`**. La otra mitad son turnos que
> el analizador cerró y el agente no llegó a contestar —el JSON lo registra como
> `None`, que es la verdad—, pero significa que la métrica principal se calcula
> sobre la mitad de los turnos. Vale la pena mirarlo antes de citar el p50.
- **El desglose por etapa.** La espera de endpointing queda aislada en
  `fin_de_turno_ms`. Las etapas posteriores —`llm_ttft_ms`, `tts_ttfb_ms`— no la
  arrastran, así que se pueden comparar directas.

Al leer el comparativo: `turno_completo_ms` incluye la política de endpointing de
cada framework (es la latencia que **percibe** el transportista, y por eso es la
métrica principal); `fin_de_turno_ms` dice cuánto de esa cifra es política de
configuración y no capacidad del framework. Las dos se reportan.

#### 2.2.1.b Tres conclusiones previas que quedaron corregidas (17 de septiembre de 2026)

Las tres estaban escritas en este contrato o en `poc2_livekit/CAPTURA_METRICAS.md`
y las tres se decidían por argumento indirecto. Quedan corregidas por medición, y
se dejan anotadas aquí **con lo que se creía antes**, porque un documento que
solo muestra la conclusión nueva no deja ver que el método anterior fallaba.

**1. La cola de la `AudioSource` son ~20 ms, no 200.**
Se decía: la `AudioSource` se crea con `queue_size_ms=200`, en régimen se
mantiene llena, y ese retraso cae entero dentro del sesgo de PoC2.
Es falso porque confunde un tope con una ocupación: **un tope solo se alcanza
empujando más rápido que el tiempo real**, y el reproductor empuja contra
deadline absoluto, o sea exactamente a tiempo real. Medido, la cola se queda en
**una trama (20 ms), con picos de 44**, un 22% del tope.
El testigo que lo cierra es la corrida con el tope puesto en **una sola trama**:
ahí `capture_frame` **sí** bloquea (15.6 ms de mediana contra 0.38 ms con el tope
de 200), o sea que el lado Rust retiene audio y aplica contrapresión de verdad.
Que con 200 ms no bloquee nunca es entonces evidencia de que ese buffer no se
llenó, y no un artefacto del modelo que el SDK lleva en Python. Tabla en §2.2.2.

**2. `min_silence_duration` sí se resta, y son 576 ms.**
Se decía: el sesgo de PoC2 supera los 550 ms de `min_silence_duration`, luego
LiveKit podría no estar descontando la espera de su VAD — y en ese caso habría
que restar medio segundo a **toda** la latencia de PoC2 en la comparativa.
El error estaba en la vara. `silence_duration` mide **576.0 ms exactos** en los 6
clips del banco (18 ventanas de Silero de 32 ms, no los 550 nominales) y el SDK
la resta entera. Y el argumento aritmético que lo cierra: **sin transporte de
ningún tipo, el sesgo sale en 170 ms** — imposible si los 576 no se restaran,
porque ninguna medición podría quedar por debajo de ellos.
El sesgo nunca fue `min_silence_duration + algo`; es
`desajuste de criterio + camino del audio`, y los 550 no eran un término de la
suma. **Resuelto: no hay que restar nada a la latencia de PoC2**, y
`veredicto_del_cero()` venía dando el veredicto correcto por un argumento
indirecto. Detalle en §2.2.2 y §2.2.3.

**3. "Silero pierde el final de algunos clips" era un bug del arnés, no del VAD.**
Se reportó que en el escenario 5 un clip había perdido sus últimos 2.1 s y que
eso era un hallazgo de turn-taking. **No lo era.** La causa era `SondaDelCero`,
que deduplicaba frames por `id(frame)`: CPython recicla el `id` de un objeto
liberado, así que un fin de habla posterior heredaba el de uno ya visto y se
descartaba. Con la clave correcta (`clave_de_frame`, que además une las dos
copias de un broadcast) los turnos perdidos pasaron de 2 en 5 corridas a **0 en
40 turnos**. No hay evidencia de que el VAD de PoC1 pierda finales de clip, y el
turn-taking de PoC1 **no** debe calibrarse con ese supuesto.

#### 2.2.1.c Lo que queda abierto: el remanente del sesgo de PoC2

Descontadas las piezas medidas —criterio, cola y SFU— del sesgo de PoC2 sobran
**~403 ms** en la corrida de las 16:41 y **~132 ms** en la de las 18:39
(desglose en §2.2.2). Dos cosas que hacen a ese remanente el número a vigilar:

- **Los 263 ms que separan las dos corridas caen enteros ahí.** El criterio es
  propiedad del clip y el transporte del canal de datos dio 66-68 ms en las
  cuatro corridas, así que no son ellos. (La cola sí se mueve — ver el aviso de
  §2.2.2 — y es ahora el primer candidato a mirar, no una constante.)

- **Y dentro de una misma llamada el sesgo crece.** La corrida del 17 de
  septiembre lo muestra por primera vez: descontadas las piezas medidas, el
  remanente va de 75-156 ms en T2-T4 (primeros 60 s) a 338-426 ms en T5-T7. No
  es solo un desplazamiento por llamada; hay una deriva **dentro** de la
  llamada, que es la firma que se esperaría de un buffer adaptativo creciendo.
- **El sospechoso es el jitter buffer de recepción**, que es adaptativo y puede
  asentarse distinto en cada llamada. **No está probado**, pero el cruce turno
  por turno (§2.2.2) ya eliminó a los dos competidores: no es el endpointing —la
  latencia de *inicio*, que no lleva resta de VAD adentro, crece igual (r =
  +0.894 contra el tiempo)— y no es el arnés, cuyo aporte total está acotado en
  ~48 ms típicos contra un crecimiento de ~650. También se había descartado la
  carga, cruzada contra `llm_ttft_ms` pareando por clip: r = +0.06 con n=6.
  **El remanente es entonces camino de red**, y el SFU es LiveKit Cloud, así que
  hay red que lo justifique.

**Se caracteriza con la campaña, no con más análisis sobre estas tres corridas.**
Cada llamada del arnés publica su bloque `arnes`, y con 15 el contraste tiene la
n para acotar cuánto se mueve el desplazamiento por llamada. Hasta entonces: la
n efectiva son las **llamadas**, no los turnos (la varianza entre llamadas domina
a la de dentro), así que el contraste se hace sobre la mediana por llamada.

Esto **no** bloquea la comparativa ni toca los números de latencia: el remanente
vive dentro del sesgo, que es diagnóstico, y el sesgo ya no se lee como evidencia
de que el cero de PoC2 esté mal anclado.

### 2.2.2 El sesgo del cero **no mide el mismo camino** en PoC1 y PoC2

El arnés publica el instante real en que soltó el último audio del turno, y cada
PoC lo resta de lo que su worker estimó:

    sesgo = cero_estimado_por_el_worker - fin_de_voz_del_arnés

El **minuendo** es comparable: el sello lo pone el arnés antes de entregar el
paquete (§1, "instante de envío"), así que lo que cueste la ruta de la marca —el
canal de datos en PoC2, un archivo en disco en PoC1— no lo mueve.

Lo que **no** es comparable es el tramo que el sesgo encierra, porque entre "el
arnés soltó la trama" y "el agente la oyó" cada arquitectura pone lo suyo:

| | PoC1 | PoC2 |
|---|---|---|
| Ruta del audio | WebSocket de Media Streams, directo | `AudioSource` → Opus → SFU → worker |
| Búfer propio antes del agente | ninguno | cola de la `AudioSource`, `queue_size_ms=200` |
| Ruta de la marca | archivo en disco, fuera de banda | canal de datos, en banda |

**Por lo tanto: los dos sesgos no se restan ni se comparan de frente**, por bien
anclado que esté el cero en las dos. Cada uno sirve para lo que fue diseñado —
decidir si el worker de *esa* PoC descuenta bien la espera de su VAD— y esa
pregunta se contesta por separado en cada arquitectura.

#### La cola de 200 ms está acotada: son ~20 ms, no 200 (17 de septiembre de 2026)

Se medía como pendiente y ya está medida, con
`python -m poc2_livekit.app.diagnostico_cero cola` — la `AudioSource` real, el
mismo FFI, sin sala y sin costo. 30 s de tramas de 20 ms contra deadline
absoluto, que es como publica el arnés:

| tope declarado | ocupación p50 | p95 | máx | % del tope | `capture_frame` p50 |
|---|---|---|---|---|---|
| 20 ms (una trama) | 0.0 | 27.0 | 40.0 | 200% | **15.6 ms** |
| **200 ms (el de la campaña)** | **20.0** | **20.0** | **44.0** | **22%** | **0.38 ms** |
| 1000 ms | 20.0 | 20.0 | 20.0 | 2% | 0.37 ms |

**`queue_size_ms` es un tope, no una ocupación.** Un tope solo se alcanza
empujando más rápido que el tiempo real, y en banco el reproductor empuja *a*
tiempo real: la cola se queda en una trama.

> ### ⚠ En llamada real sí se alcanza el tope (corrida del 17 de septiembre, 15:28)
>
> La primera corrida con esta instrumentación **contradice el banco**, y la
> conclusión de arriba solo vale para el banco:
>
> | | banco aislado | llamada real |
> |---|---|---|
> | ocupación mediana | 20 ms | **40 ms** |
> | ocupación p95 | 20 ms | **203 ms** |
> | ocupación máx | 44 ms | **208 ms (104% del tope)** |
> | desvío del arnés contra deadline, p95 | ~1 ms | **27.5 ms** |
> | desvío máx | ~2 ms | **207.6 ms** |
>
> **El mecanismo es el mismo que decía el banco, leído al revés.** La cola se
> llena empujando más rápido que el tiempo real, y en una llamada eso pasa: el
> bucle del arnés se atora —está además decodificando las 11,423 tramas del
> agente, midiéndoles RMS y atendiendo eventos de transcripción— y después
> recupera publicando en ráfaga contra el deadline absoluto. Esa ráfaga es la
> que llena la cola. El banco no se atora porque no hace nada más.
>
> **No es la grabación de auditoría.** Su costo por trama es un `append` de
> `(float, bytes)` con `como_pcm16` como identidad en PoC2, y el WAV se arma al
> colgar: no puede producir atorones de 200 ms. Queda descartada por
> construcción, no por medición.
>
> **Consecuencia para el desglose de abajo:** la fila de "cola de la
> `AudioSource` = 20 ms" es una **cota inferior**, y en los turnos donde el
> bucle venía de un atorón puede valer diez veces eso. El desglose se mantiene
> publicado porque las otras piezas no cambian, pero el reparto entre "cola" y
> "sin explicar" queda por decidir. Lo decide el cruce turno por turno — ver el
> apartado siguiente, que lo resuelve y **acota la cola en ~48 ms típicos**.

La fila de 20 ms es el testigo independiente, y hace falta porque
`queued_duration` es un modelo que el SDK lleva del lado de Python y podría no
corresponder con lo que hace Rust. Con el tope en una sola trama,
`capture_frame` **sí** bloquea (15.6 ms de mediana contra 0.38): el lado Rust
retiene audio y aplica contrapresión de verdad. Que con 200 ms no bloquee nunca
es entonces evidencia de que ese buffer no se llenó, no un artefacto del modelo.

En campaña esto ya no se supone: el arnés muestrea `queued_duration` por trama y
publica `entrega_del_audio.cola_de_la_fuente` en su bitácora, con
`uso_del_tope_pct`. Si alguna corrida se acercara a 100, el audio estaría
saliendo tarde y ese retraso caería dentro del sesgo del cero.

#### El cruce turno por turno: el atorón del arnés queda descartado (17 de septiembre de 2026)

El aviso de arriba dejaba dos sospechosos para la deriva del sesgo dentro de una
llamada: que **el arnés se fuera atorando** conforme avanza —en cuyo caso estaría
influyendo en lo que mide y habría que aligerarle el bucle— o que fuera el
**camino del audio** después de que el arnés suelta la trama. El cruce los
separa, y sale el segundo.

**El instrumento: la latencia de INICIO, no la de fin.** El sesgo de fin lleva
adentro la resta de `min_silence_duration` y el desajuste de criterio, así que
mezcla endpointing con transporte. El **inicio** no lleva ninguna de las dos:
entre que suena la primera muestra con voz del clip y que el VAD del worker
marca `started_speaking_at` solo hay camino. Descontándole a `sesgo_de_inicio` la
cabeza de silencio de cada clip —medida con el mismo criterio del manifiesto,
que reproduce el `fin_de_voz_medido` en vivo al milisegundo— queda una latencia
de inyección limpia:

| turno | t en la llamada | cabeza del clip | **latencia de inicio** | sesgo de fin |
|---|---|---|---|---|
| T1 | 0.0 s | 570 ms | **153.4 ms** | 401.2 ms |
| T2 | 15.3 s | 40 ms | **220.8 ms** | 408.5 ms |
| T3 | 28.3 s | 40 ms | **342.3 ms** | 332.5 ms |
| T4 | 45.1 s | 360 ms | **434.8 ms** | 366.1 ms |
| T5 | 57.3 s | 1,030 ms | **804.3 ms** | 639.1 ms |
| T6 | 72.7 s | 450 ms | **646.3 ms** | 824.2 ms |
| T7 | 90.4 s | 40 ms | **689.3 ms** | 690.3 ms |

La latencia de inicio **crece con la llamada** (r = +0.894 contra el tiempo) y
**acompaña al sesgo de fin** (r = +0.788). Como no tiene endpointing adentro, eso
ya resuelve la primera bifurcación: **la deriva es del camino, no del cero.**

La cabeza del clip no lo explica: está incorrelada con el momento de la llamada
(r = +0.023), y controlándola la pendiente queda en **+6.9 ms de latencia por
cada segundo de llamada** — unos 690 ms en 100 s, que es el orden del
crecimiento observado.

**Y el camino empieza después del arnés.** `inicio_del_clip` es el *deadline
nominal*, no el envío real, y el arnés honró ese deadline: desvío medio +7.6 ms,
p95 27.5 ms, máximo 207.6 ms, y **final +23.9 ms** sobre 115 s. Lo único que el
arnés pone entre el envío y el cable es la cola de la `AudioSource`, con mediana
40 ms y **tope duro de 200** que el lado Rust hace cumplir con contrapresión
real. O sea:

| | típico | techo posible |
|---|---|---|
| desvío contra el deadline | 7.6 ms | 207.6 ms (medido) |
| cola de la `AudioSource` | 40 ms | 200 ms (tope duro) |
| **total del arnés** | **~48 ms** | **~230 ms** |

Contra eso, la latencia de inicio tiene mediana **435 ms** y pico **804 ms**, y
crece **~650 ms** a lo largo de la llamada. El arnés no alcanza ni el pico ni el
crecimiento.

**Los percentiles bastan aunque no haya serie temporal.** T5-T7 ocupan ~47 s de
una llamada de 115 s, o sea el 41% de las tramas. Para que el arnés explicara
400-650 ms extra en ese tramo, la mediana de la cola tendría que estar en ese
orden; está en 40 ms, y el tope de 200 la hace imposible por construcción. No
hace falta ordenar las tramas en el tiempo para descartarlo.

> **La latencia de inicio sirve en PoC2 y NO en PoC1** (19 de septiembre de
> 2026). Corrida contra el arnés de PoC1, los siete turnos dan de −975 a
> +1,353 ms, con p50 de 974 — contra un sesgo de FIN de 141 ms en la misma
> llamada. Un valor negativo es imposible por construcción, así que la marca no
> es del clip que se le atribuye.
>
> La causa es la perilla, no el arnés: `SondaDelCero` toma **el primer** arranque
> de habla del turno, y el turno empieza en cuanto cerró el anterior. Con
> `stop_secs = 0.2` el VAD de pipecat se dispara en el hueco entre turnos —el
> ruido de confort basta— así que ese primer arranque cae mucho antes del clip.
> LiveKit no lo hace porque sus 0.55 s de `min_silence_duration` no se cruzan con
> ruido de confort, y por eso allá los siete turnos salen limpios.
>
> **Es la misma asimetría de §2.2.1.a vista por el otro lado**, y acota el
> instrumento: el discriminante de "camino contra cero" existe para PoC2 y para
> PoC1 hay que contestarlo de otra forma. `inicio_de_voz` aparta los turnos
> imposibles en vez de promediarlos, y si **todos** lo son se niega a publicar
> un p50. El sesgo de FIN de PoC1 no está afectado: es otro extremo del turno,
> lo toma el último arranque y no el primero, y su veredicto salió concluyente.

**Consecuencia operativa: no hay que aligerar el bucle del reproductor.** La
decodificación y el RMS del audio del agente no están compitiendo con la
publicación de tramas de ninguna forma que se note en la medición, así que el
`Reproductor` compartido se queda como está y el arnés de PoC1 se construye
sobre él sin cambios.

**Lo que sí queda como sospechoso, con evidencia circunstancial.** El SFU es
LiveKit Cloud (`*.livekit.cloud`), así que el audio hace un viaje redondo a
internet aunque el arnés y el worker corran en la misma máquina: hay red de
verdad en el camino, y un jitter buffer adaptativo tiene de qué adaptarse. La
grabación de auditoría lo respalda: el canal entrante tiene 51 huecos, pero 43
son silencio del agente (ceros exactos a los dos lados, supresión de silencio) y
solo **8 caen en medio de la voz** — y **7 de esos 8 se apiñan en 2.9 s**, entre
los segundos 62.07 y 64.94. El clip de T5 arranca en el 68.52. Es decir: la
única perturbación de red de toda la grabación ocurre 4 s antes del turno de peor
latencia, y después de ella la latencia nunca vuelve al piso de T1-T3. Eso es lo
que haría un buffer que crece ante jitter y decae despacio.

**Sigue sin probarse**: es un evento en una llamada. Lo que cambió es que el
arnés dejó de ser candidato, y el remanente de §2.2.1.c es ahora íntegramente
camino de red.

#### De qué está hecho el sesgo de PoC2, con las piezas medidas

Con la cola acotada y el piso del sesgo medido aparte
(`diagnostico_cero vad`, §2.2.3), el 620-735 ms se descompone así:

| pieza | corrida 16:41 | corrida 18:39 | cómo se midió |
|---|---|---|---|
| `sesgo_de_voz` p50 | 660.9 ms | 388.7 ms | el arnés, en la llamada |
| − desajuste de criterio | 170.0 ms | 170.0 ms | banco del VAD, sin transporte |
| − cola de la `AudioSource` | 20 ms | 20 ms | banco de la cola |
| − ida por el SFU | 67.9 ms | 66.2 ms | canal de datos de esa corrida |
| **= sin explicar** | **~403 ms** | **~132 ms** | Opus + jitter buffer |

Lo que queda es el camino del audio que no se puede medir desde este lado, y
**es justo donde vive la diferencia entre corridas**: los 263 ms que separan las
dos corridas del 15 de septiembre caen enteros ahí. Sigue apuntando al jitter
buffer de recepción, que es adaptativo y puede asentarse distinto en cada
llamada. Eso no está probado y con tres llamadas no se va a probar; lo que sí
está probado ahora es cuáles piezas **no** lo explican.

#### La pregunta abierta cambia de signo: LiveKit sí descuenta

Hasta hoy el razonamiento era: el sesgo supera los 550 ms de
`min_silence_duration`, luego LiveKit podría no estar descontando la espera de
su VAD, y en ese caso habría que restar medio segundo a toda la latencia de
PoC2. **Ese razonamiento está mal y la medición lo cierra**, por dos vías
independientes:

1. **La resta ocurre, y es de 576 ms.** En los 6 clips del banco,
   `silence_duration` mide **576.0 ms exactos** — 18 ventanas de Silero de 32 ms,
   no 550 — y `audio_recognition.py:1425` la resta entera. Además
   `inference_duration` llega en **0.0**: el plugin de Silero no lo puebla en el
   evento `END_OF_SPEECH` (`livekit/agents/vad.py:58`, es el default del
   dataclass), así que la corrección es solo `silence_duration`.
2. **Sin transporte, el sesgo sale en 170 ms.** Si los 576 no se restaran,
   **ninguna** medición podría salir por debajo de 576, y el banco da 122-345.

O sea que el sesgo nunca fue `550 + algo`: es `desajuste de criterio + camino del
audio`, y los 550 no eran un término de la suma. Compararlo contra ellos era
medir contra la vara equivocada.

**Consecuencia para la comparativa: no hay que restar nada a la latencia de
PoC2, y `veredicto_del_cero()` estaba dando el veredicto correcto.** Lo que se
corrige es el argumento —era indirecto, por magnitud— no el número.

### 2.2.3 El piso del sesgo: por qué ninguno de los dos llega a cero

Los dos ceros que se restan no usan el mismo criterio para decidir dónde termina
la voz, y esa diferencia aparece en el sesgo aunque el transporte fuera
instantáneo:

- **El arnés** fecha el fin de voz con `shared/audio/nivel.py::fin_de_voz`: la
  última ventana por encima de una caída de 25 dB bajo el pico. Es reproducible
  y viaja con su umbral, que es lo que se necesita de una medida.
- **El VAD** decide con su propia probabilidad, y no tiene por qué coincidir.
  Sobre estos clips no coincide: se queda oyendo voz durante la cola.

Medido en banco, sin sala, sin Opus y sin SFU
(`python -m poc2_livekit.app.diagnostico_cero vad`):

| clip | cola del clip | piso del sesgo |
|---|---|---|
| e1_t5 | 50 ms | 126.1 ms |
| e1_t4 | 70 ms | 122.7 ms |
| e1_t3 | 80 ms | 170.0 ms |
| e1_t2 | 100 ms | 166.0 ms |
| e1_t7 | 230 ms | 265.0 ms |
| e1_t6 | 230 ms | 345.3 ms |

**El piso sigue a la cola del clip**, que es lo que dice que el VAD la cuenta
como voz. Por eso `sesgo_de_clip` —contra el fin del archivo— es el número más
estable de los dos, y por eso se publican los dos: su diferencia es la cola, no
latencia.

Es una propiedad de los clips y del VAD, no del framework, así que **las 3 PoC
pagan un piso del mismo orden** y ninguna puede dar un sesgo cercano a cero por
bien anclada que esté.

### 2.2.4 El cero de PoC1, medido contra el instante real del arnés

§2.2.1 verificó el anclaje de PoC1 **por magnitud**, sobre `fin_de_turno_ms` de
13 llamadas. Con el arnés se puede verificar de frente, y sin gastar:
`python -m poc1_pipecat.app.diagnostico_cero` levanta el `/ws` de PoC1 con su
transporte, su serializer, su VAD y su analizador de fin de turno —todo local,
$0— y le habla desde el arnés con los clips de la campaña.

5 corridas (3 del escenario 1, 2 del escenario 5), **40 turnos**, ninguno
perdido:

| | p50 | q1 – q3 | rango |
|---|---|---|---|
| `sesgo_de_voz_ms` | **192.2** | 154.6 – 262.5 | 55.3 – 702.6 |
| `sesgo_de_clip_ms` | **48.7** | 23.0 – 99.7 | −97.5 – 142.2 |

**Veredicto: el cero de PoC1 está bien anclado, y su camino de inyección no
cuesta casi nada.** Los 192 ms de `sesgo_de_voz` son del orden del piso de
§2.2.3 (170 ms): el WebSocket de Media Streams, el serializer y el remuestreo a
16 kHz apenas agregan sobre el desajuste de criterio que pagaría cualquiera.
Contra el fin del archivo, que quita la cola, quedan **49 ms**.

**Esos 49 ms y los 528 ms del `sesgo_de_clip` de PoC2 no se restan** — es el
mismo punto de §2.2.2 con números. Lo que sí se puede decir, y es la conclusión
útil, es que la diferencia entre las dos **no es error de estimación del cero en
ninguna de las dos**: las dos anclan donde dicen anclar, y lo que las separa es
lo que cada arquitectura pone entre el arnés y el VAD del agente.

Dos límites de esta medición, declarados:

- **Es un piso.** El pipeline del banco no lleva STT, LLM ni TTS, porque no están
  en el camino del cero — pero sí cambian la contención de CPU. Una corrida con
  el pipeline completo puede subir el número, no bajarlo.
- **La frontera del turno la declara el arnés, no el analizador.**
  `TurnAnalyzerUserTurnStopStrategy` espera la transcripción antes de cerrar el
  turno, así que sin STT casi no cierra. El cero sigue saliendo del VAD sin
  tocar; lo único que decide el arnés es cuál de los fines de habla del clip es
  el del turno, y con 3 s de silencio entre clips no hay ambigüedad.

#### El pipeline completo lo contesta, y el primer límite estaba mal (19 de septiembre de 2026)

`python -m poc1_pipecat.app.arnes --escenario 1` corre el pipeline **de
producción** —STT, LLM, TTS y tool calling— contra el mismo guion, por un
WebSocket local y sin Twilio. Escenario 1, 7 turnos, acuerdo a 18,000:

| | banco ($0, 40 turnos) | pipeline completo (7 turnos) |
|---|---|---|
| `sesgo_de_voz_ms` p50 | 192.2 | **140.7** |
| rango | 55.3 – 702.6 | 93.1 – 293.4 |
| `sesgo_de_clip_ms` p50 | 48.7 | 60.9 |
| veredicto | anclado en fin de habla real | **anclado en fin de habla real** |

**El número bajó, y "es un piso" era falso.** No porque la contención no exista,
sino porque **los dos límites eran el mismo**: el banco no tiene STT, así que el
analizador no cierra turnos y `ceros_por_ventana` elige el último fin de habla
de la ventana del clip. Con el pipeline completo el analizador sí cierra, el
cero del turno es el que la campaña usaría de verdad, y sale mejor puesto. Los
192.2 ms del banco llevaban dentro el error de esa heurística.

**Lo que queda en pie del banco es el veredicto, no el número.** Sigue siendo la
forma de verificar el anclaje sin gastar, y las dos vías coinciden en que el
cero de PoC1 cae en el fin de habla real: 140.7 ms de mediana contra un
discriminante de 200 ms (`stop_secs`), esta vez con margen. La cifra citable del
sesgo de PoC1 es la del pipeline completo.

**Cobertura del cruce, que es lo que lo hace leíble:** el arnés escribió 8
marcas y declaró 8 (completa), 7 emparejaron y **ninguna salió con el final
perdido**. El mismo `call_sid` nombra los cinco archivos —el JSON del pipeline,
el `.jsonl` de marcas, el `cero_*.json` del cruce, la bitácora y el WAV— así que
juntarlos es un `join` por nombre.

**Y el cruce destapa algo que el JSON solo no dice:** para esos 7 clips el
pipeline publicó **13 muestras de latencia** — 1 de apertura, 6 con
`turno_completo_ms` y **6 sin número**. Con `stop_secs = 0.2` el VAD cierra
alrededor de dos turnos por clip, y desde el JSON no hay forma de saber cuáles 6
de las 13 corresponden a una intervención real. El arnés sí: él sabe cuántos
clips soltó y cuándo. Es evidencia para el pendiente de calibrar el turn-taking,
y una razón concreta por la que los percentiles de PoC1 no se leen sin su
cobertura al lado.

### 2.2.5 La unidad de análisis es la INTERVENCIÓN, no el turno del pipeline

**Decisión del 19 de septiembre de 2026, antes de la campaña. No re-litigar.**

Un **turno del pipeline** es lo que el VAD de cada framework decidió que era un
turno. Una **intervención** es una línea del guion soltada al cable. Hasta hoy
se contaba por turno del pipeline, y eso está mal por una razón que la primera
corrida de PoC1 dejó a la vista: **los dos frameworks no parten las
intervenciones igual**, así que contar por turno mide, sin decirlo, qué tan
finito corta cada VAD.

| | PoC1 (pipecat) | PoC2 (LiveKit) |
|---|---|---|
| intervenciones oídas | 7 | 7 |
| turnos que cerró el VAD | **11** | **7** |
| muestras de latencia publicadas | **12** | **7** |
| **fragmentación** (turnos / intervención) | **1.57** | **1.00** |
| fragmentación de muestras | 1.71 | 1.00 |

**El arnés es el único que puede contar intervenciones**, porque él soltó los
clips. Por eso la unidad vive en el arnés y no en el worker, y por eso el
agrupamiento vive una sola vez, en `shared/carrier_simulator/cruce.py::agrupar`:
si PoC1 agrupara de una forma y PoC2 de otra, la tasa dejaría de comparar.

**Cómo se agrupa.** Un evento del pipeline es de la **última** intervención que
empezó antes que él — misma idea que `emparejar`, por cercanía y no por índice,
con la diferencia que importa: aquí los eventos no compiten por su intervención.
Varios eventos en la misma ventana **son** el dato. Lo que cae antes del primer
clip (el saludo del agente dispara alguno) o mucho después del último se reporta
como `eventos_sin_intervencion` en vez de pegarse a la fuerza.

**La latencia de la intervención va de mi fin de voz al primer audio del
agente**, sin importar en cuántos eventos la haya partido el VAD. No se arma
sumando etapas y no depende del recorte de turnos, que es justo lo que la hace
sobrevivir a que un framework fragmente y otro no:

| | PoC1 | PoC2 |
|---|---|---|
| p50 | **2,280 ms** | **3,428 ms** |
| rango | 2,050 – 3,141 | 3,343 – 6,438 |

> ⚠ **Esos dos p50 todavía no son un veredicto de framework.** La medida incluye
> el camino de vuelta, y los dos despliegues no son el mismo: PoC1 corre entero
> en la máquina (WebSocket local) y el SFU de PoC2 es LiveKit Cloud, o sea un
> viaje redondo a internet en las dos direcciones. Parte de esos ~1,150 ms de
> diferencia es despliegue, no arquitectura. Se cita con esa salvedad hasta que
> la campaña dé n suficiente para separarlos.

**El denominador son las intervenciones que el pipeline oyó**, no las que el
arnés soltó. En el escenario 1 el agente cierra en T7 y T8 sale contra una
llamada muerta: contarla diluiría la fragmentación hacia 1.0 — o sea, haría ver
más parejo justo al framework que más corta.

#### La causa es la perilla, y no se iguala

La diferencia sale entera de dos constantes congeladas:

| | PoC1 | PoC2 |
|---|---|---|
| perilla | `stop_secs = 0.2` | `min_silence_duration = 0.55` |
| espera real del VAD | 6 ventanas = 192 ms | 18 ventanas = 576 ms |
| ¿un respiro a media frase corta el turno? | **sí** | no |

Con 192 ms de espera, cualquier pausa normal entre dos frases del transportista
cierra el turno; con 576 ms, no. **No se igualan las perillas.** Cada framework
corre su detección como su fabricante la recomienda, y esa diferencia es parte
de lo que se compara — es la misma decisión que §2.1 y la misma asimetría de
mecanismo de §2.2.1.a, vista ahora en el conteo.

**Por eso la fragmentación es una métrica comparativa por derecho propio y no un
artefacto de captura.** Dice cuántas veces por intervención el framework cree
que le tocó hablar, que en producción es cuántas veces puede interrumpir a
alguien que solo estaba tomando aire. Va al reporte junto a la latencia, no
escondida en una nota de cobertura.

**Consecuencia para los percentiles:** se calculan sobre intervenciones, no
sobre muestras del pipeline. Un p50 de PoC1 sobre sus 12 muestras promediaría
fragmentos con turnos completos y saldría más bajo de lo real; el mismo p50 de
PoC2 sobre 7 muestras no tendría ese problema. Compararlos sería comparar dos
cosas distintas con el mismo nombre.

### 2.2.6 De quién es la fragmentación: del estímulo, de la perilla, o de las dos

**El enunciado de §2.2.5 se corrigió el 20 de septiembre de 2026, y el número no
cambió.** La duda era legítima: si los cortes del VAD caen en las pausas
internas de los clips, la tasa mide cómo habló quien grabó el estímulo y no cuán
fino corta el framework. Se cruzó offline, con
`shared/metrics/fragmentacion.py`, y la respuesta es **las dos, en
papeles distintos**.

**Dónde cortó el VAD — el estímulo pone el hueco.** Los 9 cortes prematuros de
las dos corridas del 20 de septiembre cayeron, los 9, dentro de una pausa
interna del clip que sonaba. Ninguno cortó a media palabra.

**Cuántos hizo — la perilla decide.** Esos mismos clips ofrecían **27** pausas
de más de 192 ms, y solo **9 (33%)** se volvieron corte. El VAD no corta en cada
hueco: corta en uno de cada tres.

**El nivel de azar, sin el cual lo anterior no dice nada.** Estos clips tienen
tanto silencio interno que un punto cualquiera de su tramo hablado cae en una
pausa el **57%** de las veces. Que los 9 de 9 cayeran dentro sigue estando muy
por encima de eso (p ≈ 0.008 contra esa base), pero el número que hay que citar
es el 57% junto al 100%, nunca el 100% solo.

**El contrafactual, sobre los 49 clips del guion:** 156 pausas internas de más
de 200 ms repartidas en 48 de los 49 clips (p50 380 ms, max 1,500 ms). De esas,
**118 (76%) caen en la banda [192, 576) ms** — largas para que pipecat las tome
por fin de turno, cortas para que LiveKit no. Por escenario, la peor es la 5 (el
de estrés de WER): 50 pausas, 38 en esa banda.

| | pipecat | LiveKit |
|---|---|---|
| perilla configurada | `stop_secs=0.2` | `min_silence_duration=0.55` |
| cómo la cuantiza el SDK | `round(0.2/0.032)` = 6 tramas | acumula duración, `>= 0.55` |
| **umbral efectivo** | **192 ms** | **576 ms** |

**Las dos perillas se cuantizan en direcciones contrarias, y eso ensancha la
banda.** pipecat cuenta tramas con `round()`
(`pipecat/audio/vad/vad_analyzer.py`: `self._vad_stop_frames =
round(self._params.stop_secs / vad_frames_per_sec)`), así que 6.25 baja a 6 y
espera **menos** de lo configurado. LiveKit acumula duración real y dispara con
`>=` (`livekit/plugins/silero/vad.py`: `silence_threshold_duration +=
window_duration`), así que hacen falta 18 ventanas y espera **más**. La banda
que las separa no es [200, 550) sino [192, 576) ms. No es un detalle de
implementación que se pueda promediar — es el ancho exacto del desacuerdo.

**Cómo se enuncia en el reporte, entonces.** No "el VAD de pipecat corta más
fino", que atribuye al framework algo que la grabación provocó. Tampoco "es un
artefacto del estímulo", que lo descarta. Se enuncia como la pareja que es:
**sobre el mismo audio, con las perillas que cada fabricante trae por defecto,
pipecat cierra 1.57–1.62 turnos por intervención y LiveKit 1.00, porque 76% de
las pausas del habla real caen en la banda donde uno corta y el otro no.** Eso
se traslada a producción en la medida en que un transportista real haga pausas
de 200–576 ms, que es exactamente lo que hace.

**⚠ Este párrafo decía lo contrario hasta el 20 de septiembre de 2026 y estaba
mal.** Afirmaba que durante una pausa interna el cable llevaba clip **más** piso
de ruido, y lo daba como explicación de por qué solo un tercio de las pausas se
cobra. La medición de §2.2.9 lo desmintió: el arnés **alternaba** las dos
fuentes, así que durante el clip no había piso ninguno y las pausas internas
estaban **9 dB por debajo** de lo que una línea real habría dejado. El sesgo
apuntaba al revés del que se había declarado —empujaba al VAD a cortar, no a no
cortar— y no era común a las dos PoC.

**Las cifras de esta sección son de antes del arreglo y llevan ese artefacto
dentro.** Se conservan porque el método y el contrafactual siguen valiendo, pero
la tasa de 1.57–1.62 **no se cita sin decir que es de la etapa alternada**. La
tasa de la campaña se mide con el piso continuo y es la que va al reporte. Ver
§2.2.9 para qué se movió y qué no.

### 2.2.7 El camino de vuelta, medido: cuánto de la latencia es despliegue

**Sin esto la sección de latencia del reporte compara despliegues, no
frameworks**, y §2.2.5 ya lo advertía sin poder cuantificarlo. La latencia por
intervención se mide desde fuera y lleva tres cosas dentro:

    latencia = camino de ida + lo que tarda el pipeline + camino de vuelta

El de ida ya estaba acotado (`sesgo_de_voz`). El de vuelta se **mide** desde el
20 de septiembre de 2026, con las dos puntas en reloj de pared y resta directa.
No se despeja restando medianas: eso es una cota con tres incertidumbres dentro
y no sobrevive a que alguien pregunte por un turno concreto.

| | punta izquierda | de dónde sale |
|---|---|---|
| PoC1 | primera trama **con voz** que sale del pipeline | `SondaDeSalidaDeAudio`, entre el TTS y el transporte |
| PoC2 | `started_speaking_at` | lo publica LiveKit y **ya venía en el JSON**; no se estaba usando |

La punta derecha es la misma en las dos: `primera_voz_del_agente`, que el arnés
ya medía por intervención. Las dos son `time.time()` de procesos en la misma
máquina, así que la resta vale por la misma razón que el `sesgo_de_voz`.

**Medido en las corridas de validación del 20 de septiembre de 2026:**

| | camino de vuelta (p50) | rango | qué lleva dentro |
|---|---|---|---|
| **PoC1** | **5 ms** | 3–6 | serializar a μ-law, WebSocket local, búfer del cliente |
| **PoC2** | **202 ms** | 162–227 | encoder Opus, subida al SFU de LiveKit Cloud, bajada al suscriptor y su jitter buffer |

Sumado al camino de ida, el transporte a descontar es **~146 ms en PoC1** (141 +
5) contra **~610 ms en PoC2** (408 + 202). Sobre p50 de intervención de ~2,710 y
~3,430 ms, descontar transporte deja **~2,564 contra ~2,820 ms**: la mayor parte
de la diferencia que §2.2.5 no podía atribuir era despliegue.

**Las dos puntas tienen que medir el mismo evento, y la primera versión no lo
hacía.** La izquierda era "salió una trama" y la derecha "se oyó voz", así que el
arranque del TTS entraba dentro: un turno dio **879 ms** contra una mediana de 6.
Con la punta izquierda movida a la primera trama que supera `RMS_CON_VOZ` —el
mismo umbral con el que el arnés decide que oyó al agente— el atípico desaparece.

**⚠ PoC2 no puede hacer esa corrección**, porque su punta izquierda
(`started_speaking_at`) no distingue trama muda de trama con voz. Por eso PoC1
publica `arranque_del_tts_ms` aparte: es la magnitud del término que el número de
PoC2 lleva dentro y el de PoC1 no. En la validación fue 0 ms de mediana, así que
hoy no mueve nada; hay que vigilarlo con la campaña.

**Vigilado: es 0 ms en todas las corridas, no solo de mediana** (22 de
septiembre de 2026, 92 corridas de arnés de PoC1, 650 turnos). `salidas_audibles`
y `salidas_de_audio` salieron **listas idénticas en las 92**, así que el término
vale exactamente 0.0 ms en los 650 turnos: p50, p90, p99 y máximo, todos 0. O
sea que la primera trama que el pipeline de PoC1 suelta **siempre** cruza ya
`RMS_CON_VOZ`; ElevenLabs Flash no emite un arranque mudo a esta granularidad.
El término que PoC2 lleva dentro y PoC1 no está acotado en 0 ms, no solo
centrado en 0.

### 2.2.7.1 El acuse no está definido igual en las tres

Hay **dos** asimetrías en la definición del acuse, y se tratan por separado
porque son de naturaleza distinta:

- **Dónde se fecha respecto del transporte.** PoC1 y PoC3 fechan **antes** de
  entregarle la trama al transporte; PoC2 **después**. Es §2.2.7.2.
- **Qué cuenta como "con voz"**, que es lo que sigue en esta sección y que solo
  separa a PoC1 de PoC3.

**⚠ El encabezado de esta sección decía otra cosa hasta el 22 de septiembre de
2026, y estaba mal.** Afirmaba que *"las tres PoC reportan «hasta que la primera
trama con voz sale hacia el transporte»"*. Era **falso para PoC2** en las dos
mitades de la frase: su marca no es "hacia el transporte" sino ya dentro de él
(§2.2.7.2), y tampoco filtra por energía (la tabla de abajo). O sea que la frase
que encabezaba la declaración de asimetrías describía a dos de las tres.

Lo que sí es común a las tres —y lo que el reporte compara— es la **magnitud**:
fin de habla del transportista hasta que el agente empieza a sonar. Cómo se
obtiene cada punta es justo lo que estas dos secciones declaran.

La condición de "con voz", entonces:

| PoC | Qué dispara el acuse | ¿Filtro de energía? |
|---|---|---|
| 1 | primer `TTSAudioRawFrame` (`app/latencia.py:430-434`) | **no** |
| 2 | `started_speaking_at` de LiveKit — y además **del otro lado del transporte**, ver §2.2.7.2 | **no**, y no se puede agregar |
| 3 | primer `output_audio.delta` con alguna ventana de 20 ms ≥ `RMS_CON_VOZ` (`app/telefonia.py:523`) | **sí** |

En PoC1 el frame se emite ante cualquier payload de audio no vacío que devuelva
el proveedor, suene o no — `pipecat/services/elevenlabs/tts.py:919-921`:

```python
if msg.get("audio"):
    audio = base64.b64decode(msg["audio"])
    frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=received_ctx_id)
```

y `colector.primer_audio()` se llama sin mirar la energía.

**PoC3 no puede hacer lo mismo, y no es una elección de estilo.** El alpha
entrega una **portadora continua que incluye el silencio**: en la corrida de
calibración del 22 de septiembre de 2026, **491 de 861 deltas** de la llamada
eran relleno sin texto. Sin el filtro de energía, el acuse de PoC3 dispararía en
el primer delta de relleno —o sea al abrir la sesión— y no mediría nada. El
umbral es el que cierra esa brecha, y es el mismo `RMS_CON_VOZ` de las otras dos.

**Por qué la asimetría no mueve el número hoy.** El término que separa las dos
definiciones es el arranque mudo del productor, y en PoC1 está medido en **0.0 ms
en los 650 turnos** de sus 92 corridas de arnés (arriba). O sea que en PoC1 "la
primera trama" y "la primera trama con voz" son el **mismo instante**, y las dos
definiciones coinciden en la práctica aunque difieran en el texto. La asimetría
es documental, no numérica, **mientras el TTS no cambie**: si se sustituye
ElevenLabs Flash por un proveedor que sí mande un arranque mudo, el número de
PoC1 se desplaza y el de PoC3 no. Por eso `arranque_del_tts_ms` se sigue
publicando aparte en vez de darse por cerrado.

**Lo que esta asimetría NO cubre.** El acuse de PoC3 se fecha cuando el delta
sale hacia el transporte, y en ese instante suele haber audio del agente
**todavía por reproducir** por delante. Lo que mide ese encolado es el **cabezal
monótono de la grabadora de auditoría**, que corre cada bloque para no pisar el
anterior: el desfase del cabezal en el arranque de voz ES el audio pendiente.
Medido sobre los 24 arranques de voz de `CA221584…`: entre **1.9 y 53.9 ms
encolados por delante** (p25 10.8, p50 22.8, p75 41.6, media 25.8), y **nunca
cero**.

Eso no toca el acuse bajo la definición de §2.2.7 —lo que se fecha es la salida
hacia el transporte, no la reproducción— pero acota cuánto después lo oye el
transportista, y es el término que habría que sumar si alguna vez se reportara
"hasta que se oye". No se corrige: se caracteriza.

**En campaña el transporte no es Twilio.** `CA221584…` fue una llamada
telefónica y de ahí salen esos milisegundos, pero la campaña de PoC3 corre con
el arnés por **WebSocket local**, sin trunk y sin marcar. El encolado no
desaparece por eso —lo produce el cabezal, no la red— y la grabadora lo publica
por corrida en `arranques_de_voz.desfase_ms`, así que en la campaña se mide en
vez de heredarse. Es el mismo término de encolado que entra en el presupuesto
de δ del discriminante.

**`playback_latency` de LiveKit no es el camino de vuelta.** Cubre solo el búfer
local del worker **antes** de publicar: p50 de 3.05 ms sobre los 699 turnos de
la campaña. Los 202 ms del camino de vuelta están enteros del otro lado del
`capture_frame`, así que los dos números miden tramos distintos y contiguos, no
el mismo tramo con dos instrumentos. Sí es, en cambio, exactamente el término
que separa la definición del acuse de PoC2 de la de PoC1 y PoC3: ver §2.2.7.2.
Se cita en el reporte, si acaso, como evidencia de que el retraso **no** está en
el worker.

### 2.2.7.2 Dónde se fecha el acuse respecto del transporte: PoC2 va después

PoC1 y PoC3 sellan su marca **antes** de entregarle la trama al transporte;
PoC2 la sella **después**, ya dentro de su propia salida de audio. Es la
asimetría de definición más grande de las dos, y hay que declararla antes de
comparar los percentiles.

| PoC | Qué instante se sella | ¿De qué lado del transporte? | Dónde |
|---|---|---|---|
| 1 | primer `TTSAudioRawFrame` del servicio de TTS | **antes**: no pasó por el serializer ni por el WebSocket | `app/latencia.py:430-434` |
| 3 | primer `output_audio.delta` con voz, al llegar al puente | **antes**: no se remuestreó a μ-law ni se mandó | `app/telefonia.py:523` |
| 2 | `started_speaking_at` | **después**: la salida ya sacó la trama de su búfer y está por empujarla a la pista | ver abajo |

**Cómo se verifica en el SDK, sin creerle al nombre del campo.** LiveKit sella
dos instantes distintos y publica su diferencia:

- `started_forwarding_at` se sella en la **primera trama que el TTS le entrega a
  la salida de audio** (`livekit/agents/voice/generation.py:490-491`). Ése es el
  análogo exacto del punto de PoC1 y PoC3.
- `started_speaking_at` es el `created_at` del evento `playback_started`
  (`livekit/agents/voice/room_io/_output.py:197-200`), que la salida de la sala
  dispara **dentro de su bucle de reenvío**, justo antes del `capture_frame`
  que mete la trama en la pista.
- El propio SDK los resta:
  `playback_latency = started_speaking_at − started_forwarding_at`
  (`voice/agent_activity.py:2936-2938`, y otra vez en 3280 y 3401).

O sea que el término que separa la definición de PoC2 de la de las otras dos
**es** `playback_latency`, y no hay que estimarlo: LiveKit lo publica por turno
y el JSON de PoC2 lo guarda en `livekit_extra.por_turno[].asistente`.

**Qué incluye y qué no.** Incluye el búfer interno de la salida de audio del
worker y la espera de `_playback_enabled`. **No incluye nada de la red**: el
`AudioSource`, la subida al SFU, la bajada al suscriptor y su jitter buffer
están todos **después** del `capture_frame`, así que quedan fuera de las tres
definiciones por igual. Ese tramo es el que mide el camino de vuelta de §2.2.7
—5 ms en PoC1 contra 202 en PoC2—, y por eso no se cuenta dos veces.

**En qué dirección sesga.** El acuse de PoC2 cae **más tarde** que el de PoC1 o
el de PoC3 sobre el mismo evento físico, así que la asimetría **castiga a
PoC2**: le mete a su latencia un tramo que las otras dos no pagan dentro de su
marca. Si el reporte concluyera que PoC2 es la más lenta, esta asimetría apunta
en el mismo sentido que la conclusión — que es justo la razón por la que hay que
declararla. Un instrumento no puede aportar parte de la diferencia que el
reporte atribuye al framework.

**Cuánto vale, medido sobre la campaña que se compara** (100 llamadas de PoC2,
**699 turnos**, leyendo `playback_latency` de cada JSON):

| | p50 | p95 | p99 | máx |
|---|---:|---:|---:|---:|
| `playback_latency` | **3.05 ms** | 5.11 | 6.76 | **8.72** |

Ningún turno pasa de 20 ms. Comparado con el término de §2.2.7.1 —el arranque
mudo del TTS, que en PoC1 mide **0.0 ms exactos en los 650 turnos** de sus 92
corridas de arnés— es mayor, pero los dos son de orden de milisegundos sobre
percentiles de régimen de ~2,400 ms.

**Ninguna de las dos asimetrías de definición explica una brecha de centenas de
milisegundos**, y conviene decirlo con todas sus letras para que no se les
atribuya lo que no hacen. El acuse de PoC2 llega, como mucho, 8.72 ms tarde
respecto de dónde lo pondría el criterio de PoC1. Lo que sí separa a las dos por
centenas es el **camino de vuelta** de §2.2.7 (5 contra 202 ms) y el de ida
(141 contra 408 ms), y los dos ya se descuentan en la tabla de §2.2.8.

**Y no se puede igualar.** PoC2 no expone un gancho entre el TTS y su salida que
permita fechar ahí, así que `started_forwarding_at` es lo más cerca que se
llega, y no está publicado como llave propia del JSON. Restarle
`playback_latency` a cada muestra de PoC2 sería la corrección exacta y está al
alcance —el dato está guardado turno por turno— pero mueve el p50 **3 ms**. Se
declara y no se corrige: una corrección de 3 ms que obliga a recalcular la
campaña entera compra menos precisión de la que cuesta en trazabilidad.

### 2.2.8 Cómo se publica la latencia: tres cifras por PoC, nunca una

La latencia corregida **no se publica sola**. Cada PoC va al reporte con las tres
cifras en la misma tabla, en este orden:

| | latencia observada | transporte descontado | latencia corregida |
|---|---|---|---|
| | p50 de intervención, medido desde el arnés | ida + vuelta, con las dos partes desglosadas | observada − transporte |

El transporte se desglosa siempre en sus dos sumandos con su procedencia —el de
ida es `sesgo_de_voz` (§2.2.2), el de vuelta es la medida directa de §2.2.7—,
porque son mediciones distintas con métodos distintos y una puede envejecer sin
la otra.

**La razón de publicar las tres es de auditoría, no de estilo.** La cifra
corregida es una resta entre dos números que se midieron con instrumentos
distintos, y el ajuste es grande: en la validación descuenta 146 ms de PoC1
contra 610 de PoC2, o sea que por sí solo mueve la comparación en ~460 ms. Con
solo el resultado, nadie puede verificar de dónde salió el ajuste, ni rehacerlo
si mañana se remide una de las dos puntas, ni notar que el ajuste es del mismo
orden que la diferencia que pretende explicar. Un lector que solo vea la cifra
corregida no puede distinguir "los frameworks se parecen" de "el ajuste se comió
la diferencia".

#### Por qué se descuenta el transporte en vez de igualar los despliegues

La alternativa sería montar las dos PoC sobre la misma topología y no tener nada
que descontar. **No existe esa topología sin inventarla.** LiveKit Agents está
construido sobre su propio SFU: el worker se suscribe a una sala y publica en
ella, y no hay modo soportado de que el agente reciba y entregue audio sin ese
salto. Pipecat, en cambio, se conecta directo al WebSocket de Media Streams y no
tiene SFU que atravesar. Igualarlos exigiría o meterle a Pipecat un SFU que su
arquitectura no pide, o sacarle a LiveKit el suyo por un camino que su SDK no
ofrece. En los dos casos se mediría una configuración que nadie desplegaría.

Descontar respeta las dos cosas que el reporte tiene que decir a la vez:

- **La cifra observada es la verdad de uso.** Quien despliegue LiveKit Agents va
  a pagar su SFU, porque el framework lo requiere por diseño. Ese costo es real
  y no se esconde: por eso la observada va primero en la tabla.
- **La cifra corregida aísla lo que hace el framework de dónde vive el
  servidor.** El SFU de la validación está en LiveKit Cloud y el WebSocket de
  Pipecat está en `localhost`; buena parte de la diferencia es geografía, no
  arquitectura de software. Sin descontar, el reporte estaría comparando dos
  despliegues y llamándolo comparación de frameworks.

Las dos lecturas son legítimas y contestan preguntas distintas —"¿cuánto va a
tardar si lo despliego así?" contra "¿cuál framework hace más trabajo por
turno?"—, y el transporte desglosado en medio es lo que permite pasar de una a
la otra. **El reporte no elige una: publica las tres y dice cuál contesta qué.**

### 2.2.9 El piso de ruido va sumado por debajo del clip, no alternado con él

**Cambio del 20 de septiembre de 2026, aplicado antes de la campaña.** Hasta
aquí el arnés publicaba **dos fuentes alternadas**: ruido de confort entre clips
(`Reproductor.trama_de_relleno`) y el clip crudo durante el clip
(`partir_en_tramas`). No había una sola línea que mezclara el piso debajo del
clip. Medido sobre los 49 clips:

| | dBFS RMS | centroide espectral |
|---|---|---|
| ruido de confort, **fuera** del clip | −60.0 (−59.5 tras μ-law) | 2,007 Hz (uniforme, plano) |
| piso propio del clip, **dentro** | −69.4 | 1,180 Hz (pasa-bajos) |

O sea **un escalón de ~9 dB y de ~800 Hz de centroide en los dos bordes de cada
intervención** — exactamente los dos instantes que el detector de fin de turno
tiene que encontrar. 122 de las 156 pausas internas (78%) bajaban del piso
publicado en su punto más callado. Una línea telefónica no hace eso: el piso
está siempre y la voz se suma encima.

**Por qué había que arreglarlo antes de gastar en la campaña, y no después.** El
artefacto no es de modo común. Una pausa interna 9 dB más callada de lo que
debería empuja al VAD a declarar fin de turno ahí, y eso pega mucho más al que
cierra a 192 ms que al que espera 576. El escalón contaminaba a la vez la tasa
de fragmentación, el `sesgo_de_voz`, el `fin_de_turno_ms` y el hallazgo de Smart
Turn v3 sobre banda telefónica — y en la dirección que **castiga a pipecat**, que
es justo el término que se está comparando.

**Cómo quedó.** `cargar_clip_para_publicar` suma el ruido al PCM del clip, lo
codifica a μ-law y devuelve las dos representaciones decodificadas del mismo
byte stream, para que PoC1 y PoC3 (μ-law) y PoC2 (PCM) oigan **el mismo audio** y
no dos cuantizaciones del mismo audio. La semilla del ruido sale del sha256 del
clip: dos corridas publican bytes idénticos y dos clips no comparten la
realización del ruido.

- **Los 49 `.ulaw` de disco no se tocaron**, ni el manifiesto, ni sus hashes.
  `cargar_clip` sigue devolviendo el estímulo **limpio**, que es lo que hay que
  analizar: las pausas del clip son del clip.
- **La verificación de hash de PoC1 pasó a comparar el archivo fuente**
  (`PasoDelGuion.ulaw_fuente`) y no lo publicado, que por construcción ya no
  puede coincidir. Lo que la evidencia tiene que sostener sigue siendo lo mismo
  —las 3 PoC parten del mismo estímulo— y el piso es una función determinista de
  él. La bitácora imprime los dos hashes.

Medido después del arreglo, sobre los mismos 49 clips:

| | antes | después |
|---|---|---|
| punto más callado de la pausa interna (p50) | −69.4 dBFS | **−59.1** |
| lo mismo, p5 | −84.7 dBFS | **−60.1** |
| pausas por debajo del piso publicado | 122 / 156 | **14 / 156** |
| mínimo en los primeros 20 ms del clip | −81.5 dBFS | **−59.7** |
| mínimo en los últimos 20 ms | −77.3 dBFS | **−59.7** |

El p50 de la voz no se movió (el piso está 40 dB por debajo) y el censo de
pausas pasó de 156 a 157: una pausa larga se partió en dos al subir una ventana
por encima del umbral. La caracterización del estímulo se sostiene.

#### Cuánto se movió la fragmentación: nada, y está verificado con el propio Silero

El arreglo podía invalidar el 1.57, así que la pregunta se contestó **antes** de
la campaña y sin gastar: se corrió el **mismo `SileroVADAnalyzer` que usa
pipecat** (confianza 0.7, ventana de 32 ms, 8 kHz) sobre los 49 clips dos veces,
con el estímulo limpio y con el publicado, contando los huecos de no-voz
internos que superan el umbral efectivo de cada framework.

| huecos internos que Silero declara | limpio | publicado |
|---|---|---|
| ≥ 192 ms — pipecat cierra turno | **76** | **76** |
| ≥ 576 ms — LiveKit también | **13** | **13** |

**Idénticos.** El piso de −60 dBFS está muy por debajo del umbral de habla de
Silero, que es justamente por qué el mismo detector cierra turnos durante el
ruido de confort entre clips. Sumarlo dentro del clip no le quita ni le pone un
solo corte.

Lo mismo para los sesgos, que se anclan en el fin de voz: **el fin de voz que ve
Silero no se mueve** (p50 +0 ms sobre 49 clips, nunca más tarde, como mucho 32 ms
antes). El inicio sí se mueve en 19 de los 49 (hasta 544 ms antes), pero el
inicio no ancla ninguna métrica publicada.

**Conclusión para el reporte: el 1.57 no baja y las conclusiones que lo citan no
se corrigen.** El arreglo no era para mover ese número —era para que no
dependiera de un artefacto del arnés—, y ahora hay evidencia de que no dependía.
Lo que cambia es que se puede defender: antes la respuesta a "¿y si el escalón
está haciendo el corte?" era una suposición, y ahora es una medición.

**⚠ Marca de corte: las mediciones de antes del 20 de septiembre de 2026 llevan
el artefacto dentro y NO se mezclan con las de la campaña.** Esto incluye la
tasa de fragmentación de 1.57–1.62, los `sesgo_de_voz` de 141 y 408 ms, el
`fin_de_turno_ms` de las corridas de validación y las cuatro llamadas del lote
de prueba. No entran a ninguna poza de percentiles junto con las de la campaña,
no se promedian con ellas, y cualquier conclusión que las cite tiene que decir
que son de la etapa de fuentes alternadas. Las corridas nuevas se distinguen sin
ambigüedad: su `origen` dice `+ piso de ruido`.

### 2.3 Visibilidad del turn-taking

Pipecat publica la probabilidad de su detector por turno; LiveKit no la expone al
código de la aplicación. **Es un hallazgo legítimo del comparativo, no un hueco
de la medición.**

La asimetría es más grande de lo que parece, y conviene medirla como lo que es:
**una diferencia de observabilidad entre frameworks**, no una limitación de la
medición.

Ninguna PoC graba audio — la referencia del WER es el guion actuado (§1). Eso
deja el diagnóstico de un falso fin de turno con dos regímenes distintos:

| | PoC1 (Pipecat) | PoC2 (LiveKit) y PoC3 |
|---|---|---|
| Qué queda del turno | probabilidad del detector por turno, en el JSON | nada: el detector acústico no la expone al código |
| Cuándo se puede juzgar | **cuando sea**, post-hoc, sobre datos | **solo en vivo**, durante la llamada |
| Qué se puede distinguir | "el modelo se equivocó convencido" de "cerró por timeout" | solo que hubo un corte |
| Si nadie tomó nota | se recupera del JSON | se perdió |

Dicho de otra forma: Pipecat deja **evidencia** de sus decisiones de
turn-taking y LiveKit deja **una impresión**. Sin grabación no hay a dónde
volver en PoC2, así que la ventana para juzgar el turn-taking dura lo que dura
la llamada — de ahí que `naturalidad` y `turn_taking` se llenen al colgar y no
después (ver la rúbrica en `guion_escenarios.md`).

Eso cambia el costo de calibrar un agente en producción, que es exactamente el
tipo de diferencia que esta comparativa debería sacar a la luz: en Pipecat las
perillas del turn-taking se ajustan contra datos, y en LiveKit contra criterio
humano no reproducible.

En números, la asimetría se reporta así: `veredictos_fin_de_turno` lleno en PoC1,
vacío en PoC2 y PoC3. El campo es opcional en el contrato y **no entra a ningún
percentil**, así que su ausencia no deja huecos en el reporte — el hueco está en
lo que se puede *explicar*, no en lo que se puede *contar*.

### 2.3.1 El solapamiento se mide sobre el WAV de auditoría, y es turn-taking

`shared/metrics/solapamiento.py` cuenta la voz simultánea en los dos canales de
la grabación de auditoría. Separa dos cosas que una mezcla confundiría:

- **El agente entró encima del transportista.** Es la interrupción no deseada
  que §Métricas de CLAUDE.md pide contar, y describe al framework.
- **El transportista entró encima del agente.** Aquí el transportista es el
  arnés, que tiene regla de avance explícita —esperar a que el agente se calle—,
  así que esto **no** describe al framework: dice que la regla falló, y esa
  corrida hay que mirarla antes de usar su latencia.

**Es la única excepción a "la auditoría no entra a ninguna métrica", y está
acotada.** El WAV sigue fuera de latencias, del WER y de todo percentil, por la
razón de siempre: su canal 1 se fecha **cuando la trama llega**, con el jitter
del transporte dentro. Lo que sí soporta es una pregunta de coincidencia —"¿estos
dos sonaron a la vez medio segundo?"—, porque una interrupción real dura cientos
de milisegundos y el jitter son decenas. Por eso se publican **conteos** y
duraciones redondeadas a la decena de ms, nunca un p50 ni un p95.

**Dos guardas, porque el resultado esperado es cero y un cero engaña.**

1. **El umbral de actividad es la caída de recorte (35 dB), no la de nivel
   (25).** Con 25 dB el canal del arnés de la corrida del 19 de septiembre daba
   6.2 s de voz para ocho clips que suenan el triple; con 35 da 10.7 s, y el
   umbral (−36 dBFS) sigue 24 dB por encima del piso de ruido de confort (−60
   dBFS), que es lo que no puede contar como voz del transportista. Aquel umbral
   existe para *medir el nivel* de una grabación; aquí hace falta saber si
   **había alguien hablando**.
2. **Cada medición publica su cobertura** (segundos de voz detectados por
   canal). Un detector ciego en un canal produce cero solapes y se lee idéntico
   a una conversación perfectamente por turnos. Con la cobertura al lado se
   distinguen, y una corrida con un canal en 0 s se marca como no medible.

**Lo medido hasta ahora: cero.** En las 5 corridas de auditoría existentes y
medibles (2 de PoC1, 3 de PoC2), **ningún solape en ninguna dirección**, ni
siquiera bajando el mínimo a 0 ms. La lectura correcta no es "los frameworks no
interrumpen" sino que **el arnés, como está configurado, casi no puede producir
la condición**: espera a que el agente calle antes de avanzar, y el agente
contesta 2–3.5 s después de que el clip terminó de sonar. El caso en que sí
podría pasar existe y es el de §2.2.6 —el VAD cierra turno a media intervención,
y si el pipeline contestara rápido el agente hablaría sobre el resto del clip—;
con 2.3–2.7 s de latencia contra ~2.7 s de clip restante queda al filo. Por eso
la campaña graba auditoría: para ver si con 50 llamadas ocurre.

**Ocurrió, y no por donde se esperaba.** En la corrida de verificación
`spots-poc2-arnes-20260920-195644` el WAV marcó dos solapes en T4, y el que
importa es el segundo: **el arnés le habló encima al agente**. Es fallo de la
regla de avance del arnés, no del framework, y lo arregla §2.3.2.

### 2.3.2 `SILENCIO_DEL_AGENTE_SEG`: de 0.8 a 2.0 s

| | valor | de dónde salía |
|---|---|---|
| **antes** (hasta el 20/09/2026) | **0.8 s** | elegido a ojo, "más alto que el `stop_secs` del VAD" |
| **desde el 20/09/2026** | **2.0 s** | la distribución medida de pausas internas del agente |

Es lo que el arnés espera en silencio del agente antes de soltar su clip. **No
es una constante de la comparativa** y no se compara con ningún `stop_secs` ni
`min_silence_duration`: aquellos deciden un fin de turno que se mide, esta solo
decide cuándo al arnés le toca hablar. Pero sí decide qué corridas quedan
válidas, y por eso se fija con datos.

**Por qué 0.8 no alcanzaba.** Con ese valor el arnés le habló encima al agente en
**2 de 10** corridas con auditoría. El sesgo no es benigno: pasa cuando el agente
hace una pausa larga a media respuesta, o sea en las respuestas **largas**, que
son las que pueblan el p95. Tirar esas corridas —que es lo que manda §2.3.1—
sesgaría el percentil hacia abajo justo donde más importa.

**La distribución, medida sobre las 10 auditorías.** Canal 1 del WAV, rachas de
voz del agente, y el canal 0 para descartar las fronteras de turno: un hueco
cuenta como interno solo si el arnés no habló en medio. Mismo método en las dos
PoC. 741 pausas internas (265 en PoC1, 476 en PoC2):

| | p50 | p95 | p99 | luego |
|---|---|---|---|---|
| PoC1 | 70 ms | 368 ms | 437 ms | máx 640 ms |
| PoC2 | 70 ms | 370 ms | 480 ms | **1,430 y 1,630 ms** |

**El p95 no es el criterio que separa, y conviene decirlo.** 0.8 s ya estaba muy
por encima del p95 (370 ms) y del p99 (468 ms), y aun así fallaba. La
distribución es **bimodal**: todo por debajo de 640 ms salvo dos casos de 1.43 y
1.63 s. El criterio que decide es la cola, no el percentil.

| umbral | corridas con al menos un solape | pausas afectadas |
|---|---|---|
| 0.8 s | 2 / 10 (20%) | 2 / 741 |
| 1.5 s | 1 / 10 (10%) | 1 / 741 |
| **2.0 s** | **0 / 10** | **0 / 741** |

2.0 s deja **23% de margen** sobre la pausa más larga observada. No se aprieta a
1.8 (que también daría cero) porque la cola son dos casos y ajustar al máximo
observado es elegir el número a ojo con pasos de más.

**Lo que cuesta.** La espera corre una vez por turno, así que son ~**8.4 s por
llamada** de 7 turnos: +28 minutos en una campaña de 200. Es tiempo de silencio,
no de tokens.

**Lo que NO cuesta: la latencia no se mueve.** El cero del turno es el fin de voz
del arnés y la punta derecha es la **primera** voz del agente; las dos son
anteriores al instante de avance, así que la perilla no puede tocarlas. Está
verificado con un par de pruebas que corren la misma actuación a dos valores muy
distintos: `test_el_cero_del_turno_no_se_mueve_con_la_perilla` comprueba que la
primera voz cae en el mismo punto, y su control
`test_la_perilla_alarga_la_espera_pero_no_la_primera_voz` comprueba que la espera
sí se alarga — sin el control, el primero pasaría aunque la perilla no hiciera
nada.

**Idéntica en las 3 PoC.** Vive una sola vez, en
`shared/carrier_simulator/reproductor.py`; PoC1 y PoC2 la importan de ahí y PoC3
la importará cuando tenga arnés. Una prueba verifica que el símbolo reexportado
sea el mismo objeto, para que ninguna PoC pueda definir el suyo sin que se note.

**Marca de corte:** las corridas anteriores al 20/09/2026 se hicieron con 0.8 s.
Comparten la marca de §2.2.9 y no se mezclan con las de la campaña.

### 2.4 El mecanismo de reintento del LLM

Los números del transporte se igualan (§1); el **mecanismo** no se puede, porque
cada framework expone otra cosa:

| | PoC1 — pipecat | PoC2 — LiveKit |
|---|---|---|
| Qué hace | 1 intento acotado a 5 s, luego **uno paciente sin tope** | hasta 4 intentos, todos acotados |
| Perilla | `retry_on_timeout` + `retry_timeout_secs` | `SessionConnectOptions.llm_conn_options` |
| Peor caso acotado | el primer intento; el segundo depende del SDK | ~40 s |

No se fuerza a ninguno a imitar al otro: sería medir un framework fuera de cómo
se despliega. Lo que importa para la comparativa es que **ningún atorón del
proveedor entre a los percentiles sin marca**, y de eso se encarga la captura de
fallos: PoC2 escribe la llave `fallos_tecnicos` al nivel de la llamada —con
`tipo`, `recuperable`, `proveedor`, `modelo` y el segundo en que ocurrió— y le
agrega a la muestra del turno afectado una nota diciendo que su latencia no
describe al framework de voz. La muestra **no se borra ni se anula**: el número
que midió el framework sigue siendo cierto, lo que deja de ser cierto es que
hable de él.

Un atorón del proveedor no descalifica la llamada entera (a diferencia de la
degradación del turn detector, §2.1): afecta a un turno y queda marcado ahí.
Quien consolide decide si lo saca de la poza, con la evidencia a la vista.

**PoC3 no puede aplicar nada de esta sección.** Su llamada al modelo de texto
ocurre **dentro** del servicio de OpenAI, del otro lado del WebSocket, y el
alpha no expone ni timeout ni política de reintentos de la delegación. Lo único
que el código de PoC3 controla es su propio WebSocket. No es una omisión: es
que la perilla no existe. Ver §2.5.

### 2.4.1 El formato de audio del TTS — confounder declarado de `tts_ttfb_ms`

> Sección agregada el 16 de septiembre de 2026, al medir el audio del agente en
> el WAV de auditoría de `spots-poc2-arnes-20260916-113508`.

**Las dos PoC le piden a ElevenLabs formatos distintos, y ninguna lo eligió.**
Las dos construyen el TTS sin pasar formato, así que cada una se queda con el
default de su framework:

| | PoC1 — pipecat | PoC2 — LiveKit |
|---|---|---|
| Formato | `pcm_8000` | `mp3_22050_32` |
| Códec | ninguno | **MP3 a 32 kbps** |
| Tasa de origen | 8 kHz | 22.05 kHz |
| De dónde sale | `output_format_from_sample_rate(sample_rate)`, y el transporte telefónico fija 8 kHz | `_DefaultEncoding` del plugin |

Es la misma voz y el mismo modelo (`eleven_flash_v2_5`, §1), así que **no
invalida el WER ni la comparación de LLM**: lo que cambia es la calidad del audio
del agente y el instante en que se puede medir su primer byte.

**Dónde entra en la métrica.** `tts_ttfb_ms` es "primer token → primer byte de
audio" (§3), pero los dos frameworks fechan cosas distintas:

- **PoC1** para el cronómetro cuando llega el primer trozo de audio **del cable**
  (`stop_ttfb_metrics()` justo antes de decodificar el base64). Con PCM no hay
  decodificación, así que cable y audio son el mismo instante.
- **PoC2** lo para en el primer `AudioFrame` **ya decodificado** — el TTFB de
  `livekit.agents.tts` se marca sobre `ev.frame`, y el plugin le entrega los
  bytes a un emisor que decodifica según el `mime_type`.

O sea que PoC2 paga dentro de `tts_ttfb_ms` dos cosas que PoC1 no tiene:

1. **Encuadre del encoder.** Una trama MP3 son 1152 muestras; a 22.05 kHz el
   proveedor no puede emitir la primera hasta tener **52.2 ms** de audio. Es
   aritmética del formato, no estimación.
2. **Un término de decodificación** que depende de la implementación del emisor y
   que aquí no se acotó.

**Cota: al menos ~52 ms de `tts_ttfb_ms` de PoC2 son códec y no arquitectura.**

**Lo medido va en contra de esa dirección, y por eso la asimetría no rescata
ninguna conclusión.** Sobre las muestras de régimen ya capturadas:

| | n | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| PoC1 | 110 | 235 | **359** | 516 | 735 |
| PoC2 | 181 | 219 | **228** | 241 | 310 |

PoC2 sale más rápido *cargando* el handicap, así que el códec **no explica su
ventaja**; si acaso, la ventaja real es mayor que la medida. Lo que la asimetría
sí impide es afirmar el tamaño: **no se cita la diferencia de `tts_ttfb_ms` entre
PoC1 y PoC2 como magnitud**, solo su signo.

Queda una observación sin verificar, anotada para que nadie la repita como
hallazgo: la dispersión de PoC2 es llamativamente estrecha (IQR de 22 ms) y el
encuadre MP3 produciría por sí solo un jitter de ~26 ms. Coincide, pero no se
comprobó que sea la causa.

**No se cambia el formato a media campaña.** Igualarlo a `pcm_24000` obligaría a
repetir las 15 llamadas de PoC2 —el ambiente está congelado (§1)— y compraría
precisión en una etapa cuya conclusión ya es robusta al confounder. Si algún día
se repiten por otra razón, el valor correcto es PCM en las dos.

**Efecto colateral que sí hay que reportar.** El audio del agente en PoC2 pasa
por un códec con pérdida a 32 kbps y por un remuestreo de 22.05 a 24 kHz, y eso
es audible: entra en la **nota manual 1-5 de naturalidad**, que sí es métrica de
la comparativa. Quien califique tiene que saber que está oyendo MP3 de 32 kbps en
PoC2 y PCM en PoC1, y **no puede leer esa diferencia como propiedad del
framework**. En el WAV de auditoría el remuestreo se ve directo: por encima de
11.025 kHz —el Nyquist del origen— hay tanta potencia como justo por debajo
(−0.9 dB), y eso no puede ser voz.

> Lo que **no** es: no es μ-law, y no es un piso de ruido. Las dos hipótesis se
> midieron y se descartaron sobre el mismo WAV. La de μ-law, por pertenencia a la
> tabla G.711 (0.72% de las muestras fuertes contra 0.4% por azar; μ-law puro
> daría 100%). La del piso, porque en 109 cierres oclusivos **dentro de palabra**
> la banda de 8–11.5 kHz cae a −79 dB de mediana y hasta −101.5 dB, lo que un
> piso aditivo a −37 dB haría imposible. El "piso plano" que se ve al promediar
> potencia en bloques largos es **artefacto del promedio sobre sibilancia
> ráfagosa**: el mismo método aplicado al canal del arnés —que es banda
> telefónica limpia— inventa un piso de −41 dB que no existe.

### 2.5 Sobre la madurez del producto de PoC3: era alpha, es GA

> Sección agregada el 12 de agosto de 2026, al sustituir la OpenAI Realtime API
> por el alpha de GPT-Live. Va como §2.5 y no como §2.4 porque ese número ya
> estaba ocupado y está referenciado desde `shared/llm_transport/` y desde
> CLAUDE.md; renumerar habría roto esas referencias.
>
> **⚠ Reescrita el 23 de septiembre de 2026: la premisa de esta sección se cayó.**
> El modelo salió del alpha el 21 de septiembre de 2026
> (`docs/hallazgo_gpt_live_ga.md`): `gpt-live-1-boulder-alpha` **ya no existe**,
> `gpt-live-1` se sirve **sin** el header del alpha, con `owned_by: "system"` y
> `shutdown_date: null`, y se factura ($0.05 por minuto de sesión). La campaña
> de 90 llamadas del 22 de septiembre de 2026 corrió **entera sobre el GA**.
>
> **Lo que esto cambia, y hay que decirlo en el reporte:** las cuatro
> prohibiciones de abajo —tráfico de producción, exponer el audio, API
> inestable, sin nivel de servicio— **ya no aplican al modelo que se midió**.
> PoC3 deja de ser una «sonda exploratoria» y pasa a ser un **candidato
> elegible en igualdad de condiciones** con PoC1 y PoC2. Es el cambio de
> narrativa más grande de toda la comparativa y no lo decide este archivo: se
> toma con el equipo.
>
> **Lo que NO cambia:** (a) las **15 llamadas de agosto** sí corrieron sobre el
> alpha, y sus cifras no se mezclan con las del GA — cambió la cadencia de
> entrega y el desfase del cabezal (regla de no-mezcla, A8); (b) la voz sigue
> sin poder igualarse (TTS nativo contra ElevenLabs Flash); (c) PoC3 sigue sin
> desglose por etapa. Esas tres asimetrías sobreviven al GA.
>
> El texto que sigue se conserva **como estaba** porque describe el régimen bajo
> el que se corrieron las 15 de agosto, que son las únicas a las que aplica.

#### 2.5.1 El régimen del alpha (vigente sólo para las 15 llamadas de agosto)

Todas las asimetrías anteriores son de **configuración**: dos frameworks que
resuelven la misma pieza de forma distinta, cada uno como su fabricante
recomienda. Esta es de otra clase, y por eso va aparte: es una asimetría de
**madurez de producto**. No se corrige — se reporta.

PoC3 corre sobre un alpha de acceso limitado cuyos propios términos declaran:

- **Prohibido el tráfico de producción**; uso limitado a pruebas de bajo volumen.
- **Prohibido exponer el audio generado** a usuarios finales o a cualquiera fuera
  de la organización que participa en las pruebas.
- El modelo y la forma de la API **van a cambiar** antes del GA, para el que no
  hay fecha.
- La confiabilidad y la latencia **no cumplen estándares de servicio de
  producción** — lo dice el proveedor, no nosotros.

Qué significa para la comparativa, en concreto:

| | PoC1 y PoC2 | PoC3 |
|---|---|---|
| Estado del producto | GA, con SLA | alpha de acceso limitado, sin SLA |
| ¿Se puede desplegar? | sí | **no**, los términos lo prohíben |
| ¿El número de latencia describe al producto final? | sí | **no**: el proveedor avisa que hoy no cumple estándares de producción |
| ¿La API va a seguir igual? | cambios normales de versión | **cambios grandes garantizados** antes del GA |

Por eso los números de PoC3 se leen como **sonda exploratoria** —"esto es lo que
viene y así se comporta hoy"— y no como los de un candidato elegible en igualdad
de condiciones. Una comparación que pusiera las tres columnas al mismo nivel
estaría recomendando algo que no se puede desplegar. **El reporte final tiene que
citar esta sección cada vez que presente un número de PoC3.**

Lo que sí es legítimo medir con él: cuánto cuesta integrarlo, qué visibilidad
da, y si la arquitectura full-duplex cambia el turn-taking de forma que valga la
pena esperar al GA. Esas conclusiones sobreviven a los cambios de API que vienen;
un p50 al milisegundo, no.

Dos consecuencias operativas que ya están resueltas en el código:

- **Las 15 llamadas van a un celular propio** (CLAUDE.md), así que el audio
  generado no sale de la organización y la restricción de los términos se
  cumple. Eso es justo lo que hace defendible correr PoC3 — y lo que impide
  usarla en una demo con un transportista real mientras siga en alpha.
- **La voz no se puede igualar.** §1 congela "ElevenLabs Flash v2.5, misma voz"
  para las 3 PoC, y PoC3 no puede cumplirlo: su TTS es nativo y el alpha solo
  documenta `marin`. La asimetría ya existía en la tabla de arquitecturas (PoC3
  siempre iba con TTS nativo); se nombra aquí para que quien lea la puntuación de
  naturalidad sepa que compara dos voces distintas, no dos frameworks con la
  misma voz.

### 2.6 Chequeo obligatorio de humo antes de aprobar una campaña

**Regla, no narrativa.** Ninguna PoC arranca sus 15 llamadas sin que el humo
nivel 1 verifique estas cuatro cosas sobre el JSON que produjo. Si alguna falla,
la campaña no se aprueba:

| # | Qué se verifica | Cómo se ve en el JSON | Si falla |
|---|---|---|---|
| 1 | **La transcripción del agente llega** | `transcripcion` con al menos una entrada de rol `agente` | no hay WER del agente ni auditoría escrita del límite financiero |
| 2 | **El tool calling ejecuta** | `resumen.historial_ofertas` con ≥1 entrada, o sea que `proponer_precio` corrió | el agente puede pronunciar montos que el motor nunca validó: rompe la regla no negociable de CLAUDE.md |
| 3 | **La latencia produce números** | `latencias[].turno_completo_ms` no nulo en la mayoría de los turnos | no hay criterio de decisión de latencia |
| 4 | **El agente habló de verdad** | energía RMS del audio de salida por encima del umbral en una fracción no trivial de las tramas | un agente callado hace pasar (1) y (3) por vacíos y parecen bugs del framework |

**Por qué existe esta regla: el mismo bloqueante apareció en dos PoC seguidas.**
PoC1 lo tuvo el 3 de agosto de 2026 —el anotador del agente estaba antes del TTS
y el JSON salía sin lo que el agente dijo— y PoC3 lo repitió el 12 de agosto: el
humo nivel 1 salió con exit 0 y `transcripcion` solo con rol `transportista`.
Los chequeos codificados pasaban porque verificaban que las llaves existieran,
no que tuvieran el contenido que hace falta para medir.

El renglón 4 es el que se aprendió tarde. En PoC3, la transcripción vacía del
agente parecía un evento faltante del alpha; medir la energía del audio de
salida mostró que solo el 2% de las tramas traían voz — el agente había dicho
`" Hola,"` y se había callado. **Sin ese cuarto chequeo, (1) y (2) no se pueden
interpretar**: no se distingue "el framework no reporta" de "el agente no habló".

### 2.7 Asimetría de método: cómo se compone `turno_completo_ms` en cada PoC

**Las 3 PoC reportan la misma magnitud** —fin de habla del transportista hasta
que empieza a sonar el agente, con red y transporte adentro— pero **no la
obtienen igual**, y eso hay que declararlo antes de comparar los percentiles.

| PoC | Cómo se compone | Cero del turno | Fin del turno |
|---|---|---|---|
| 1 — Pipecat | eventos internos del framework | fin de habla del VAD | primer `TTSAudioRawFrame` |
| 2 — LiveKit | eventos internos del framework | marca del VAD menos su `stop_secs` | primer audio del TTS |
| 3 — GPT-Live | **observación externa de energía** | fin de la última **racha** de voz que entra por el transporte | primera trama con voz que sale hacia el transporte |

**Por qué PoC3 no puede componerla desde eventos.** Se intentó y se midió que no
funciona (llamada real del 12 de agosto de 2026):

- El `turn.done` del transportista —el único evento del alpha con un fin de
  habla— llega **10 a 16 s después de su propio `end_ms`**. De 9 turnos de
  régimen, 8 no se pudieron medir con él.
- `output_audio.delta` es un stream **continuo que incluye el silencio**, así
  que "el primer delta después del turno" no es "cuándo empezó a hablar el
  agente": es el siguiente tick, a menos de 100 ms.
- El alpha entrega el audio por detrás del reloj de pared (`ritmo_de_entrega`
  0.93 en esa llamada, 16.7 s de atraso acumulado en 3.7 min), y ese atraso
  contaminaba cualquier número derivado de sus marcas de tiempo.

Antes de eso se intentó traducir el eje del servidor al reloj local, y el censo
de eventos del mismo día (`grabaciones/humo/censo_*.jsonl`) descartó los dos
orígenes posibles: anclando en `session.started`, el residual mediano fue de
6,117 ms con 4,925 ms de deriva; anclando en el primer `output_audio.delta`,
5,227 ms y 4,984 ms. El origen no era el problema: unos 39 s de eje contiguo
tardaron 48.0 s de pared en llegar, con un atraso que creció monótono de 0 a
8,885 ms. (El código citaba ese tramo de dos formas que no cuadran entre sí —393
deltas sobre 0–39,360 ms con un salto, y 392 sobre 0–39,200 ms sin huecos— y el
censo de agosto no está en el repo para decidir.) En la llamada real,
8 de 9 turnos de régimen tuvieron que caer al reloj local y sus
`turno_completo_ms` crecieron de 3,991 a 16,197 ms siguiendo ese atraso, no la
conversación.

**Qué es y qué no es la señal de energía.** `bridge.hay_voz` compara el RMS de
la trama contra un umbral fijo, en los dos sentidos del puente. **No es un VAD y
no toma ninguna decisión conversacional**: el fin de turno, el turn-taking y las
interrupciones los sigue resolviendo el servidor del alpha sobre el mismo audio,
que es justo el mecanismo que la comparativa quiere medir. La señal solo pone
marcas de tiempo, exactamente como haría un observador externo con un
osciloscopio en la línea.

**Cómo leerlo en el reporte.** La asimetría corre a favor de la comparabilidad,
no en contra: PoC3 mide lo que el transportista oye —incluye red, puente y
remuestreo— que es la misma definición extremo a extremo de PoC1 y PoC2. Lo que
cambia es de dónde salen las marcas. Dos consecuencias:

1. **PoC3 no tiene desglose por etapa** y nunca lo va a tener: no hay STT, LLM
   ni TTS aislables. Eso ya estaba declarado en §3.
2. **El umbral de energía es una perilla de PoC3 que las otras dos no tienen**
   (`bridge.UMBRAL_DE_VOZ_RMS`, y `SILENCIO_ENTRE_TURNOS_SEG` para separar dos
   intervenciones del agente). Se fijan antes de la campaña y **no se tocan a
   media campaña**; si hubiera que cambiarlas, se repiten las llamadas ya
   hechas.

**Rachas, no tramas — corrección del 12 de agosto de 2026, con la llamada 1.1 en
la mano.** La primera versión anclaba el cero en la **última trama** con voz y
abría muestra en cada arranque de voz saliente. Sobre una línea telefónica real
eso produjo **19 muestras para 8 turnos**, 7 de ellas sin número, por dos causas
de la misma familia que los turnos fantasma que PoC1 ya había tenido:

- cualquier trama de 20 ms sobre el umbral —una respiración, un clic— reescribía
  el cero, y el "silencio" del transportista salía de 15 a 204 ms;
- las pausas del agente a media respuesta (el relleno "Un segundo." mientras
  corre la delegación) abrían muestra nueva contra un cero viejo, y de ahí los
  valores de 2.5 a 5.4 s.

Lo que corre desde entonces, y hay que citarlo así en el reporte:

- la voz entrante se agrupa en **rachas** (hangover de
  `SILENCIO_PARA_TURNO_TERMINADO_SEG`, **800 ms**) y el cero es la última trama con
  voz **de la racha**, no de la llamada. El hangover decide dónde termina la
  racha; **no se suma** al número;
- una racha más corta que `VOZ_MINIMA_DEL_TURNO_SEG` (150 ms) es ruido de línea y
  no puede ser el cero de nada;
- **una muestra por racha contestada.** El agente reanudando su propia
  intervención no produce muestra: se cuenta en
  `gpt_live_extra.turn_taking.reanudaciones_del_agente`;
- si el agente arranca con la racha del transportista abierta, la muestra sale con
  `turno_completo_ms = None` y nota — nunca con el hueco entre dos palabras como
  si fuera latencia. Es el mismo criterio de los deltas negativos de PoC1;
- **`es_apertura` es la primera muestra de la llamada, punto.** Derivarlo de "no
  se había oído al transportista" no funciona sobre teléfono, porque el
  transportista contesta hablando: en la llamada 1.1 **ninguna** muestra salió
  marcada como apertura y el arranque en frío se colaba a la cubeta de régimen.

**El apoyo verbal no arma un cero — corrección de la llamada 1.2.** Faltaba un
caso: el transportista mete un "ajá" **mientras el agente habla**, esa racha cierra
y arma el cero, el agente sigue hablando encima, y su siguiente frase —ocho
segundos más tarde— consume ese cero como si fuera una respuesta. La 1.2 reportó
así un p75 de 2,687 ms y un máximo de **8,266 ms** junto a un p50 de 1,609 ms.
Ahora el cero pendiente sólo vale si el agente estuvo **callado** desde que la
racha terminó; si ya había hablado después, esto es una reanudación y se cuenta en
`ceros_vencidos_por_apoyo_verbal`.

`gpt_live_extra.turn_taking` publica los conteos (`rachas_del_transportista`,
`rachas_descartadas`, `rachas_sin_respuesta`, `reanudaciones_del_agente`,
`arranques_sobre_el_transportista`, `ceros_vencidos_por_apoyo_verbal`) para poder
auditar el cuadre y para calibrar las perillas con datos, no a ciegas. Es el mismo
patrón que los `veredictos_fin_de_turno` de PoC1. **El cuadre se revisa en cada
llamada**, y si no da, el p50 no se cita:

```
muestras = 1 apertura
         + (rachas_del_transportista - rachas_descartadas
            - rachas_sin_respuesta - ceros_vencidos_por_apoyo_verbal)
```

(los `arranques_sobre_el_transportista` también producen muestra, pero sin número).

**El hangover: 250 → 800 ms, calibrado con datos y congelado antes de la campaña.**
Con 250 ms, una pausa normal a media frase cerraba la racha y armaba un cero en
mitad de la intervención del transportista. La llamada de la tarde del 12 de agosto
de 2026 produjo **129 rachas para 7 intervenciones** y cuatro muestras de 313–453 ms
que eran el hueco entre dos de sus palabras, no una respuesta. **El p50 de régimen
pasaba de 2,156 a 1,117 ms según se contaran o no** — un segundo entero en la
métrica principal.

El valor sale de la distribución de huecos internos de su propia habla, medida
sobre los `input_transcript.added` de esa llamada: 26 de 42 huecos pasaban de
250 ms, 21 de 400 ms, 10 de 600 ms y sólo **2 de 800 ms**. El JSON publica esa
distribución en `gpt_live_extra.turn_taking.huecos_entrantes_ms` con su nota, así
que la perilla se puede re-derivar en cada llamada en vez de defenderse de memoria
— el mismo patrón que los `veredictos_fin_de_turno` de PoC1.

Dos cosas que hay que declarar al citarlo:

1. **La dirección del sesgo.** Un hangover más largo mueve el cero **más tarde** —al
   final real de la frase— y por lo tanto **baja** la latencia medida. No se eligió
   por eso: medir desde la mitad de una frase es medir desde donde el transportista
   todavía hablaba. Pero el sesgo existe y el lector tiene derecho a saberlo.
2. **No se compara contra el `stop_secs` de PoC1 (0.2 s) ni el
   `min_silence_duration` de PoC2 (0.55 s).** Aquellos son perillas
   *conversacionales*: deciden cuándo contesta el agente. Esta es de *medición*
   nada más, porque el turn-taking de PoC3 lo resuelve el alpha y este número no lo
   toca. Que los tres valores sean distintos no es un sesgo a corregir.

**Congelado desde la primera de las 15.** Si una corrida muestra muchos huecos por
encima de los 800 ms, se anota y se repiten las llamadas ya hechas — no se cambia a
media campaña.

**⚠ `retraso_de_entrega` NO es una latencia y no se cita como tal.** En la llamada
1.2 el bloque publicó `atraso_final_ms: 31312` y `deriva_ms: 26996` mientras los
turnos medidos por energía local salían en 1,300–1,700 ms. Los dos números son
correctos: el eje `start_ms` del alpha avanza más despacio que el reloj de pared
—959 deltas contiguos, con sólo 2.6 s de huecos, cubrieron 97.6 s de eje en 128.8 s
de llamada— así que la distancia entre los dos relojes crece sola. Eso es
**evidencia de por qué la medida principal se toma por fuera**, no un retraso de
respuesta. Los umbrales `umbral_residual_ms` / `umbral_deriva_ms` pertenecen al
anclaje que se abandonó: el JSON ahora trae `los_umbrales_aplican: false` cuando
se midió por energía local, para que nadie lea un "fallo" donde no lo hay.

**`interrumpido` llevaba un signo de menos.** El criterio era
`(fin_de_voz_del_agente - inicio_del_solape) <= VENTANA_DE_CORTE_SEG` **sin cota
inferior**, así que una brecha negativa —el agente ya se había callado antes de
que el transportista hablara, o sea que nadie lo cortó— pasaba como corte. En la
llamada 1.1 marcó 8 de 19 muestras, con `interrupciones_no_deseadas: 0` anotado a
mano por quien actuó el escenario. Ahora la brecha tiene que caer en
`[0, VENTANA_DE_CORTE_SEG]`. El solape se detecta además con una ventana propia y
corta (`VENTANA_DE_SOLAPE_SEG`, 150 ms): con la de corte, hablar medio segundo
*después* de que el agente terminó contaba como solaparse, y la llamada 1.1
reportó 76.

### 2.7.1 `llm_ttft_ms` **no es comparable** entre PoC1 y PoC2

**Hallazgo del 21 de septiembre de 2026, sobre las 179 llamadas con al menos
una muestra de latencia de la
campaña.** La tabla de etapas publicaba **1,469 ms de TTFT en PoC1 contra 681 en
PoC2** — con el mismo modelo (`gpt-4.1-mini`), el mismo `v0.md` y las mismas
cuatro funciones. Una diferencia de 2.2× ahí no puede ser del framework, y no lo
es: **las dos llaves miden tramos distintos.**

| PoC | Dónde para el cronómetro | Qué le queda adentro |
|---|---|---|
| 1 — Pipecat | el primer **`LLMTextFrame`** (`app/latencia.py`), o sea el primer token de *texto* | la llamada a función, su ejecución en el motor, y la **segunda** petición al modelo |
| 2 — LiveKit | **`llm_node_ttft`**, el primer token que sale del nodo del LLM (`app/latencia.py`) | nada de eso: la llamada a función **es** ese primer token, y ahí se detiene |

El mecanismo es el tool calling. Cuando el modelo decide llamar a
`proponer_precio`, lo primero que emite no es texto sino la función. PoC1 no ve
todavía un `LLMTextFrame` y sigue contando hasta que llega el texto — que
llega después del round trip completo. PoC2 ya paró.

Las dos mediciones son correctas para lo que cada framework expone. Lo
indefendible es ponerlas en la misma columna.

#### La evidencia, por tres caminos independientes

**1. La forma de las distribuciones.** PoC1 es bimodal y PoC2 no, sobre los
turnos de régimen de la poza limpia:

| cubeta | PoC1 (n=663) | PoC2 (n=590) |
|---|---|---|
| 500–750 ms | **233** | **443** |
| 750–1,000 | 78 | 133 |
| 1,000–1,250 | 2 | 7 |
| 1,250–1,500 | 27 | 1 |
| 1,500–1,750 | **184** | 2 |
| 1,750–2,000 | 88 | 1 |
| > 2,000 | 51 | 0 |

El p50 de 1,469 de PoC1 **cae en el valle entre los dos modos**: hay 2 turnos de
663 en su cubeta. Ese número no describe ningún turno de ninguna llamada.

**2. La correlación con el motor de negociación.** Cruzando, por llamada,
cuántas veces se llamó a una función contra cuántos turnos cayeron en el modo
lento (n=88 llamadas de PoC1):

| tool calls en la llamada | turnos lentos, media |
|---|---|
| 3 | 2.00 |
| 4 | 3.24 |
| 5 | 4.12 |
| 6 | 5.24 |
| 7 | 6.00 |

Pendiente ≈ 1, Pearson 0.635: **cada tool call produce exactamente un turno
lento** (`lentos ≈ tool_calls − 1`; el cierre no deja turno de texto medible
después).

**3. El propio PoC2 lo confirma por resta, sin tocar PoC1.** Su
`turno_completo_ms` es el `e2e_latency` de LiveKit, que sí arranca en el fin de
habla y sí contiene el round trip. El residual

```
residual = turno_completo_ms − fin_de_turno_ms − llm_ttft_ms − tts_ttfb_ms
```

es **bimodal** (n=583): 348 turnos en **−106 ms** de p50 —el desglose cuadra— y
235 turnos en **+923 ms** —el desglose se queda corto por casi un segundo—. Y
la pieza que cierra el caso: **el `llm_ttft_ms` de PoC2 vale lo mismo en los dos
modos (686 contra 677 ms)**. Su reloj es literalmente ciego a un segundo entero
de trabajo que el turno sí pagó. Eso solo pasa si para antes.

#### Comparable contra comparable

Partiendo PoC1 en el valle (1,100 ms) y PoC2 por su residual (500 ms):

| | n | p50 | p95 |
|---|---|---|---|
| PoC1, turnos **sin** round trip | 313 | **672 ms** | 875 |
| PoC2, `llm_node_ttft` (nunca lo contiene) | 590 | **681 ms** | 906 |
| PoC1, turnos **con** round trip | 350 | 1,687 ms | 2,531 |

**672 contra 681 ms: nueve milisegundos.** Mismo modelo, mismo prompt, mismo
resultado — que es lo que tenía que salir. Los 788 ms de diferencia eran, los
788, instrumentación.

Eso también **descarta el tamaño del contexto** como explicación, sin tener que
medirlo: un contexto más grande en PoC1 correría *toda* su distribución, no le
agregaría un segundo modo. El modo rápido de PoC1 está encima del de PoC2.

#### El tramo que sí se puede comparar con lo que ya está medido

Sin gastar una llamada nueva se recupera un desglose honesto, porque hay un
tramo que en las dos contiene el round trip:

- **PoC1:** `llm_ttft_ms + tts_ttfb_ms`.
- **PoC2:** `turno_completo_ms − fin_de_turno_ms` (por resta, porque la suma de
  sus etapas publicadas deja el round trip fuera).

Es el mismo tramo —LLM y TTS, desde que el turno quedó cerrado hasta el primer
byte de audio— y **también hay que partirlo por modo**, porque las fracciones de
mezcla son distintas (54.6% de turnos con round trip en PoC1, 40.3% en PoC2) y
un p50 sobre dos mezclas distintas mide la mezcla:

| | PoC1 | PoC2 | diferencia |
|---|---|---|---|
| **sin** round trip | 1,016 ms (n=291) | 822 ms (n=348) | +194 ms |
| **con** round trip | 2,000 ms (n=350) | 1,840 ms (n=235) | +160 ms |

Una diferencia estable de **~160–195 ms**, en la misma dirección en los dos
modos, de la cual **77 ms son el confounder de formato de audio del TTS ya
declarado en §2.4.1** (`tts_ttfb_ms` p50: 297 en PoC1 contra 220 en PoC2). Eso
es el desglose por etapa defendible; el 1,469 contra 681 no lo era.

Por qué difieren las fracciones (54.6% contra 40.3%) **queda abierto**: puede ser
la fragmentación de PoC1 (§2.2.5) repartiendo una intervención en dos turnos del
pipeline, o simplemente que el modelo llamó a función un número distinto de veces
—que sería comportamiento de LLM, no de framework, y por tanto ruido—. **No se
resuelve con estos datos**, y es justo la razón por la que los p50 sin partir no
se pueden comparar.

#### El número que sí es del reporte: el round trip cuesta ~1 segundo

Medido de forma **independiente en las dos arquitecturas**, que es lo que lo hace
citable:

| medido en | cómo | costo |
|---|---|---|
| PoC1 | p50 del tramo comparable con round trip − sin round trip | **984 ms** |
| PoC2 | ídem, partiendo por su residual | **1,018 ms** |

**~1,000 ms, y a 34 ms entre las dos.** Tres cosas que el reporte tiene que decir
juntas:

1. **Lo pagan las dos arquitecturas por igual.** No es un diferenciador entre
   PoC1 y PoC2: es lo que cuesta que el LLM llame a una función a media
   respuesta, y ninguna de las dos lo esconde ni lo amortigua.
2. **Está dentro de la latencia observada de las dos.** El arnés mide desde
   fuera, así que ese segundo ya está contado en los 2,602 ms de PoC1 y los
   3,081 de PoC2. No hay nada que sumar.
3. **Contra PoC3 la lectura cambia por completo**, y ahí sí es diferenciador: en
   PoC3 el mismo ida y vuelta lo tapa el relleno del modelo full-duplex y **no**
   cae dentro de `turno_completo_ms` (CLAUDE.md, `_tapado_por_voz_n`). En PoC1 y
   PoC2 se paga como aire muerto. **Comparar los p50 crudos de las 3 sin decir
   esto es engañoso.**

#### Lo que esto NO toca: las tres cifras de latencia de §2.2.8

**El defecto vive en la tabla de etapas y en ningún otro lado.** Las tres cifras
de latencia no derivan de `llm_ttft_ms` por ninguna vía:

- la **observada** la mide el arnés **desde fuera** de los dos frameworks —fin de
  voz del clip hasta el primer audio del agente que oye—, con el mismo método en
  las dos;
- la **ida** es el `sesgo_de_voz` (§2.2.2), medido contra el cero del pipeline;
- la **vuelta** se mide punta a punta con sondas propias (§2.2.7).

O sea que **la diferencia corregida de 44 ms no está compensando dos errores**:
está construida con tres instrumentos que no tocan esta llave. Si acaso, este
hallazgo la refuerza — el segundo del round trip está dentro de la observada de
las dos, y aun así, descontado el transporte, quedan a 44 ms.

#### Reglas que quedan

- **El 1,469 no se cita.** Se reemplaza por los dos modos de PoC1 (672 sin round
  trip, 1,687 con) o por el 672 comparable, y nunca sin decir cuál es cuál.
- **`llm_ttft_ms` no va en una tabla de dos columnas** PoC1 | PoC2 sin esta
  sección al lado. Mide tramos distintos y sumar sus etapas da totales distintos.
- **La suma de etapas de PoC2 no reconstruye su turno** (le faltan 923 ms de p50
  en el 40% de los turnos). Para el turno completo se usa `turno_completo_ms`,
  no la suma.
- **Ninguno de los dos números está mal medido.** Cada framework reporta
  fielmente lo que su API expone. Lo que estaba mal era ponerlos en la misma
  columna sin declarar el tramo — y eso es exactamente el tipo de error que este
  contrato existe para atrapar.

#### Pendiente: llenar `round_trips_del_llm` (no se hace ahora)

CLAUDE.md ya define la llave neutral `round_trips_del_llm` y sus cinco columnas
de CSV (`llm_delegado_n`, `_p50_ms`, `_max_ms`, `_con_funcion_n`,
`_tapado_por_voz_n`) precisamente para esto, y dice que **PoC1 y PoC2 pueden
llenarla con sus tool calls**. Hoy **sale vacía en las dos**, y por eso el round
trip hubo que reconstruirlo por inferencia estadística en vez de leerlo.

Llenarla es el arreglo de fondo: haría directo lo que arriba es una partición por
umbral, y además daría el conteo por turno en vez de por llamada.

**No se hace con la campaña ya corrida.** Cuesta instrumentación en las dos PoC
(un gancho en el manejador de funciones de cada framework) más **una campaña
nueva de 200 llamadas** para poblarla, porque la llave se escribe en vivo y no se
puede derivar de los artefactos existentes. Reabrir la campaña por un desglose
que ya se recuperó por inferencia no se justifica — y a la fecha de este
hallazgo no hay TTS disponible para correrla. Queda anotado en los pendientes de
CLAUDE.md.

### 2.7.2 La latencia de apertura queda FUERA del reporte

**Decisión del 22 de septiembre de 2026.** El turno de apertura se sigue
capturando y marcando (`es_apertura`), y sigue saliendo de los percentiles de
régimen como hasta ahora. Lo que cambia es el entregable: **no se reporta en
ninguna PoC.** No va al `.md`, no va al CSV, no se compara entre arquitecturas y
no lleva columna propia.

El motivo es que los tres ceros no encierran el mismo arranque en frío, y por lo
tanto el número no mide lo mismo en las tres:

- PoC1 arranca en `on_client_connected` del Media Stream
  (`poc1_pipecat/app/telefonia.py:456`), antes de que el pipeline haya levantado
  el WebSocket del TTS;
- PoC2 arranca cuando contestaron, justo antes de `sesion_voz.start()`
  (`poc2_livekit/app/agente.py:776`);
- PoC3 arranca tras leer el `start`, antes de conectar contra el modelo
  (`poc3_gpt_live/app/telefonia.py`).

Los tres se llaman "arranque en frío" y los tres contienen cosas distintas. Lo
que lo vuelve indefendible como cifra comparable no es el argumento sino la
medida: sobre los 84 pares de la campaña, **la apertura de PoC1 es 2,359 ms de
mediana contra 1,164 de PoC2 —el doble— mientras que sus p50 de régimen quedan
a 33 ms (2,356 contra 2,389)**. Un factor de dos que aparece solo en el turno
cuyo cero está definido distinto en cada PoC es un artefacto de la definición,
no un resultado, y publicarlo invitaría a leerlo como "PoC2 arranca más rápido".

Igualarlo tampoco sirve: cada framework levanta su motor en un orden distinto y
no hay un instante común a los tres que sea a la vez observable y equivalente.
Se declara fuera de alcance y se deja de reportar, que es lo honesto.

**No se re-litiga.** Si alguna vez hace falta el arranque en frío como métrica
de producto, se diseña su propia medición con un cero definido para eso — no se
recicla éste.

### 2.8 El orden de la campaña: las PoC se alternan, no se corren en tandas

**Regla, no narrativa.** Una campaña que compare dos arquitecturas **intercala
sus llamadas**. `shared/lote/` no ofrece la otra opción: el plan sale con las dos
PoC del mismo escenario pegadas —`poc1-e1-r1`, `poc2-e1-r1`, `poc1-e2-r1`, …— y
entre cada par pasan minutos.

**Por qué.** Correr 100 llamadas de PoC1 un día y 100 de PoC2 al siguiente hace
que *todo* lo que cambie entre los dos días entre en la diferencia medida sin
forma de separarlo después: la carga de OpenAI, Deepgram y ElevenLabs a esa hora,
el estado de la red de salida, y —el peor— una versión nueva detrás de la misma
etiqueta de modelo. Esa última ya es un riesgo declarado en CLAUDE.md para
`gpt-4.1-mini`. Con las llamadas intercaladas, lo que le pase al proveedor le
pasa a las dos PoC casi al mismo tiempo, y deja de ser un término que solo carga
una de las dos.

No elimina el confound —los dos despliegues siguen siendo distintos, y el §2.2.5
ya dice que parte de la diferencia de latencia es despliegue y no arquitectura—.
Lo que elimina es el confound **temporal**, que es el que se puede eliminar
gratis con solo elegir el orden.

**Lo que el lote NO hace, y es a propósito:**

- **No promedia, no agrega y no calcula percentiles de campaña.** Lanza procesos,
  lee el resumen que cada arnés escribe y lleva la cuenta del gasto. Todo número
  que salga de `progreso.jsonl` lo produjo el arnés de esa llamada. Los agregados
  de la comparativa los sigue calculando `shared/metrics/`.
- **No iguala perillas ni reintenta por su cuenta.** Una corrida que el arnés
  marca como no utilizable se anota con su motivo y el lote sigue; repetirla es
  una decisión de dinero y la toma quien paga, agregando repeticiones al plan.
- **No etiqueta sus llamadas como parte de las 15.** El arnés de PoC2 manda
  `{"poc": 2}` sin `escenario` ni `repeticion` a propósito (ver el comentario en
  `arnes.correr_sesion`), y eso no cambia por correr en lote.

**El WER del lote no es el WER de la campaña.** El resumen por llamada trae un
WER calculado contra *la línea que el arnés reprodujo*, y se puede porque el
estímulo es un clip pregrabado: no hay nadie que haya improvisado y no hay
referencia que corregir a mano. Sirve para ver de un vistazo si una llamada salió
sorda. El WER que va al reporte sigue siendo el de `shared/metrics/` con
referencia corregida, que es el método que §1 congela para las 3 PoC.

**El tope de gasto se comprueba antes de lanzar cada llamada**, con el estimado
que el propio arnés cotizó para ese escenario —el lote no lleva su copia de las
tarifas—. Comprobarlo después lo convertiría en un aviso. A una llamada que se
cayó sin dejar resumen se le carga el estimado y no cero: pudo gastar igual, y
subestimar el gasto es la única forma en que un tope deja de ser un tope.

### 2.9 Advertencia de lectura: en este estudio las distribuciones son bimodales

**Esto no es estadística general: es un patrón medido tres veces en estos
mismos datos, y las tres veces el p50 mentía.**

| Dónde | Los dos modos | Qué los separa | Dónde está |
|---|---|---|---|
| `fin_de_turno_ms` de PoC2 | ~577 ms (87.4%) contra ~2,500 ms (9.7%) | los dos caminos de `_bounce_eou_task`: `min_delay` cuando el detector acústico pasa su umbral, `max_delay` cuando no | `comparativo_resultados.md` |
| pausas internas del agente | todo ≤ 640 ms, y luego 1,430 y 1,630 ms | dos casos de 741: la respuesta larga que el modelo compone en dos tramos | §2.3.2 |
| `llm_ttft_ms` de PoC1 | 672 ms (n=313) contra 1,687 ms (n=350) | el turno llamó a una función o no | §2.7.1 |

**El p50 sobre dos modos no describe ningún turno.** En el caso de PoC1 es
literal: su p50 de 1,469 ms cayó en el valle, donde hay 2 turnos de 663. Se citó
como "el TTFT de Pipecat" durante media campaña y no era el TTFT de nada.

Cuatro reglas que salen de ahí, y que valen para cualquier percentil que vaya al
reporte:

1. **Mirar la forma antes de citar el número.** Un p50 no se publica sin haber
   visto el histograma de lo que resume. Cuesta una línea y es lo único que
   distingue una mediana de un artefacto.
2. **Un modo casi siempre es un mecanismo, no ruido.** Las tres veces el segundo
   modo resultó ser un camino de código o un evento discreto —una rama del
   endpointing, un tool call, una respuesta en dos tramos—. Cuando aparece, la
   respuesta correcta es **buscar el mecanismo**, no ampliar la muestra ni
   recortar la cola: el segundo modo es el hallazgo.
3. **Si es bimodal, se publican los modos con su `n` y su fracción de mezcla**,
   no el percentil de la mezcla. Y **nunca se comparan entre PoC dos p50 cuyas
   fracciones de mezcla difieren** — eso mide la mezcla, no la arquitectura.
   §2.7.1 tiene el caso: 54.6% contra 40.3% de turnos con tool call, así que los
   p50 sin partir estaban comparando composiciones distintas.
4. **Elegir una perilla por el p95 no protege de la cola.** En §2.3.2 el valor
   viejo ya era **2.2× el p95** y aun así falló en el 20% de las corridas,
   porque el fallo estaba en el segundo modo y no en la cola del primero. Las
   perillas se eligen contra la distribución entera, con su forma a la vista.

Es el mismo principio que ya está en §2.2.8 con
`deriva_por_restar_percentiles_ms`: **los percentiles no se restan, no se suman y
no se promedian.** Ahí se resolvió corrigiendo por intervención antes de
percentilar; aquí se resuelve partiendo por modo antes de percentilar. La
operación va siempre sobre las muestras, nunca sobre los resúmenes.

#### Y el corolario que ya pasó dos veces: un número razonable no es evidencia

**Dos defectos en direcciones opuestas producen un resultado creíble.** No es
una hipótesis: en este estudio pasó dos veces, y las dos el número equivocado se
citó sin que nada chirriara.

| | los dos defectos | lo que producían juntos | lo real |
|---|---|---|---|
| **TTFT** (§2.7.1) | PoC1 medía un tramo con el round trip de la herramienta adentro y PoC2 uno sin él; encima el p50 de PoC1 caía en el valle de su bimodal | 1,469 contra 681 ms — una diferencia grande y con una explicación a la mano ("Pipecat es más lento") | 672 contra 681 ms: **9 ms** |
| **WER** (§2.10) | el corte por `alcanzados` tiraba 5 turnos con error real, todos de PoC2; y la hipótesis llevaba los interinos duplicados de LiveKit, todos de PoC2 | 7.91% contra 7.77% — una brecha de 0.14 pp, o sea "empatan" | 7.68% contra 6.52%: **1.15 pp** |

El segundo es el más instructivo, porque tiene tres respuestas distintas según
cuánto se arregle: **+0.14 pp** con los dos defectos dentro, **−0.14 pp**
arreglando sólo el corte —el signo se invierte, o sea que cambia **quién gana**—
y **+1.15 pp** con los dos arreglados y la poza pareada. La cifra publicada era
un octavo de la real y apuntaba al lado equivocado.

Tres reglas prácticas, que es lo único que sirve de esto:

1. **Que un número se vea razonable no es evidencia de que la medición esté
   bien.** Un resultado plausible es igual de compatible con "medimos bien" que
   con "dos errores se cancelaron". La verosimilitud no discrimina entre las dos.
2. **Lo que discrimina es la verificación cruzada**: el mismo hecho medido por
   un camino independiente. El round trip salió a 984 ms por PoC1 y 1,018 por
   PoC2 partiendo por criterios distintos; la referencia derivada se cruzó
   contra el manifiesto en 1,276 de 1,276 turnos. Cuando dos caminos
   independientes coinciden, la coincidencia sí es evidencia.
3. **Las dos veces el defecto apareció mirando la forma, no el resumen** — el
   histograma bimodal, el residual bimodal, una mutación que sobrevivía. Ningún
   promedio los habría delatado.

### 2.10 El WER de la campaña: referencia derivada de los clips, no corregida a mano

**Decisión del 21 de septiembre de 2026.** El método de §1 —"la referencia la
escribe quien actuó, corrigiendo las líneas `yo:`"— existe porque una persona
actuando se desvía del guion y solo ella sabe qué dijo. **Con el arnés esa
premisa no aplica**: el transportista es un `.ulaw` pregrabado, así que lo que
sonó es exactamente el texto de ese clip, y la bitácora registra cuál se
reprodujo en cada turno, ramas incluidas.

La referencia entonces **se deriva, no se transcribe**, y lo único manual que
queda son las dos notas 1-5.

**Verificado con dos registros independientes del mismo hecho**: la
`linea_reproducida` que el arnés anotó al reproducir y el `texto` que el
generador de clips anotó al sintetizar. Sobre los 1,276 turnos de la campaña
coinciden **1,276 de 1,276**, sin un solo clip ausente del manifiesto. La
comprobación no es opcional: `cli referencia-de-clips` aborta si aparece una
discrepancia, porque una referencia derivada que no cuadra con el manifiesto no
es confiable.

> **Revisado el 23 de septiembre de 2026 — ese cruce no probaba lo que decía.**
> `linea_reproducida` y el `texto` del manifiesto **no son dos registros
> independientes del mismo hecho**: son el mismo guion copiado dos veces. Ninguno
> de los dos mira el audio, así que el cruce daba verde en las 218 corridas sin
> validar nada sobre lo que sonó. Sigue sirviendo —detecta que el arnés reprodujo
> el clip equivocado— pero no es evidencia de que el clip diga el guion. El
> docstring de `verificar_contra_manifiesto` afirma lo contrario y hay que
> corregirlo.

#### La referencia se corrige contra el audio — 23 de septiembre de 2026

Al mirar el audio aparecieron dos cosas. El banco se generó recortando silencio
con un umbral **absoluto** (`umbral_recorte_dbfs`, de −35.9 a −39.3 dBFS) sobre
grabaciones sin normalizar cuyo pico va de −19.4 a −31.0 dBFS: la puerta queda
6–17 dB por debajo de la sílaba más fuerte, y el habla normal abarca 30–40 dB,
así que una frase final dicha más bajo cae entera por debajo y se elimina como si
fuera silencio. Pasó en **4 de los 49 clips**. Y **1 clip** se actuó con una
palabra distinta de la escrita.

Cinco correcciones, ninguna otra:

| clip | referencia vieja (guion) | referencia nueva (audio) | de dónde sale |
|---|---|---|---|
| `e5_t7` | …Quedamos en 18,500 **con todo incluido**. | …Quedamos en 18,500. | prueba geométrica: «incluido» termina **1,140 ms** después del final del clip publicado |
| `e3_t6` | …pero de ahí **no me muevo**. | …pero de ahí… | prueba geométrica: **−510 ms** |
| `e2_t6` | …Diecisiete… no, dieciocho **mil**. | …Diecisiete… no, dieciocho. | prueba geométrica: «mil» a **−210 ms**; «18» cabe con +190 |
| `e2_t5_alt1` | …tiene que ser dieciocho **mil**. | …tiene que ser dieciocho. | prueba geométrica: **−270 ms**; «18» cabe con +130 |
| `e1_t3` | ¿Y para cuándo la **necesitan** en Laredo? | ¿Y para cuándo la **necesita** en Laredo? | desvío al actuar, confirmado por quien grabó el clip |

La prueba geométrica no depende de ningún oído ni de ningún umbral: el clip
publicado es un **trozo contiguo** del crudo —verificado en los 49 a ±0.2 ms— así
que basta preguntar si la última palabra, fechada **en el crudo sin recortar**,
cae dentro de la ventana que el publicado conserva.

**Seis clips más tienen el final atenuado pero presente** (márgenes de +140 a
+410 ms): entran completos. Excluirlos por poco audibles convertiría la
referencia en "lo que un humano resuelve sobre μ-law a 8 kHz", que depende del
oyente y no del estímulo. Se declara en su lugar como característica del banco:
hay finales de frase 16–33 dB por debajo del cuerpo, y **las tres PoC las pagan
por igual**.

**El estímulo no cambió.** Los 49 `.ulaw` y sus sha256 son los mismos bytes que
reprodujeron las tres PoC; lo único que cambia es el texto contra el que se
compara, y cambia igual para las tres. Método, márgenes en milisegundos, la
revisión a oído y los 11 errores del transcriptor de apoyo:
`docs/respaldo_referencia_de_audio.md` — **que no forma parte del reporte**.

#### Qué movió la corrección

Toca **58 de 569 turnos (10.2%)** en PoC1 y PoC2 y **72 de 633 (11.4%)** en
PoC3. La diferencia es que PoC3 reprodujo `e5_t7` y las otras dos no; **sobre los
39 clips comunes la corrección es simétrica** y mueve a las tres casi lo mismo.

| | | WER antes | WER después | CER antes | CER después | montos antes | montos después |
|---|---|---:|---:|---:|---:|---:|---:|
| **población propia** | PoC1 | 0.0712 | **0.0600** | 0.0554 | 0.0465 | 0.3557 | 0.2258 |
| | PoC2 | 0.0802 | **0.0700** | 0.0615 | 0.0528 | 0.2752 | 0.1290 |
| | PoC3 | 0.0403 | **0.0237** | 0.0324 | 0.0159 | 0.1579 | 0.0244 |
| **39 clips comunes** | PoC1 | 0.0712 | **0.0600** | 0.0554 | 0.0465 | 0.3557 | 0.2258 |
| | PoC2 | 0.0802 | **0.0700** | 0.0615 | 0.0528 | 0.2752 | 0.1290 |
| | PoC3 | 0.0335 | **0.0223** | 0.0235 | 0.0141 | 0.1726 | 0.0211 |

El orden es **PoC3 < PoC1 < PoC2** antes y después, en las dos poblaciones. El
error de montos es lo que más se mueve, y era de esperarse: dos de los cuatro
clips cortados perdían justo el «mil» de un monto, así que la referencia pedía
"dieciocho mil" donde el audio decía "dieciocho" y las tres PoC pagaban un error
de monto por una palabra que nunca sonó.

**Sensibilidad, una línea:** excluir del cálculo los cuatro clips con pérdida
confirmada deja el orden igual (PoC1 0.0550, PoC2 0.0654, PoC3 0.0229). La
exclusión **no es simétrica en n** —7.0% de los turnos en PoC1 y PoC2, 9.2% en
PoC3— y por eso no se adopta: se publica solo para mostrar que el resultado no
depende de esos cuatro clips. Corregir la referencia domina a excluir los turnos,
porque conserva la n y los turnos difíciles sin la asimetría.

#### El párrafo que va al reporte

Esto es lo único de todo lo anterior que entra en el reporte, en métodos:

> **Referencia del WER.** La referencia se deriva del **audio reproducido**, no
> del texto del guion. El banco de clips se generó recortando silencio con un
> umbral absoluto sobre grabaciones sin normalizar, y en cuatro de los 49 clips
> ese recorte eliminó palabras que el guion sí pide; además un clip se actuó con
> una palabra distinta de la escrita. Comparar contra el guion habría penalizado
> a los tres sistemas por palabras que nunca sonaron. La referencia se verificó
> transcribiendo los 49 clips publicados y sus 49 originales sin recortar con un
> cuarto motor de reconocimiento, independiente de los tres evaluados, y
> contrastando cada final de frase contra el original mediante la correspondencia
> exacta entre el clip publicado y el tramo del que procede; 43 de los 49 se
> revisaron además a oído, con criterio fijado antes de escuchar. El estímulo no
> cambió —los clips son los mismos bytes que reprodujeron los tres sistemas— y la
> referencia corregida se aplica de forma idéntica a los tres.

**El archivo derivado declara de dónde sale su hipótesis** (`hipotesis:
archivo`), y hace falta. Por defecto el WER toma la hipótesis del JSON de la
llamada y no de las líneas `stt:`, porque en un archivo corregido a mano esas
líneas son una copia para leer y pudieron editarse por error. En un archivo
derivado la relación se invierte: las `stt:` **llevan dentro las dos exclusiones**
repartidas turno por turno, así que tomar el JSON entero devuelve el texto de los
turnos excluidos como **inserciones**. Medido: **11.61% contra 7.68%**, con 287
inserciones contra 7.

**La hipótesis NO sale de la bitácora.** Sale de la transcripción del pipeline.
El campo `stt` por turno de la bitácora sirve para leer la corrida, pero en PoC2
concatena resultados **interinos** de LiveKit y repite texto —"…me lo pagan a
veintiocho Ese viaje me lo pagan a veintiocho mil"—: **187 palabras de más en la
campaña, todas de PoC2**, que como hipótesis serían inserciones inventadas.

#### Las dos reglas de exclusión

**1. La cola muerta.** Un turno del final cuyo clip sonó cuando la conversación
ya había terminado no entra. El corte exige las **dos** condiciones a la vez —
sin STT **y** sin respuesta del agente — y esa conjunción es la regla, no un
detalle:

| caso | qué es | qué se hace |
|---|---|---|
| cola, sin STT, sin respuesta | el clip sonó contra nadie | **se excluye** |
| cola, **con STT**, sin respuesta | el STT sí lo oyó; su error es real | **cuenta** |
| sin STT, **con respuesta** | el STT tuvo su oportunidad y perdió el turno | **cuenta** |
| mudo **en medio** de la llamada | es una falla, no cola | **cuenta** |

Los dos casos del medio son los que se pierden si la regla se escribe de menos.
Están medidos: en la campaña hubo **45 turnos de cola muerta, los 45 de PoC1**
(324 palabras, 4.38% de su referencia contra 0.00% de la de PoC2 — contarlos
habría metido borraduras fantasma en una sola PoC); y **5 turnos con STT y sin
respuesta, los 5 de PoC2**, con errores reales adentro (uno perdió "pero de ahí
no me muevo" entero).

Uno de esos casos era un **defecto vivo**: el arnés cortaba por `alcanzados`,
que cuenta turnos *contestados*, así que descartaba esos 5. Corregido en
`shared/carrier_simulator/resumen.py`. La dirección importaba: con ellos fuera
la brecha de WER daba **+0.14%** y con ellos dentro **−0.14%** — el signo de
quién gana. **Por eso el WER por llamada que quedó escrito en los resúmenes de
la campaña no se cita**: lleva ese defecto y el de los interinos.

**2. El pareo.** Las dos pozas no contenían el mismo estímulo. La campaña
alterna PoC1 y PoC2 sobre el mismo (escenario, repetición) —§2.8— así que hay
pares, y el WER se calcula sobre la **intersección de clips de cada par**:
84 pares, 5 corridas sin pareja descartadas.

**El portón son 84 pares y se queda en 84 — decisión del 23 de septiembre de
2026, ahora sí elegida y no heredada.** El filtro es `estado == "ok"`, que en el
lote equivale a `criterios_fallidos == 0`: un semáforo de **confiabilidad**. Un
portón de `fin.valida` —"la corrida es analizable"— daría **86 pares**, dos más,
los dos del escenario 5 (`e5 r16` y `e5 r17`), rojos porque el agente de PoC2
emitió texto sin audio en T6. Ese es un defecto de la **salida del agente**, no
del estímulo: los clips del transportista sonaron completos y el STT tuvo su
oportunidad, así que en estricta doctrina el WER no debería mirar ese criterio.

Se publican las dos cifras porque la diferencia es la respuesta:

| portón | turnos por PoC | palabras | WER PoC1 | WER PoC2 | brecha |
|---|---:|---:|---:|---:|---:|
| **84 pares** (`estado == ok`) | 569 | 6,104 | **0.0600** | **0.0700** | 1.00 pp |
| 86 pares (`fin.valida`) | 581 | 6,308 | 0.0614 | 0.0710 | 0.96 pp |

Mueve **+0.14 pp** a PoC1 y **+0.10 pp** a PoC2, en la dirección conservadora —
los 2 pares son del escenario de estrés de reconocimiento, así que incluirlos
sube el WER de las dos—, no cambia el orden, no cambia la brecha de forma
apreciable y **no cambia los 39 clips comunes**. 84 se queda porque es la
población de la que sale todo lo demás del contrato, incluida la latencia, y
partir el reporte en dos poblaciones para ganar una décima no se paga. El 86 va
publicado al lado, que es lo que contesta la objeción.

**Esto no toca a PoC3.** El 84 de `docs/costos_poc3.md` es el número de llamadas
planeadas de PoC3, no un conteo de pares; la campaña ya corrió con 90 llamadas y
82 en verde, y su poza del WER son esas 82 corridas (633 turnos).

**Se intersecta por `clip_id`, no por nombre de turno.** No es lo mismo: en **6
pares** las dos PoC llegaron al mismo `T5` por ramas distintas (`alt1` contra
`alt2`), que son frases distintas. Parear por nombre habría medido el WER de una
contra el texto que sonó en la otra. Y la exclusión no siempre es de cola: en 2
casos cae **en medio**, así que el turno excluido tiene que llevarse su grupo del
STT con él o todo lo que viene después queda corrido.

Costo del pareo: 4.1% de la referencia de PoC1 y 1.8% de la de PoC2. Resultado:
**6,149 palabras idénticas en los dos lados, desbalance 0.00%.**

#### La cobertura va en el método, no como hallazgo

Que una PoC alcance turnos que la otra no **no se interpreta aquí**: el
comportamiento del agente es el vehículo, no el objeto de estudio (§4). Lo que sí
es del método es que las dos pozas contengan el mismo estímulo, que es validez de
muestra. Por eso la tabla de cobertura se publica —`cli referencia-de-clips` la
imprime— y por eso se parea.

Antes del pareo, la diferencia más grande estaba en el escenario 5, el de estrés
de reconocimiento: PoC1 llegaba a T7/T8 en el 33% de sus llamadas y PoC2 en el
**0%**, y esos dos turnos cargan "18,500" y "carta porte". Después del pareo las
dos tablas son idénticas fila por fila en los 5 escenarios.

#### Los dos WER, con su alcance declarado

Igual que las tres cifras de latencia de §2.2.8: son dos números que contestan
preguntas distintas y **ninguno reemplaza al otro**.

| | campaña de 200 | las 15 de agosto |
|---|---|---|
| **alcance** | PoC1 contra PoC2 | **las 3 PoC** — único comparable contra PoC3 |
| **referencia** | el guion **reproducido** (clip) | el guion **actuado** (memoria de quien actuó) |
| **corrección** | ninguna a mano: derivada del clip y **verificada contra el audio** (23/09/2026) | manual, 2-3 min por llamada |
| **n** | 6,104 palabras por PoC, 84 pares | 1,165 palabras por PoC, 15 llamadas |
| **transporte** | arnés local | telefonía real por Twilio |

El de la campaña es más exacto —la referencia es el audio, no el recuerdo— y
tiene cinco veces más palabras.

**Corregido el 23 de septiembre de 2026: PoC3 ya corrió el arnés.** Esta tabla
decía que el WER de agosto era "el único que alcanza a PoC3"; dejó de ser cierto
con la campaña de PoC3 del 22–23/09/2026 (90 llamadas, 82 en verde, 633 turnos
medibles). Hoy el comparable de las 3 PoC es la campaña del arnés, con la misma
referencia derivada del audio y el mismo tokenizador, y el de agosto queda como
antecedente telefónico. La comparación entre las tres se hace sobre los **39
clips comunes**, porque las pozas no son iguales: PoC1 y PoC2 reprodujeron 39
clips y PoC3 reprodujo 46.

#### El STT no es determinista, y por eso las repeticiones son muestras

Sobre el mismo `clip_id` —o sea, exactamente el mismo audio, byte por byte— el
STT produjo transcripciones distintas: hasta **14 variantes del mismo clip en
PoC1** y 12 en PoC2, y solo el 14% de los clips de PoC1 salieron siempre igual.

Es lo que justifica el `n`. Si el mismo audio diera siempre la misma hipótesis,
las 20 repeticiones de un clip serían **una** muestra repetida 20 veces y el
intervalo de confianza del WER sería una ficción. No lo son: cada repetición
mide una realización distinta del STT sobre el mismo estímulo, que es
precisamente el diseño que hace comparables a las dos PoC.

---

## 3. El contrato de captura, por PoC

`shared/metrics/` **calcula**; cada PoC **mide**. Lo que sigue es lo que cada PoC
tiene que llenar para que sus números signifiquen lo mismo.

Por turno, en la llave `latencias` del JSON de la llamada:

| Campo | PoC1 | PoC2 | PoC3 | Ancla |
|---|---|---|---|---|
| `turno_completo_ms` | ✅ | ✅ | ✅ | fin de habla → primer audio del agente |
| `fin_de_turno_ms` | ✅ | ✅ | — | fin de habla → el detector declara cerrado el turno |
| `stt_ms` | ✅ | ✅ | — | fin de habla → transcripción final |
| `llm_ttft_ms` | ⚠️ | ⚠️ | — | fin de turno → primer token — **pero no el mismo token: ver §2.7.1 antes de compararlos** |
| `tts_ttfb_ms` | ✅ | ✅ | — | primer token → primer byte de audio |
| `interrumpido` | ✅ | ✅ | ✅ | el transportista habló encima del agente |
| `es_apertura` | ✅ | ✅ | ✅ | turno de arranque en frío — **se captura, no se reporta: ver §2.7.2** |
| `veredictos_fin_de_turno` | ✅ | vacío | vacío | probabilidad del detector, si la expone |

En PoC3 solo existe el extremo a extremo: el modelo de GPT-Live es full-duplex y
no tiene STT, LLM ni TTS aislables — no es que no se hayan instrumentado, es que
no existen como etapas. Se **omiten** del JSON y del resumen en vez de aparecer
como nulos, así que un reporte de PoC3 no arrastra columnas vacías y "no aplica"
no se confunde con "no se midió".

**El cero de PoC3 se resuelve con dos relojes, y se verifica.** El alpha no
expone ningún evento de fin de habla, ni la ventana de su VAD, ni un `stop_secs`
que restar: lo único con tiempo es el `end_ms` del `turn.done` del transportista,
en la **línea de tiempo del servidor**. El primer audio del agente sí se marca
con reloj local al llegar al puente. Se anclan las dos con `session.started`
(origen de la línea del servidor) y **se comprueba el anclaje con los datos de la
propia llamada**: `veredicto_del_reloj`, en `poc3_gpt_live/app/latencia.py`, mide
el residual entre la llegada local de cada delta de audio y su `start_ms`
traducido. Si ese residual deriva a lo largo de la llamada, el origen no está
donde se supuso y la corrida sale marcada con `excluir_de_pooled`. Es el mismo
patrón con el que PoC2 verificó su propio cero (§2.2), y por la misma razón: una
afirmación sobre el interior de un servicio ajeno no se deja en la palabra de
nadie.

**`interrumpido` significa otra cosa en PoC3, y tiene que significarla.** El
contrato lo define como "el transportista habló encima del agente". En PoC1 y
PoC2 eso es un evento raro; en una arquitectura full-duplex es el **modo normal
de operación** —el modelo escucha mientras habla, por diseño—, así que el
criterio literal mandaría casi todos los turnos a la cubeta `interrumpido` y
dejaría a PoC3 sin p50 de régimen que comparar. Es el mismo fallo que PoC1 ya
cometió una vez con `InterruptionFrame`. En PoC3, `interrumpido` es **barge-in
que corta**: el transportista se encimó **y** el agente dejó de emitir audio
dentro de medio segundo. El solape que no calla a nadie no marca el turno, pero
se cuenta aparte en `gpt_live_extra.turn_taking` — es un dato propio de esta
arquitectura y tirarlo sería perder justo lo que la distingue.

**El remuestreo de PoC3 está dentro de `turno_completo_ms`.** El transporte
—Media Streams, y en campaña el arnés por WebSocket local— entrega μ-law de
8 kHz, y el modelo exige PCM16 de 24 kHz: hay una conversión en cada sentido que
ninguna otra PoC paga en ese punto del camino. Medida: **24.2 ms en régimen por
sentido** (~48 ms de ida y vuelta), con picos de 73 ms. No se resta —es parte de
lo que cuesta esta arquitectura sobre audio telefónico— y queda anotada en el
JSON de cada llamada. **La conversión es de la arquitectura, no del proveedor**:
existe igual con Twilio y sin él, porque lo que la causa es la banda de 8 kHz.

Reglas que valen para las tres:

- **El cero es el fin de habla real**, con el `stop_secs` de ese framework restado.
- **`None` nunca es cero.** Una etapa que no se midió se excluye del percentil y
  se reporta como cobertura incompleta.
- **Un delta negativo no se escribe.** Significa marcas fuera de orden; sale
  `None` con el motivo en `notas`.
- **Un turno es un intercambio**, no una pausa del VAD. La frontera es el fin de
  turno del detector.

## 4. Lo que no se mide

El resultado de la negociación no es una métrica de esta comparativa y no aparece
en ningún entregable. El motor de negociación es el **vehículo de la
conversación**: le da al WER el vocabulario de dominio que la métrica de STT
necesita y ejercita el tool calling —que sí diferencia frameworks— pero lo que el
agente negocie bien o mal es comportamiento de prompt + LLM, idéntico en las 3
porque las tres corren el mismo modelo y el mismo `v0.md`.

Decisión del 5 de agosto de 2026, registrada en CLAUDE.md. No re-litigar.
