# shared/audio/ — preparación de los clips del arnés

El arnés corre los 5 escenarios reproduciendo **clips pregrabados** en vez de
que alguien actúe el guion al teléfono. Este paquete prepara esos clips: los
degrada a banda telefónica una sola vez y deja un manifiesto con el hash de
cada uno.

Mismo criterio que `shared/metrics/`: si cada PoC degradara su audio a su
manera, el WER mediría el preprocesamiento y no el STT.

## Qué hacer, en orden

```bash
# 1. Qué grabar y con qué nombre (49 clips: 5 escenarios × 8 turnos + 9 alternativas)
python -m shared.audio.cli inventario

# 2. Grabar cada línea en shared/test_scenarios/clips_crudos/<id>.wav

# 3. Degradar y escribir el manifiesto
python -m shared.audio.cli preparar

# 4. Antes de cada campaña: ¿siguen siendo los mismos clips?
python -m shared.audio.cli verificar
```

`python -m shared.audio.cli demo` corre el flujo entero con audio sintético:
sirve para validar el código sin tener una sola grabación.

## Cómo grabar

- **WAV PCM de 16 bits**, mono o estéreo, a 16 kHz o más (48 kHz es lo ideal).
  Un `.m4a` del celular o un WAV en flotante de 32 bits se rechazan con un
  mensaje que dice qué hacer.
- Una línea por archivo, con el nombre exacto que da `inventario`.
- Di la línea y para: el silencio de los extremos se recorta solo. El silencio
  **interno** no se toca — una pausa a media frase es parte de la actuación.
- No te acerques tanto al micrófono que sature: `preparar` avisa si el pico
  pasa de -0.5 dBFS, y eso se arregla volviendo a grabar, no en software.
- **No te preocupes por el volumen.** Mientras no sature, `preparar` lo lleva al
  nivel correcto solo. Lo que sí importa es que sea parejo dentro de una toma.

## La calibración de nivel

Un clip grabado en la laptop puede salir 15 dB más fuerte o más débil que lo que
un teléfono entrega. Eso no cambia lo que *dice*, pero cambia lo que se **mide**
con él: el VAD decide por energía, así que un clip fuerte cruza el umbral antes
y mueve el cero de la latencia; y el STT degrada con la relación señal/ruido, así
que un WER sacado con audio más fuerte que el de las 15 llamadas reales no es
comparable con el de esas llamadas. Sin calibrar, el arnés mediría su propia
ganancia.

Por eso `preparar` normaliza cada clip: ajusta la ganancia —una sola, uniforme,
sin comprimir— para que el **RMS de los segmentos con voz** caiga en el objetivo.
De los segmentos con voz y no del archivo completo, porque el RMS completo cuelga
de cuánto silencio traiga la toma.

El objetivo vive en [`calibracion_nivel.json`](calibracion_nivel.json), junto a
la medición de la que salió, y es un parámetro: para moverlo se cambia ese número
y se vuelve a correr `preparar`. `verificar` caza los clips que quedaron
preparados contra un objetivo viejo — mezclarlos haría que el WER dependiera de
cuándo se grabó cada línea.

### El objetivo no significa nada sin su criterio

Medido sobre una grabación real: el "RMS de voz" del **mismo** audio se mueve
**9 dB** según qué ventanas cuenten como voz (de −25.0 con un umbral de 15 dB
bajo el pico a −34.2 con uno de 45). O sea que "−18 dBFS" es media especificación;
la otra mitad es la caída del umbral, y por eso las dos viajan juntas en el JSON
y las dos quedan en el manifiesto.

El valor es **25 dB bajo el pico**, y no se eligió a ojo: es el que reproduce el
factor de cresta de la medición de referencia (16.05 y 17.11 dB en sus dos
canales; el mismo clip da 16.97 dB con ese criterio y 19.82 con uno de 35 dB).

Ese criterio **no** es el del recorte de silencio, que usa 35 dB. Son dos
preguntas distintas y conflacionarlas costó 3 dB de error medido: para recortar
hay que ser permisivo, porque morder una fricativa final movería el cero del fin
de voz; para medir el nivel hay que ser estricto, porque incluir los colgajos de
baja energía hunde el RMS y hace pedir ganancia de más.

⚠ **Hoy el objetivo es provisional (-18.0 dBFS).** Se midió sobre una grabación
de Twilio del agente despachador: es voz humana por la misma línea telefónica,
pero de otro sistema. Los dos canales midieron -16.4 y -21.4 dBFS y -18.0 es el
punto medio. **Mientras siga provisional, cualquier WER medido con estos clips es
validación de plomería, no un dato citable.** Se cierra re-midiendo contra las
grabaciones de 2 canales de PoC1, que existen en Twilio y nunca se usaron para
métricas.

La ganancia se aplica al audio **limpio, antes** de bajar a 8 kHz y de μ-law, para
que el códec cuantice la misma amplitud que cuantizaría en una llamada real —
μ-law es logarítmico, así que su error depende del nivel de entrada. El precio es
que el objetivo no se alcanza exacto: el filtro antialias se lleva lo que hay
arriba de 4 kHz. Por eso el nivel se **verifica después** y se avisa si la
desviación pasa de ±1 dB. Con voz medida da menos de 0.1 dB; ±1 dB solo salta
cuando el espectro no es de voz.

## La degradación, y por qué va antes y no durante

Las 15 llamadas de cada PoC pasaron por teléfono real: 8 kHz y G.711 μ-law. Un
clip grabado en la laptop a 48 kHz tiene un espectro que el transportista nunca
pudo haber mandado, y un STT lo reconoce mucho mejor de lo que reconocería la
misma frase por teléfono. Medir WER con audio de estudio daría un número bonito
y sin relación con lo que se comparó.

El clip que se reproduce ya viene degradado, y la degradación corre **una sola
vez**: hacer DSP en vivo le metería su propio retraso a la latencia que se está
midiendo.

Lo que cambia entre PoC es solo cómo se entrega el mismo audio:

| PoC | Transporte | Qué recibe |
|---|---|---|
| 1 y 3 | WebSocket de Media Streams | los bytes μ-law, idénticos a los de Twilio |
| 2 | pista de audio de la sala | el PCM de 8 kHz, que LiveKit sube a 48 kHz y codifica en Opus |

Esa última fila es asimetría **de la arquitectura, no del arnés**: en una
llamada real por el trunk SIP, el mismo salto de 8 kHz a Opus lo hace LiveKit
igual. Va declarada en `docs/CONTRATO_METRICAS.md`.

## El manifiesto es la evidencia

`*.wav` y `*.ulaw` están gitignoreados: pesan y son la voz de una persona
(LFPDPPP). O sea que **los clips no se versionan y el manifiesto sí**. Es lo
único commiteable que prueba que las tres PoC reprodujeron el mismo audio; sin
él, esa afirmación del reporte sería un recuerdo.

`verificar` revisa cuatro cosas:

1. **Que estén todos** — el inventario sale del guion, no de una lista a mano.
2. **Que el audio no haya cambiado** — re-hashea los artefactos en disco.
3. **Que el guion no se haya movido debajo de los clips** — si alguien edita una
   línea de `escenarios.yaml`, el clip grabado deja de decir lo que el guion
   dice y el WER se calcularía contra una referencia que nadie pronunció. Esa
   deriva es silenciosa por naturaleza; aquí es un error con nombre.
4. **Que el nivel siga siendo el vigente** — ver la calibración, arriba.

Por clip, el manifiesto guarda el objetivo con el que se preparó, el RMS de voz
que se midió ya degradado, la ganancia aplicada, la bandera de provisional y los
dos hashes.

## Dependencias, y por qué el motor de remuestreo va al manifiesto

Solo `numpy` (está en los tres entornos) y la biblioteca estándar. `soxr` se usa
**si está** y solo al preparar: es el remuestreador de PoC1 y PoC3, pero no está
instalado en el de PoC2, y agregarlo cambiaría el `uv.lock` con el que corrieron
sus 15 llamadas. Sin `soxr` hay un respaldo en numpy puro que cubre las razones
enteras (48, 32, 24 y 16 kHz hacia 8 kHz).

Los dos motores no dan bytes idénticos, así que el manifiesto guarda con cuál se
preparó cada clip. No afecta a la comparación: los tres arneses reproducen los
mismos artefactos ya degradados, y el remuestreo no vuelve a aparecer en el
camino de ninguna llamada.

## Lo que NO está aquí

- **La reproducción.** El reproductor por turnos, el cero de fin de voz y el
  árbol de decisión viven en el arnés de cada PoC (Fase 2 en adelante).
- **El códec en vivo.** `poc3_gpt_live/app/bridge.py` tiene su propia copia del
  μ-law, congelada con su campaña. `tests/test_codec.py` verifica que las dos
  copias sigan siendo bit-idénticas, y también contra `audioop`.
