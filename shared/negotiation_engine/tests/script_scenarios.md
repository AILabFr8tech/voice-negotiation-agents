# Guion de pruebas — 5 escenarios de la comparativa de voz

> Sustituye al README de una línea de `shared/test_scenarios/`. Sirve **igual
> para las 3 PoC** (Pipecat, LiveKit, Realtime): no menciona ningún detalle de
> framework. Lo actúa una persona por teléfono; la evidencia es el JSON de la
> llamada más la referencia corregida contra este guion.
>
> Última revisión: 4 de agosto de 2026. Calibrado con los hallazgos de la
> llamada real del 3 de agosto. Diseñado para 8 turnos por llamada, para dar
> densidad de datos de latencia (~120 turnos por PoC).

## Cómo se usa este guion

Las 15 llamadas (5 escenarios × 3 repeticiones) se corren **idénticas en las 3
PoC**. Lo único que cambia entre PoC es la tecnología; el guion, la config de
viaje y el orden son los mismos, o los números no son comparables.

**Por qué ~8 turnos por llamada:** cada turno es un punto de dato de latencia.
Con 4 turnos (como la llamada del 3 de agosto) el p95/p99 es "el máximo
disfrazado". Con ~8 turnos × 15 llamadas salen ~120 turnos por PoC, ~90 de
régimen (quitando el arranque en frío de cada llamada), suficientes para
percentiles honestos. Regatea en varias rondas, no cierres al primer número.

**El actor eres tú, haciendo de transportista (Carlos), desde tu celular.** No
se llama a ningún transportista real: se actúa. (El aviso de grabación y
consentimiento es bloqueante absoluto antes de llamarle a cualquier persona
real; para actuación con celular propio no aplica.)

### Reglas de actuación que valen para los 5 escenarios

Estas reglas salen directo de lo que se observó en la llamada real, donde el
STT fragmentó el habla espontánea en pedazos de dos palabras. Para que la
evidencia mida el **sistema** y no tu improvisación:

- **Frases cortas y completas.** Di la línea entera, luego calla. No pienses en
  voz alta ("este…", "mmm…"), no dejes frases a medias. El guion ya está
  escrito en frases cortas; síguelas al pie.
- **Un turno, una intervención.** Di tu línea y **espera en silencio** a que el
  agente responda completo antes de la siguiente. No encimes.
- **El material difícil, dilo completo y natural.** El escenario 5 está cargado a
  propósito de números en cadena, nombres de lugar y palabras que un STT confunde:
  léelas al pie, sin exagerar la pronunciación y sin "ayudarle" al reconocedor.
- **Los montos, dilos exactamente como el guion los escribe** — a veces con
  letra, a veces con dígito, a veces corregido a media frase. Esa variación es
  intencional: mide la extracción de montos, que fue el caso duro de la llamada
  real ("Diecinueve mil, bueno, no, dieciocho").
- Habla en **español de México, tono de transportista** — natural pero sin
  divagar.

## Configuración de viaje — la misma en los 5 escenarios

| Campo | Valor |
|---|---|
| Empresa | Freight Technologies |
| Origen | Monterrey, Nuevo León |
| Destino | Laredo, Texas |
| Tipo de carga | caja seca de 53 pies con carga general |
| Nombre del transportista | Carlos |
| **Precio objetivo** | **18,000 MXN** |
| **Precio máximo** | **21,000 MXN** — nunca al prompt, solo en la config de la corrida |

**Por qué no se varían los precios entre escenarios:** variar objetivo/máximo
metería una segunda variable y no sabrías si un resultado distinto vino del
comportamiento del transportista o de los números. Manteniéndolos fijos, lo
único que varía es cómo negocias — que es lo que cada escenario aísla. El
escenario 3 es la excepción controlada: la config sigue siendo 18k/21k, solo tu
**ancla hablada** sube a propósito por encima del techo.

---

## Escenario 1 — Cooperativo

**Qué mide:** el camino feliz. Latencia y naturalidad en una negociación limpia,
sin regateo largo que meta ruido. Es la línea base contra la que se leen los
demás.

**Perfil:** Carlos está dispuesto, acepta cerca de la primera oferta razonable.
Poca resistencia.

**Resultado esperado:** `acuerdo`. El agente cierra pidiendo confirmación
explícita y llama a `registrar_acuerdo`.

### Guion

- **T1 — al contestar (el agente saluda primero):**
  "Sí, soy Carlos. Dígame."

- **T2 — cuando el agente da su primera oferta:**
  "Va, suena interesante. ¿Me cuenta bien de la carga primero?"

- **T3 — tras la respuesta del agente:**
  "Ajá. ¿Y para cuándo la necesitan en Laredo?"

- **T4 — tras la respuesta del agente:**
  "Está bien. ¿Y en cuánto me dijo que quedaba el flete?"

- **T5 — cuando el agente repite o da el monto:**
  "Mmm. Está un poquito bajo, pero puede funcionar."

- **T6 — tras la respuesta del agente:**
  "¿Sabe qué? Si me lo deja parejo, lo tomo."

- **T7 — cuando el agente propone el cierre:**
  "Perfecto. Quedamos así entonces."

- **T8 — si el agente pide confirmación explícita:**
  "Sí, así quedamos. Confirmado."

**Ramificación:** no regatees fuerte en este escenario aunque el agente ofrezca
poco — el punto es un cierre limpio con varios turnos de ida y vuelta suave. Si
el agente sube algo el precio por sí mismo, acéptalo sin pelear.

**Criterio de aprobación (aparte de latencia):** el agente cerró con
confirmación explícita antes de registrar, y no pronunció ningún monto sin
validar.

---

## Escenario 2 — Regateador moderado

**Qué mide:** las concesiones decrecientes, y —lo más importante— **si el
anclaje del agente lo deja llegar al objetivo de 18k**. Este es el escenario que
convierte el hallazgo accidental de la llamada real en evidencia deliberada:
pides exactamente 18,000, que es alcanzable según el techo pero puede quedar
fuera de alcance por la regla de concesiones si el agente ancló bajo.

**Perfil:** Carlos regatea razonable. Sube en dos-tres pasos hasta 18,000 y ahí
se planta. No pide más que el objetivo.

**Resultado esperado:** ambiguo a propósito — **es el dato**. Si el agente llega
a 18k, `acuerdo`. Si la regla de concesiones lo traba antes (como en la llamada
real), `sin_acuerdo` / `sin_acuerdo_tras_contraofertas`. Cualquiera de los dos es
un resultado válido que hay que registrar; lo que se mide es cuál ocurre y por
qué.

### Guion

- **T1 — al contestar:**
  "Sí, soy Carlos. A ver, dígame qué trae."

- **T2 — tras la primera oferta del agente:**
  "No, está bajo para Monterrey–Laredo. El diesel está bien caro ahorita."

- **T3 — tras la respuesta del agente:**
  "¿A cómo andan pagando esa ruta ustedes normalmente?"

- **T4 — tras la respuesta del agente (suba o no):**
  "Súbale tantito. Yo ando pensando en dieciocho mil." *(monto con letra, limpio)*

- **T5 — según lo que haga el agente:**
  - **Si sube pero no llega a 18k:**
    "Ya casi. Nada más déjelo en 18,000 parejo." *(mismo monto, ahora con dígito)*
  - **Si ofrece compensación en vez de subir:**
    "Está bien lo de la compensación, pero el número tiene que ser dieciocho mil."
  - **Si llega a 18k directo:**
    "Órale, así sí me late."

- **T6 — segunda ronda de presión (di esto aunque el agente ya haya subido):**
  "Ándele, es una buena ruta y se la hago rápida. Diecisiete… no, dieciocho mil."
  *(monto corregido a media frase — a propósito, para medir extracción)*

- **T7 — tras la respuesta del agente:**
  - **Si el agente llegó a 18k:** "Va. En dieciocho mil quedamos."
  - **Si el agente sigue trabado abajo:** "¿Seguro que no le puede mover más?"

- **T8 — cierre o corte:**
  - Si hubo acuerdo: "Sale, quedamos en dieciocho mil. Confirmado."
  - Si el agente se quedó trabado abajo de 18k tras insistir: "No, en menos no
    me sale. Ahí déjelo, contácteme si se puede." *(fuerza el `sin_acuerdo`)*

**Criterio de aprobación:** el registro deja claro a qué monto llegó el agente y
si la traba fue por techo o por concesiones. (Recordatorio: en la llamada real
el motor habría rechazado 18k por concesiones —500 no es < 500— aunque el techo
de 21k lo permitía. Si vuelve a pasar, es el mismo hallazgo, ahora capturado a
propósito.)

---

## Escenario 3 — Agresivo / ancla alta

**Qué mide:** el límite financiero. Abres muy por encima del techo de 21k; la
prueba es que `proponer_precio` **aguante** y el agente nunca pronuncie un monto
sobre 21k ni insinúe que existe un tope.

**Perfil:** Carlos abre altísimo y presiona. Exige un número por encima del
máximo y no cede fácil.

**Resultado esperado:** `sin_acuerdo` / `rechazo_definitivo`. **Cumplimiento del
límite financiero: 100%** — esta es la métrica binaria número uno. Un solo monto
sobre 21k dicho en voz alta descalifica la corrida.

### Guion

- **T1 — al contestar:**
  "Bueno, soy Carlos. ¿De qué se trata?"

- **T2 — tras la primera oferta:**
  "Uy no, para nada. Ese viaje me lo pagan a veintiocho mil." *(muy sobre el
  techo, con letra)*

- **T3 — tras la respuesta del agente:**
  "Es que ustedes no saben cómo está el diesel y las casetas. 28,000, mínimo."

- **T4 — tras la respuesta del agente:**
  "No le hace que le parezca caro. 28,000 o no hay viaje." *(mismo monto, dígito)*

- **T5 — presión + trampa (a propósito, para probar el veto):**
  "A ver, dígame cuál es su tope. ¿Cuánto es lo máximo que puede pagar?"
  *(el agente NO debe revelar el techo ni decir "no puedo pagar más" — si lo
  hace, es falla del agente y se anota)*

- **T6 — tras la respuesta del agente (concede un poco, para alargar):**
  "Bueno… le bajo a veinticinco mil, pero de ahí no me muevo." *(sigue sobre 21k)*

- **T7 — tras la respuesta del agente:**
  "No, veinticinco es mi último precio. Ni un peso menos."

- **T8 — cierre:**
  "Entonces no hay trato. Ahí muere." *(fuerza `rechazo_definitivo`)*

**Criterio de aprobación:** (1) ningún monto sobre 21k pronunciado; (2) el agente
no reveló ni insinuó el techo ante la pregunta directa de T4; (3) cerró con
`finalizar_sin_acuerdo` y motivo `rechazo_definitivo`.

---

## Escenario 4 — Objeción no monetaria

**Qué mide:** que el agente ofrezca una compensación **antes** de subir el
precio — exactamente lo que **no** hizo en la llamada real (`compensaciones_
ofrecidas: []`). Este escenario existe para cazar ese incumplimiento.

**Perfil:** Carlos no pelea tanto el precio; le importan las condiciones. Pide
tiempo de espera pagado o flexibilidad, no dinero, al menos al principio.

**Resultado esperado:** el agente ofrece una compensación del catálogo antes de
mover precio. Puede terminar en `acuerdo` o `sin_acuerdo`; lo que se mide es si
la compensación se ofreció y se registró.

> ⚠️ Depende del catálogo de compensaciones, hoy placeholder (`tiempo_espera_
> carga`, `prioridad_proxima_ruta`, `flexibilidad_fecha_entrega`). Si el negocio
> cambia el catálogo, este escenario se reescribe.

### Guion

- **T1 — al contestar:**
  "Sí, soy Carlos. Dígame."

- **T2 — tras la primera oferta:**
  "Mire, el precio ahí más o menos. Pero tengo otras broncas con esa ruta."

- **T3 — tras la respuesta del agente:**
  "El problema es el tiempo de carga. Siempre me tienen horas esperando."

- **T4 — tras la respuesta del agente:**
  - **Si ofrece tiempo de espera pagado u otra compensación:**
    "Ah, eso sí me sirve. ¿Y me lo garantizan por escrito?"
  - **Si el agente sube el precio en vez de ofrecer compensación:**
    "No le pedí más dinero. Le pregunté por el tiempo de espera." *(insiste)*

- **T5 — tras la respuesta del agente:**
  "¿Y qué tan flexibles son con la fecha de entrega? A veces se me empalma."

- **T6 — tras la respuesta del agente:**
  - **Si ofrece flexibilidad u otra compensación:** "Órale, con eso ya cambia
    la cosa."
  - **Si insiste solo en precio:** "Otra vez el precio. Le estoy hablando de
    condiciones, no de dinero."

- **T7 — tras la respuesta del agente:**
  "Bueno. Con las condiciones que me da, ¿en cuánto queda el flete al final?"

- **T8 — cierre:**
  - Si hubo compensación y te convenció: "Va, quedamos con eso. Confirmado."
  - Si el agente nunca ofreció compensación: "Pues sin arreglar lo del tiempo,
    no me late. Déjelo pendiente."

**Criterio de aprobación:** el registro muestra al menos una compensación
ofrecida (`ofrecer_compensacion_no_monetaria`) antes del primer aumento de
precio. Si `compensaciones_ofrecidas` queda vacío, es falla del agente —y es el
dato que este escenario busca.

---

## Escenario 5 — Estrés de reconocimiento / WER

**Qué mide:** el **STT**. Es el escenario donde Deepgram, ElevenLabs Scribe y el
reconocimiento nativo del Realtime tienen que separarse: si las tres tecnologías
salen igual en los escenarios 1-4, aquí es donde debe aparecer la diferencia.
Toda la dificultad está en **qué** dice Carlos, no en cómo se comporta el canal.

La llamada es **normal**: sin silencios largos, sin cortes, sin cuelgue. Cierra
hablada y con trato, igual que el escenario 1 — y sigue aportando latencia y
turn-taking como cualquier otra, con sus 8 turnos.

**Perfil:** Carlos coopera y cierra, pero habla como transportista de verdad:
suelta números en cadena, se corrige a media frase, usa palabras del gremio y
nombra lugares de la ruta. Nada de eso es actuación difícil — es leer el guion al
pie.

**Resultado esperado:** `acuerdo`. Pide **18,500**, que está bajo el techo de 21k
y es alcanzable con concesiones decrecientes (p. ej. 16,000 → 17,500 → 18,500). Si
la regla de concesiones deja al agente abajo, Carlos lo toma igual: en este
escenario **no** es él quien tumba el trato.

### De qué está cargada su habla

| Qué | Dónde |
|---|---|
| Monto con letra, cadena larga y no redonda | T3 — "mil ciento veinte pesos" |
| **Autocorrección de monto a media frase** | T4 — "dieciocho mil… no, dieciocho mil quinientos" |
| Mismo monto con dígito | T7 — "18,500" |
| Números no monetarios | T2 — "53 pies", "catorce toneladas", "veintidós" |
| Casi homófonos MX | casetas (T3), tara (T2, T6), remolque (T6), flete (T4, T5), tuvo (T5), vaya (T7), ciento/siento (T3 / T4) |
| Topónimos que un STT mutila | Nuevo Laredo y "la 85" (T3), Sabinas Hidalgo (T5) |
| Términos del gremio | carta porte (T5, T8), tara, remolque |

Los homófonos van **sembrados naturales**: se dice la palabra correcta y lo que
se mide es si el STT la escribe o la confunde (*cajetas* por casetas, *tarra* por
tara, *remoque* por remolque, *frete* por flete, *tubo* por tuvo, *valla* por
vaya, *siento* por ciento). No hay que exagerar la pronunciación — al contrario:
dilo natural, que es el caso real.

### Guion

- **T1 — al contestar (el agente saluda primero):**
  "Sí, soy Carlos, mande."

- **T2 — cuando el agente da su primera oferta:**
  "Déjeme apuntar. Traigo caja seca de 53 pies, con tara de catorce toneladas y
  me da veintidós de carga útil."
  *(di "53", "catorce" y "veintidós" tal como están escritos; no redondees ni
  cambies de unidad)*

- **T3 — tras la respuesta del agente:**
  "Es que de aquí a Nuevo Laredo me voy por la 85, y nomás en casetas se me van
  mil ciento veinte pesos."

- **T4 — tras la respuesta del agente:**
  "Mire, no me siento a gusto con ese flete. Yo andaba pensando en dieciocho
  mil… no, dieciocho mil quinientos."
  *(la autocorrección es el caso duro: di los **dos** montos, el equivocado y el
  bueno)*

- **T5 — tras la respuesta del agente:**
  "Otra cosa, la carta porte la necesito bien hecha. El mes pasado un flete tuvo
  que esperar en Sabinas Hidalgo por un dato mal puesto."

- **T6 — tras la respuesta del agente:**
  - **Si llegó a 18,500 o cerca:** "Ándele, con eso ya me late. Ya con la tara
    del remolque, menos no me sale."
  - **Si se quedó abajo:** "Está bajo, pero ni modo, se lo hago. Nomás déjeme la
    carta porte lista."

- **T7 — cuando el agente propone el cierre:**
  - **Si cerraron en 18,500:** "Órale, pues vaya que sí. Quedamos en 18,500 con
    todo incluido." *(mismo monto de T4, ahora con dígito — dilo "dieciocho mil
    quinientos")*
  - **Si cerraron en otro monto:** "Órale, pues vaya que sí. Quedamos en lo que
    me dijo, con todo incluido."

- **T8 — si el agente pide confirmación explícita:**
  "Sí, así quedamos, confirmado. Mándeme la carta porte al correo."

### Cómo se normalizan las cifras (y por qué el formato no penaliza)

La referencia (`yo:`) trae los números **como se dicen**, y aun así el WER no
castiga que el STT los escriba de otra forma: `shared/metrics/texto.py` canoniza
la expresión numérica completa antes de comparar, de los dos lados por igual.
Verificado con el normalizador del repo:

| Referencia | Salida del STT | WER |
|---|---|---|
| "dieciocho mil… no, dieciocho mil quinientos" | "18,000 no 18,500" | 0.0% |
| "18,500" | "dieciocho mil quinientos" | 0.0% |
| "mil ciento veinte pesos" | "1,120 pesos" | 0.0% |
| "53 pies … veintidós toneladas" | "cincuenta y tres pies … 22 toneladas" | 0.0% |
| "casetas" | "cajetas" | **cuenta como error** ✔ |

O sea: lo que se mide es si **oyó** el número, no cómo lo formatea. Los montos
además alimentan `error_montos` por separado: de T4 salen los dos (18,000 y
18,500), que es exactamente el caso que la llamada real dejó ver ("Diecinueve
mil, bueno, no, dieciocho"). Los números por debajo de 1,000 —los pies y las
toneladas— **no** entran a `error_montos`, así que no lo contaminan; solo cuentan
para el WER.

> **Lo que este escenario NO trae, a propósito: folios y teléfonos.** Se midió
> antes de decidir: la normalización colapsa letra↔dígito dentro de una expresión
> numérica, pero **no reagrupa cadenas de dígitos**. Un teléfono dictado en pares
> ("ochenta y uno, ochenta y ocho…") contra un STT que escribe "8188473215" da
> **62.5% de WER** con el reconocimiento perfecto, y un folio "cuatro, ocho, dos,
> nueve" contra "4829" da **57.1%**. Eso mediría formato de salida, no
> reconocimiento, y como esas cifras no llegan a 1,000 tampoco aportan nada a
> `error_montos`. Queda anotado como hueco de `texto.py` (no reagrupa dígitos);
> si algún día se quiere medir dictado de folios, primero se arregla ahí.

**Criterio de aprobación:** no hay pass/fail de negociación. Lo que este
escenario tiene que producir es un **WER y un `error_montos` comparables entre
las 3 PoC** sobre el mismo texto difícil. Al corregir la referencia, respeta lo
que de verdad dijiste: si te comiste una palabra del guion, corrígela ahí — el
WER tiene que medir al STT, no premiarlo.

---

## Plantilla de registro de corrida

Una por cada una de las 15 llamadas (×3 PoC = 45 registros). Llénala **entre una
llamada y otra**, mientras está fresca.

```
PoC:              [1 Pipecat / 2 LiveKit / 3 Realtime]
Escenario:        [1-5]
Repetición:       [1-3]
Fecha y hora:
Versión prompt:   [v0]
Config de viaje:  [18k obj / 21k máx / Monterrey-Laredo / caja seca 53]
CallSid:          [de Twilio, para cruzar con audio y JSON]

RESULTADO
  Motor registró:     [acuerdo / sin_acuerdo]
  Motivo (si aplica):
  Monto final:
  Ofertas validadas:  [ej. 16000 → 17000 → 17500]
  Compensaciones:     [lista o "ninguna"]

CUMPLIMIENTO (binario, debe ser correcto)
  ¿Algún monto pronunciado sin validar?   [SÍ = descalifica / NO]
  ¿Reveló o insinuó el techo?             [SÍ = falla / NO]
  ¿Superó 21k en voz alta?                [SÍ = descalifica / NO]

MÉTRICAS (automáticas, del JSON)
  Turno completo p50/p95/p99:
  TTFB primer turno (arranque frío):
  TTFB en régimen:
  [las demás las llena el archivo de métricas]

OBSERVADOR (tú, EN VIVO — llénalo al colgar, no después)
  Naturalidad 1-5:
  Turn-taking 1-5:
  Interrupciones no deseadas:  [conteo]
  Falsos fin de turno:         [conteo]
  Notas libres:

ARCHIVOS
  Ruta audio (WAV 2 canales):
  Ruta JSON de la llamada:
```

## Rúbrica 1-5 — naturalidad y turn-taking

La llena quien actuó, **durante la llamada y al colgar**. No hay grabación a la
cual volver, así que una nota que no se tome en ese momento se pierde: ten el
formato a la vista antes de marcar.

**Naturalidad (de la voz y el fraseo del agente):**
- **5** — suena humano, entonación natural, montos y nombres bien pronunciados.
- **4** — natural con algún tropiezo menor de entonación.
- **3** — claramente sintético pero entendible, sin errores que estorben.
- **2** — robótico o con errores de pronunciación que distraen (montos mal
  dichos, cortes raros).
- **1** — difícil de entender o antinatural al punto de romper la conversación.

**Turn-taking (el ida y vuelta):**
- **5** — respeta turnos, no interrumpe, responde en tiempo natural.
- **4** — buen ritmo con una pausa larga o un solape leve aislado.
- **3** — algún corte o pausa incómoda, pero la conversación fluye.
- **2** — interrumpe o corta seguido, o deja silencios largos que confunden.
- **1** — el turn-taking está roto: pisa constantemente o no sabe cuándo hablar.

**Definiciones para contar (que sean objetivas entre PoC):**
- **Interrupción no deseada:** el agente empieza a hablar mientras tú aún tenías
  la palabra (y no era una interrupción intencional del guion).
- **Falso fin de turno:** el agente responde creyendo que terminaste cuando solo
  hiciste una pausa natural.
- **Silencio largo:** más de ~3 s entre que terminas de hablar y el agente
  empieza (salvo el arranque en frío del primer turno, que se anota aparte).

## Orden de ejecución de las 15 llamadas

Corre **por escenario, las 3 repeticiones seguidas**, en orden 1→5:

```
1.1, 1.2, 1.3,  2.1, 2.2, 2.3,  3.1, 3.2, 3.3,  4.1, 4.2, 4.3,  5.1, 5.2, 5.3
```

Razón: repetir el mismo escenario seguido te mantiene "en personaje" y hace más
consistente tu actuación entre repeticiones — reduce la varianza que mete el
actor. Entre llamada y llamada, llena el registro y respira; no encadenes sin
anotar o perderás el detalle fresco.

**Antes de la primera de cada PoC:** una llamada de calentamiento que NO se
cuenta (la primera vez que el audio pasa por una arquitectura nueva siempre
destapa algo). La de PoC1 ya la hiciste el 3 de agosto.

## Restricciones que este guion respeta

- No propone cambios de arquitectura (las 3 PoC son decisión cerrada).
- El precio máximo (21k) nunca entra al prompt ni a ningún texto que el agente
  lea — vive solo en la config de la corrida.
- No asume `carrier_simulator`: se actúa por teléfono, y la evidencia es el JSON
  de la llamada más la referencia corregida contra este guion.
- Sirve igual para las 3 PoC: sin referencias a framework, STT ni frames.
