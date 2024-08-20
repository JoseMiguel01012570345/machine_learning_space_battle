
# Trabajo final de la asignatura de Aprendizaje de Máquinas

Autores: 

- Yonatan Jose Guerras Perez c-411
- Jose Miguel Perez Perez c-411

##### Agradecimientos

Los autores reconocen la colaboración de **Amanda Lucía Fuentes Ponce** por lograr un diseño magnífico de las formas utilizadas en este proyecto.

## Introducción

La elección entre una partida de corta y otra de larga duración es subjetiva entre los jugadores de videojuegos, quienes prefieren partidas de duración media con una complejidad adecuada. Los mapas juegan un papel crucial en este aspecto. En este trabajo, buscamos desarrollar un mapa para videojuegos estilo arcade utilizando técnicas de machine learning para identificar formas típicas de mapas arcade en una imagen satelital. Mediante la activación o desactivación de píxeles, eliminamos partes redundantes de la imagen que no contribuyen a un mapa jugable estilo arcade. Identificar estas formas redundantes es complicado, ya que requiere jugar varias partidas y evaluar métricas específicas, las cuales pueden variar según el tipo de juego. Las métricas utilizadas son generales y consideradas fundamentales para evaluar un buen mapa:

- Batallas decisivas, ya hablaremos mas de esto.
- Caminos simples , aumentan las posibilidades de enfrentamiento.
- Que genere puntos de avanzada , quien conquista estos puntos tiene altas probabilidades de librar a su favor una batalla.
- Caminos complicadas , estos mejoran la cobertura defensiva.

Estas métricas fueron seleccionadas debido a la importancia de tener múltiples oportunidades para recuperar la posición en caso de perder una batalla decisiva. Además, se consideraron para proporcionar oportunidades de defensa y contraofensiva, creando un equilibrio dinámico en la partida.

## Motivación

La razón detrás de abordar este tema es la alta inversión requerida para producir mapas útiles, especialmente cuando su utilidad es difícil de determinar. Sería más conveniente diseñar un mapa que cumpla con ciertas restricciones y se genere automáticamente, reduciendo significativamente los costos de producción. Este problema se clasifica como un problema de satisfacción de restricciones (CSP), cuyo costo computacional es considerable debido a la gran cantidad de variables implicadas, incluidos los píxeles que componen la imagen.

## Problemática

Nuestra tarea es transformar una imagen satelital en un mapa jugable estilo arcade, manteniendo características visuales arcade, como píxeles claros y bordes poco definidos. Aunque existen estudios sobre la generación de mapas 3D y la reconstrucción de mapas 2D a partir de imágenes satelitales, la aplicación de técnicas de machine learning para identificar formas arcade en imágenes satelitales es un área no explorada.

La esteganografía, que implica ocultar información dentro de otros mensajes o objetos para evitar su detección, es similar en algunos aspectos, pero nuestro enfoque no implica ocultar información sino identificar patrones visuales. Ejemplos de esteganografía se puede encontrar [aqui](https://infosecwriteups.com/steganography-ctfs-73f7b310b1f7?gi=9291a7cee537).

Algunos referentes sobre el tema se listan:

 - [A Robust Bio-Signal Steganography With Lost-Data Recovery Architecture Using Deep Learning](https://ieeexplore.ieee.org/document/9853601)
 - [Squint Pixel Steganography: A Novel Approach to Detect Digital Crimes and Recovery of Medical Images](https://www.igi-global.com/gateway/article/163348#pnlRecommendationForm)

Estos artículos emplean técnicas de machine learning para recuperar imágenes manipuladas mediante esteganografía, pero nuestro objetivo es identificar formas arcade en imágenes satelitales sin cifrado intencional.

## Hipótesis

- ¿Es posible modificar una imagen A en otra imagen B para detectar formas de la imagen B en A, sin añadir información a A?
- ¿Qué tan eficiente puede ser este proceso?

## Objetivo general y específicos

Nuestro objetivo es convertir la imagen original en blanco y negro, activando o desactivando píxeles según la predominancia de blancos o negros. Si la imagen es predominantemente blanca, activamos píxeles; si es negra, los desactivamos. El objetivo es "agitar" la imagen hasta que se revelen formas que representen un mapa jugable. Necesitamos entrenar un modelo que decida cómo tratar cada píxel de la imagen de entrada.

Para lograrlo, entrenaremos un modelo de aprendizaje por refuerzo que interactúa con un entorno dinámico para alcanzar un objetivo específico, aprendiendo a tomar decisiones óptimas a través de la interacción con su entorno y la recepción de feedback positivo o negativo.

## Propuestas de solución

Utilizamos aprendizaje por refuerzo, un sistema que interactúa con un entorno dinámico para aprender a través de la exploración y la explotación de conocimientos previamente adquiridos. El agente aprende a maximizar una medida de recompensa acumulada a través de la interacción con su entorno.

### Componentes Clave de un Agente RL

1. **Agente**: Un sistema que realiza acciones en un entorno con el objetivo de aprender una política que maximice la recompensa acumulada.
2. **Entorno**: El mundo en el que opera el agente, que puede ser cualquier cosa desde un juego de ajedrez hasta un mercado financiero real.
3. **Estado**: Una descripción completa del entorno tal como es percibida por el agente en un momento dado.
4. **Acción**: Una elección realizada por el agente que afecta el estado del entorno.
5. **Recompensa**: Retroalimentación del entorno al agente después de realizar una acción, típicamente un valor numérico que indica cuán bueno fue el resultado de la acción.
6. **Política**: Una función que determina qué acción tomará el agente en función del estado actual.
7. **Valor de Estado**: La expectativa de recompensa futura desde un estado dado, bajo una política específica.
8. **Función de Valor-Q**: La expectativa de recompensa futura para una acción específica en un estado dado, bajo una política específica.
9. **Modelo del Entorno**: Conocimiento interno del agente sobre cómo el entorno responde a sus acciones.

### Definición Formal

Un agente de RL se puede definir formalmente como un sistema que busca maximizar la recompensa acumulada $G_t$, donde $G_t$ es la suma de recompensas $R_{t+1}, R_{t+2}, ..., R_T$ descontadas al presente, obtenidas durante un episodio que comienza en el tiempo $t$:

$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t-1} R_T$

donde $\gamma \in [0, 1]$ es el factor de descuento que determina el valor presente de las recompensas futuras, y $T$ es el tiempo final del episodio.

### Elementos Fundamentales

- **Espacio de Estados** $S$: Conjunto de todos los estados posibles en los que puede encontrarse el entorno.
- **Espacio de Acciones** $A$: Conjunto de todas las acciones posibles que el agente puede realizar.
- **Función de Transición de Estado** $P$: Probabilidad de transición entre estados, $P_{ss'}^a = Pr\{s'|s,a\}$, la probabilidad de llegar al estado $s'$ al tomar la acción $a$ en el estado $s$.
- **Función de Recompensa** $R$: Mapea pares de estado y acción a valores esperados de recompensa inmediata, $R_s^a = E[R_{t+1 | S_t=s, A_t=a]}]$.

### Proceso de Aprendizaje

El aprendizaje se realiza a través de la interacción del agente con el entorno, que sigue un ciclo de:

1. Observar el estado actual $S_t$.
2. Seleccionar una acción $A_t$ según la política $\pi(a|s)$, que es la probabilidad de elegir la acción $a$ en el estado $s$.
3. Tomar la acción $A_t$, observar la recompensa $R_{t+1}$ y el nuevo estado $S_{t+1}$.
4. Actualizar la política y/o el valor estimado de estados y acciones basándose en la experiencia $(S_t, A_t, R_{t+1}, S_{t+1})$.

### Tipos de Problemas

- **Episódicos vs. Continuos**: Los problemas pueden ser episódicos (con inicio y fin claro) o continuos (sin un final definido).
- **Determinísticos vs. Estocásticos**: En entornos determinísticos, las transiciones de estado son predecibles; en estocásticos, son aleatorias.

En nuestro caso hemos aplicado el episódico.

### Políticas

Una política $\pi$: $S \rightarrow A$ define el comportamiento del agente, mapeando estados a acciones. El objetivo es encontrar una política óptima $\pi^*$ que maximice la recompensa esperada.

### Función de Valor

La función de valor de una política $\pi$, $v_\pi(s)$, es la recompensa esperada desde estado $s$ siguiendo $\pi$:

$v_\pi(s) = E_\pi[G_t | S_t = s]$

Donde $E_\pi$ denota la expectativa bajo la política $\pi$.

### Función Q

La función de valor acción $q_\pi(s, a)$ es la recompensa esperada de tomar acción $a$ en estado $s$ y luego seguir $\pi$:

$q_\pi(s, a) = E_\pi[G_t | S_t = s, A_t = a]$

### Aprendizaje por Refuerzo

El objetivo es encontrar $\pi^*$ tal que para todo $s$ en $S$, $v_{\pi^*(s) \geq v_\pi(s)}$ para toda política $\pi$.

### Métodos de Solución

- **Value Iteration**: Actualiza iterativamente las estimaciones de valor hasta converger a las óptimas.
- **Policy Iteration**: Mejora la política basada en la función de valor actual hasta converger a $\pi^*$.

### Algoritmos Clave

- **Q-Learning**: Aprende la función Q directamente, sin necesidad de un modelo del entorno.
- **Deep Q-Network (DQN)**: Usa redes neuronales para aproximar la función Q.
- **Actor-Critic**: Combina aprendizaje por refuerzo con críticos que evalúan acciones.

En nuestro trabajo hemos usado **DQN** en un actor **Actor-Critic**

#### Detalles de implementacion de la politica

La implementación de la política es crucial para garantizar la rápida convergencia del agente. Seleccionamos un conjunto de imágenes que esperamos encontrar y que consideramos que nos asegurarán mapas acordes con las métricas descritas. Para indicar al agente cuán cerca está de generar la imagen objetivo, llevamos ambas imágenes a un espacio común y calculamos el ángulo entre la imagen y la horizontal de ese espacio, luego calculamos el ángulo comprendido entre ambas imágenes. Cuanto mayor sea el ángulo, más similares son las imágenes.

Nota: Este método presenta claramente problemas de eficiencia. Dado que se requieren múltiples imágenes para describir las líneas que se buscan en la imagen objetivo, y por ende, se debe seleccionar un gran número de imágenes para encontrar similitudes, esto resulta en costos computacionales significativos.

Si la acción seleccionada por el agente incrementa la mayor recompensa recibida en el episodio, entonces esa acción es ejecutada. La justificación detrás de esta idea se fundamenta en los siguientes argumentos:

Si al realizar una acción nos alejamos de la solución, entonces debemos revertir dicha acción y ejecutar la esperada. Evitamos actuar en el medio para no contaminar el nuevo estado, y así evitar generar más errores en la siguiente acción.

Al ajustar la máxima recompensa, es crucial considerar un aspecto importante: si para completar una tarea se espera que el agente realice un conjunto de acciones, digamos $a_0,a_1,...,a_k$ , las recompensas se otorgan en orden no decreciente y además, el orden de las acciones no importa. Sin embargo, durante el proceso de entrenamiento, puede suceder que el agente realice $a_i$ y luego no pueda realizar $a_{i−1}$ , lo que deriva en tiempos de convergencia más prolongados.

### Psceudocodigo

1. Inicializar arbitrariamente $Q(s, a)$ para todos $s \in S$, $a \in A$ , $best=0$ .
2. Repetir hasta convergencia:
   - Seleccionar $A_t$ usando política derivada de \(Q\) (por ejemplo, $\epsilon$-greedy).
   - Tomar acción $A_t$, observar $R_{t+1}, S_{t+1}$.
   - If $best < R_{t+1}$:
     -   Aplicar accion en el medio
     -   $best=R_{t+1}$
   - Actualizar $Q(S_t, A_t)$ basado en $R_{t+1}$ y $max_a Q(S_{t+1}, a)$.

Nota:
- El espacio visible por el agente es una matriz de 100x100 y se pueden activar o desactivar pixeles , por tanto tenemos un total de $20000$ acciones
- Por cada episodio desplazamos la ventana de vision del agente tal que el agente logre conectar puntos del episodio anterior con puntos del nuevo episodio


## Experimentación y resultados

## Discusión de los resultados

## Repercusión ética de las soluciones.

El impacto ético que puede tener el reconocimiento de patrones en imágenes es considerable si analizamos los usos que puede llegar a tener trabajos como este.

- Localización de imágenes obscenas dentro de imágenes.
- Generación de imágenes para fines propagandísticos.
- Modificación de imágenes, alterando su estructura original para transmitir mensajes subliminares.

Los últimos dos puntos son posibles dado que se pueden realizar acciones que agreguen información sobre la imagen.

## Conclusiones y trabajo futuro

En nuestra hipótesis, nos enfrentamos a un problema al cual hemos propuesto una solución como primer enfoque. Es importante destacar que implementar esta arquitectura de agentes conlleva diversos desafíos, tales como:

- Estabilidad: Asegurar la convergencia y gestionar la exploración/explotación.
- Escalabilidad: Manejar grandes espacios de estado/acción.
- Interpretabilidad: Explicar las decisiones del agente.

No obstante, constituye una herramienta altamente potente para describir fenómenos que no somos capaces de describir con ejemplos sencillos de clasificación u otras tareas que involucran el aprendizaje automático. Esto es especialmente relevante dado que estos sistemas tienden a requerir mucha computación y, por ende, tiempo de entrenamiento, que en casos menos graves puede llegar hasta 3 días de entrenamiento y en problemas más complejos, hasta meses de entrenamiento en grandes TPU.

Respecto a la eficiencia de nuestro algoritmo, hemos logrado un entrenamiento de 3 días con las mejoras mencionadas en la sección Propuesta de Solución.

Este problema tiene poca relación con otras técnicas del campo del aprendizaje automático y, por tanto, es probable que solo una solución con aprendizaje por refuerzo sea viable. Sin embargo, este problema se puede resolver con métodos deterministas dado que conocemos la política del medio. Aunque, la incertidumbre que aportan las redes neuronales a cada acción hace que la experiencia de un mapa generado sea más rica.

## Bibliografía

- [HIGH-DIMENSIONAL CONTINUOUS CONTROL USING
GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/pdf/1506.02438)
- [Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout/)
- [Actor Critic Method](https://keras.io/examples/rl/actor_critic_cartpole/)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html)
- [Deep Deterministic Policy Gradient](https://keras.io/examples/rl/ddpg_pendulum/)