
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

## Propuestas de solución

## Experimentación y resultados

## Discusión de los resultados

## Repercusión ética de las soluciones.

El impacto ético que puede tener el reconocimiento de patrones en imágenes es considerable si analizamos los usos que puede llegar a tener trabajos como este.

- Localización de imágenes obscenas dentro de imágenes.
- Generación de imágenes para fines propagandísticos.
- Modificación de imágenes, alterando su estructura original para transmitir mensajes subliminares.

Los últimos dos puntos son posibles dado que se pueden realizar acciones que agreguen información sobre la imagen.

## Conclusiones y trabajo futuro

## Bibliografía

- [HIGH-DIMENSIONAL CONTINUOUS CONTROL USING
GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/pdf/1506.02438)
- [Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout/)
- [Actor Critic Method](https://keras.io/examples/rl/actor_critic_cartpole/)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html)
- [Deep Deterministic Policy Gradient](https://keras.io/examples/rl/ddpg_pendulum/)