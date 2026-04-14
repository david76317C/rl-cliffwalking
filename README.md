# 🧗 RL CliffWalking

Agentes de aprendizaje por refuerzo entrenados sobre el ambiente **CliffWalking-v1** de Gymnasium. El proyecto implementa dos algoritmos desde cero: **Q-Learning** tabular y **Deep Q-Network (DQN)** con PyTorch, junto con una interfaz de línea de comandos completa para entrenar, evaluar y visualizar los agentes.

---

## 🗺️ El Ambiente

CliffWalking presenta un dilema clásico en RL entre exploración y explotación. El ambiente es **determinístico** — las acciones siempre producen el resultado esperado, lo que facilita el aprendizaje pero exige que el agente explore  suficientemente para descubrir que el borde del acantilado es peligroso antes de comprometerse con una política. Este ambiente es presentado como ejemplo ilustrativo en Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2020), Ejemplo 6.6, p. 132, precisamente para comparar entre el comportamiento dos algoritmos frente a este tipo de dilema.

CliffWalking es un grid de 4×12 donde un agente debe cruzar desde el punto de inicio hasta la meta evitando caer al acantilado.

```
 _  _  _  _  _  _  _  _  _  _  _  _
 _  _  _  _  _  _  _  _  _  _  _  _
 _  _  _  _  _  _  _  _  _  _  _  _
 S  C  C  C  C  C  C  C  C  C  C  G
```

- `S` — Inicio (estado 36)
- `G` — Meta (estado 47)
- `C` — Acantilado (estados 37-46)
- `_` — Camino seguro

---

## 📐 Espacio de Observaciones y Acciones

### Observaciones
El espacio de observación es 48, un entero que representa la posición actual del agente en el grid, calculado como `fila × 12 + columna`. El agente no puede estar en el acantilado (estados 37-46) ya que al caer regresa al inicio inmediatamente.

| Posición | Estado |
|---|---|
| Inicio [3,0] | 36 |
| Meta [3,11] | 47 |
| Acantilado [3,1..10] | 37-46 |

### Acciones
El espacio de acciones es 4:

| Acción | Dirección |
|---|---|
| 0 | Arriba |
| 1 | Derecha |
| 2 | Abajo |
| 3 | Izquierda |

### Recompensas
| Situación | Recompensa |
|---|---|
| Paso normal | -1 |
| Caer al acantilado | -100 + regresa al inicio |
| Llegar a la meta | -1 (termina el episodio) |

---

## 🔄 Flujo de Entrenamiento

### Q-Learning

```
1. Inicializar Q-table vacía (48 estados × 4 acciones)
2. Para cada episodio:
   a. Partir del estado 36
   b. Con probabilidad ε → acción aleatoria (exploración)
      Con probabilidad 1-ε → mejor acción según Q-table (explotación)
   c. Ejecutar acción → obtener recompensa y nuevo estado
   d. Actualizar Q-table:
      Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]
   e. Reducir ε gradualmente
3. Repetir hasta convergencia
```

### DQN

```
1. Inicializar red neuronal Q y red objetivo
2. Para cada episodio:
   a. Preprocesar observación → vector one-hot (48 dimensiones)
   b. Seleccionar acción con política ε-greedy
   c. Guardar (s, a, r, s', done) en ReplayBuffer
   d. Samplear mini-batch del buffer
   e. Calcular target: r + γ × max Q_target(s', a')
   f. Actualizar pesos minimizando MSE entre Q(s,a) y target
   g. Cada 10 episodios → sincronizar red objetivo
3. Reducir ε gradualmente
```

**Particularidades clave:**

Se uso una **codificación one-hot** para convertir el entero de estado en un vector que la red puede procesar. Las redes neuronales no interpretan números enteros como categorías, si le pasamos el estado 36 directamente, la red lo trataría como un valor numérico y asumiría que el estado 36 es "mayor" o "más cercano" al estado 35 que al estado 0, lo cual no tiene ningún sentido en un grid. Para evitar esto, convertimos cada estado en un vector binario de 48 dimensiones donde todas las posiciones son 0 excepto la que corresponde al estado actual, que vale 1. Así, el estado 36 se convierte en un vector donde solo la posición 36 está encendida. De esta forma, la red neuronal recibe una representación equidistante y sin jerarquía numérica entre los estados — cada estado es simplemente una dirección única en el espacio de entrada, y la red puede aprender libremente qué valor Q asignar a cada uno sin ser influenciada por la magnitud del número de estado.

---

## 🧠 Arquitectura de la Red Neuronal (DQN)

Se utiliza una red neuronal densa de dos capas ocultas:

```
Entrada (48)  →  Lineal(128)  →  ReLU  →  Lineal(128)  →  ReLU  →  Salida (4)
```

| Capa | Entrada | Salida | Activación |
|---|---|---|---|
| Entrada | 48 (one-hot) | 128 | ReLU |
| Oculta | 128 | 128 | ReLU |
| Salida | 128 | 4 (Q-values) | Ninguna |

**¿Por qué esta arquitectura?** CliffWalking es un ambiente discreto y pequeño. Una red lineal con 128 neuronas por capa es suficiente para aprender las relaciones entre posiciones y valores Q.

### Hiperparámetros

| Parámetro | Q-Learning | DQN |
|---|---|---|
| Learning rate | 0.1 | 1e-3 |
| Gamma (γ) | 0.99 | 0.99 |
| Epsilon inicio | 1.0 | 1.0 |
| Epsilon fin | 0.01 | 0.01 |
| Epsilon decay | 0.999 | 0.999 |
| Batch size | — | 64 |
| Buffer capacity | — | 100,000 |
| Target update | — | cada 10 episodios |

---

## 📊 Resultados del Entrenamiento

### Q-Learning (250,000 episodios)

El agente convergió a recompensas promedio entre **-17 y -28**, oscilando por el efecto del `epsilon_end=0.01` que mantiene una pequeña exploración residual.

```
Episode 205900/250000 | Avg Reward: -18.05 | Epsilon: 0.0500 | States visited: 37
Episode 209900/250000 | Avg Reward: -20.03 | Epsilon: 0.0500 | States visited: 37
Episode 222100/250000 | Avg Reward: -17.93 | Epsilon: 0.0500 | States visited: 37
```

El agente visitó **37 de 48 estados** — nunca exploró el acantilado completo ya que aprendió a evitarlo eficientemente. 

### DQN (10,000 episodios)

La red neuronal convergió más lentamente que Q-Learning pero logró estabilizarse en recompensas similares.

```
Episode 9980/10000 | Avg Reward: -23.40 | Epsilon: 0.0100 | Buffer: 100000
Episode 9990/10000 | Avg Reward: -13.00 | Epsilon: 0.0100 | Buffer: 100000
Episode 10000/10000 | Avg Reward: -24.00 | Epsilon: 0.0100 | Buffer: 100000
```

---

## 🔍 Reflexión de los Resultados

**Q-Learning** demostró ser claramente superior para este ambiente por varias razones. CliffWalking es determinístico, pequeño (48 estados) y tiene una estructura simple que una tabla puede representar perfectamente. La tabla Q converge con precisión exacta a los valores óptimos sin necesidad de aproximación.

**DQN**, aunque funciona correctamente, es una herramienta sobredimensionada para este problema. La red neuronal aproxima los valores Q con cierto error inherente, y requiere muchos más episodios para estabilizarse debido al proceso de entrenamiento por gradiente y la dependencia del ReplayBuffer.

Durante el entrenamiento ambos algoritmos muestran iteraciones que oscilan alrededor del valor óptimo. Según Sutton & Barto en *Reinforcement Learning: An Introduction* (2020), la política óptima para CliffWalking produce una recompensa de **-13**, correspondiente a la ruta más corta posible hacia la meta. Al renderizar ambos agentes entrenados, los dos logran completar el episodio alcanzando exactamente ese óptimo de **-13**, lo que confirma que ambos convergieron a la política correcta.

Sin embargo, utilizar **DQN no aporta beneficios adicionales** frente a Q-Learning en este ambiente. Ambos llegan al mismo resultado óptimo, pero Q-Learning lo hace de forma más directa, con menor costo computacional y sin la complejidad adicional de gestionar una red neuronal, un buffer de experiencias y una red objetivo. Para ambientes discretos y pequeños como CliffWalking, la tabla Q es la herramienta adecuada — DQN justifica su complejidad únicamente cuando el espacio de estados es demasiado grande o continuo para ser representado de forma tabular.

---

## 💭 Lo que más costó en el proyecto

Uno de los mayores desafíos conceptuales fue entender **por qué el estado no se puede pasar directamente como entero a la red neuronal**. Al principio parecía natural simplemente entregarle el número `36` a la red y dejar que aprendiera, pero esto implica un problema fundamental: la red neuronal interpreta los números como magnitudes continuas, lo que significa que asumiría relaciones de orden y proximidad entre los estados que no existen en la realidad. Deducir que la solución era la **codificación one-hot** — transformar cada estado en un vector de 48 dimensiones donde solo una posición vale `1` — requirió entender que la red no recibe escalares sino vectores, y que cada dimensión del vector es una señal independiente que la red puede ponderar libremente sin asumir ninguna jerarquía entre los estados.

El segundo gran desafío fue comprender **cómo aprende internamente el DQN**, función por función. No fue suficiente saber que "usa una red neuronal" — fue necesario entender qué hace `_encode_state` antes de `select_action`, por qué `_learn` no se llama con el estado actual sino con un batch del `ReplayBuffer`, qué rol cumple la `target_net` separada de la `q_net`, y por qué sincronizarlas cada cierto número de episodios en lugar de en cada paso. Cada una de estas decisiones de diseño tiene una razón matemática concreta: el buffer rompe la correlación temporal entre experiencias consecutivas, y la red objetivo evita que el agente persiga un blanco que se mueve en cada actualización de pesos. Entender esto no como código sino como algoritmo fue lo que más tiempo y revisión demandó del proyecto.

---

## 📁 Estructura del Proyecto

```
cliff/
├── src/
│   └── rl_games/
│       ├── cli.py # Interfaz de línea de comandos
│       └── agents/
│           ├── qlearning.py   # Agente Q-Learning tabular
│           └── dqn.py         # Agente DQN con PyTorch
│                   
├── saves/                     # Modelos guardados (generado al entrenar)
├── pyproject.toml
└── README.md
```

---

## 📦 Dependencias

| Paquete | Uso |
|---|---|
| `gymnasium` | Ambiente CliffWalking-v1 |
| `torch` | Red neuronal para DQN |
| `numpy` | Operaciones numéricas |

---

## ⚙️ Instalación

Requiere Python 3.11.

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/rl-cliffwalking.git
cd rl-cliffwalking

# Instalar dependencias con uv
uv sync

# Activar el entorno virtual
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / Mac
```

---

## 🕹️ Uso

### Ver información del ambiente

```bash
rlgames inspect --steps 10
```

Muestra el espacio de observaciones/acciones y ejecuta pasos con política aleatoria.

### Inicializar un agente

```bash
rlgames init qlearning
rlgames init dqn
```

### Entrenar

```bash
rlgames train qlearning --episodes 10000
rlgames train dqn --episodes 5000
```

### Ver información del agente guardado

```bash
rlgames load qlearning
rlgames load dqn --eval     # incluye evaluación de 10 episodios
```

### Simular episodios

```bash
rlgames sim qlearning --episodes 5
rlgames sim dqn --episodes 3 --steps 20
```

### Renderizar visualmente

```bash
rlgames render qlearning --episodes 3
rlgames render dqn --episodes 3
```

Abre una ventana gráfica con el agente moviéndose por el mapa.

### Listar agentes disponibles

```bash
rlgames list
```

### Ver versión

```bash
rlgames version
```

---

## 📝 Notas

- Los modelos entrenados se guardan en `saves/qlearning_cliff.pkl` y `saves/dqn_cliff.pt`.
- Q-Learning converge significativamente más rápido que DQN en este ambiente.
- El ambiente usa `is_slippery=False` por defecto — completamente determinístico.

---

## 📚 Referencias
- Sutton, R. S., & Barto, A. G. (2020). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. http://www.incompleteideas.net/book/RLbook2020.pdf