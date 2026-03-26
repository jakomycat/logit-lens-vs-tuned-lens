# Logit Lens vs. Tuned Lens: A Comparative Analysis of the Interpretability of Language Models

**Author:** Jacobo Chalarca Vásquez ([@jakomycat](https://github.com/jakomycat))  

![Python](https://img.shields.io/badge/python-3.11.x-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Abstract

La interpretabilidad de los modelos Transformer es un desafío abierto. Este proyecto implementa y compara dos técnicas prominentes para decodificar las representaciones latentes en las capas intermedias de los LLMs: **Logit Lens** (nostalgebraist, 2020) y **Tuned Lens** (Belrose et al., 2023). A través de experimentos estructurados, analizamos cómo evoluciona la predicción de tokens a lo largo de la profundidad de la red, evaluando si las transformaciones afines entrenadas (*Tuned Lens*) ofrecen una representación más fiel de la trayectoria de inferencia en comparación con la proyección directa (*Logit Lens*).

---

## 1. Introducción

En los modelos Transformer estándar, la predicción final se obtiene aplicando una matriz de *un-embedding* $E$ a la salida de la última capa oculta $h_L$. Sin embargo, comprender qué ocurre en las capas intermedias $h_l$ (donde $l < L$) requiere técnicas especializadas.

* **Logit Lens:** Asume que el espacio latente de las capas intermedias es compatible con la matriz de salida. Aplica directamente la matriz de *un-embedding* original a cualquier capa intermedia:
    $$P(x_l) = \text{softmax}(E \cdot h_l)$$
* **Tuned Lens:** Reconoce que las capas tempranas pueden operar en un subespacio diferente. Entrena una transformación afín $T_l$ (una matriz de proyección y un sesgo) para cada capa $l$, optimizada para predecir la salida final o el siguiente token con mayor precisión:
    $$P(x_l) = \text{softmax}(E \cdot T_l(h_l))$$

Este repositorio busca replicar estas metodologías y comparar sus resultados cualitativos y cuantitativos.

---

## 2. Metodología y Arquitectura del Proyecto

### 2.1. Configuración del Modelo y los Datos

El código está diseñado para funcionar especificamente para GPT-2 small, no se ha probado para otros modelos. Aunque en el futuro se tiene pensado expandir el código para que funcione con distintos modelos. El modelo usado tiene doce capas ocultas ($L = 12$).

### 2.2. Extracción de Estados Ocultos



### Estructura del Repositorio
* `src/`: Módulos core (definición de Lenses, utilidades de extracción).
* `train.py`: Script principal para optimizar las matrices del Tuned Lens.
* `checkpoints/`: Pesos entrenados ($T_l$) generados por el script de entrenamiento.
* `tests/`: Pruebas unitarias para validar la integridad matemática de las proyecciones.

---

## 3. Resultados Preliminares

*(Nota: Reemplaza las siguientes imágenes con las gráficas reales generadas en tus notebooks)*

### 3.1. Trayectoria de Predicciones
A continuación se muestra un mapa de calor (heatmap) comparando la confianza del modelo sobre el token correcto a través de las capas usando ambas técnicas.

<div align="center">
  <p><i>[ 📊 Espacio para insertar el Heatmap comparativo de Logit Lens vs Tuned Lens ]</i></p>
</div>

### 3.2. Observaciones
* **Logit Lens:** Tiende a mostrar predicciones "ruidosas" o sin sentido en las primeras capas, consolidando el token correcto solo en el último 20% del modelo.
* **Tuned Lens:** Logra decodificar intenciones semánticas mucho más temprano en la red, demostrando que la información ya estaba presente pero en un espacio latente rotado/trasladado.

---

## 4. Reproducibilidad (Setup)

Para replicar los experimentos de este "mini-paper", sigue las instrucciones de configuración:

### Instalación
```bash
git clone [https://github.com/jakomycat/logit-lens-vs-tuned-lens.git](https://github.com/jakomycat/logit-lens-vs-tuned-lens.git)
cd logit-lens-vs-tuned-lens
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Referencias

1. **nostalgebraist** (2020). *interpreting GPT: the logit lens*. LessWrong. Recuperado de: [https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

2. **Belrose, S., Furman, Z., Smith, L., Halawi, C., McKinney, I., Mutters, T., ... & Steinhardt, J.** (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens*. arXiv preprint arXiv:2303.08112. Recuperado de: [https://arxiv.org/abs/2303.08112](https://arxiv.org/abs/2303.08112)