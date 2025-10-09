# 🧠 NPI Supplier Selection Fuzzy Logic Simulation

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rubusarbaro/supplier-selection_fuzzy-logic_thesis)

---

## 📘 Descripción general

Este proyecto implementa un **modelo de lógica difusa** para la **evaluación y selección de proveedores** dentro del proceso de **New Product Introduction (NPI)** en la industria **HVAC**.  
El sistema simula las etapas del flujo de compras técnicas desde **Design Freeze (P2)** hasta **Start of Production (P4)**, generando datos sintéticos sobre precios, plazos de entrega, puntualidad y aprobación documental, con el fin de alimentar un modelo difuso tipo **Mamdani**.

El proyecto forma parte de la **tesis de Maestría en Logística y Cadena de Suministro** en la **Universidad Autónoma de Nuevo León (FIME)**.

---

## 🧠 Contexto académico

En proyectos NPI, la selección de proveedores implica **incertidumbre y ambigüedad lingüística** (por ejemplo, “entrega rápida”, “precio competitivo”, “buena puntualidad”).  
El modelo desarrollado aborda esta incertidumbre mediante **conjuntos difusos** y reglas del tipo **“si–entonces”**, lo cual permite una evaluación más realista de los proveedores considerando:

- ⏱️ **Tiempo de entrega (Delivery Time)**  
- 💸 **Costo o gasto anual (FY Spend)**  
- 📈 **Puntualidad (On-Time Delivery)**  
- 📅 **Tiempo hasta SOP (Due Time)**  

La salida del modelo determina si un proveedor debe **“Implementarse”** (asignar negocio) o **“Esperar”**, en función de las condiciones del proyecto.

---

## ⚙️ Estructura del repositorio

```
📂 supplier-selection_fuzzy-logic_thesis/
├── simulation.py              # Clases base de simulación y modelo difuso
├── implementation.ipynb       # Notebook de ejecución, análisis y visualización
├── .env.example               # Variables de entorno ejemplo (.env requerido)
├── LICENSE                    # MIT License
└── README.md                  # Este archivo
```

### Principales clases (`simulation.py`)

| Clase | Descripción |
|:------|:-------------|
| **Project** | Representa un proyecto NPI con sus fechas clave (DF, MCS, Pilot, SOP). |
| **Part_Number** | Define un material o número de parte con complejidad y consumo anual. |
| **ECN** | Representa un *Engineering Change Notification* (conjunto de partes). |
| **Supplier** | Simula un proveedor con perfiles de entrega, precio, cotización y puntualidad. |
| **Environment** | Entorno general donde interactúan los objetos (proveedores, ECNs, proyectos). |
| **Fuzzy_Model** | Implementa el modelo difuso de tipo Mamdani para evaluación de proveedores. |

---

## 🧮 Requisitos y dependencias

Este proyecto utiliza las siguientes librerías de Python:

```bash
numpy
pandas
matplotlib
scikit-fuzzy
python-dotenv
```

### Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/supplier-selection_fuzzy-logic_thesis.git
   cd supplier-selection_fuzzy-logic_thesis
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
   *(Si no existe el archivo, instálalas manualmente con pip)*

3. Crea un archivo `.env` a partir de `.env.example` con los valores estadísticos requeridos.

---

## 🚀 Ejecución

### 🧩 Opción 1 — Desde Jupyter Notebook
Abre `implementation.ipynb` y ejecuta las celdas en orden.  
El notebook genera:
- Simulación del proceso NPI (ECNs, proveedores y cotizaciones).  
- Evaluación difusa para un proveedor específico.  
- Gráficas de funciones de membresía y reglas aplicadas.

### 🖥️ Opción 2 — Desde consola
Puedes importar y ejecutar el módulo directamente:

```python
from simulation import Environment, Project, Fuzzy_Model
from datetime import date

env = Environment()
project = Project("RTU_Copper_Pipes", date(2025,3,1), date(2025,4,15), date(2025,5,20), date(2025,7,1))

# Crear proveedores y ECNs
env.create_supplier("Supplier A", delivery_profile="high")
env.create_supplier("Supplier B", price_profile="low")
env.gen_ecns(project, qty=3)

# Cotizar e implementar
env.quote_all_ecn_project_all_suppliers(project)
env.implement_ecn(env.ecns[0], env.get_supplier("name", "Supplier A"))

# Evaluar con modelo difuso
model = Fuzzy_Model(env.item_master, env.get_supplier("name", "Supplier A"), env.ecns[0])
print(model.get_stats())
```

---

## 🔍 Ejemplo de salida

```python
{
  'Supplier ID': '10000001',
  'New supplier': False,
  'Score': 7.42,
  'Wait': 0.15,
  'Implement': 0.87,
  'Action': 'Implement'
}
```

El resultado indica que el proveedor debe ser **implementado** según las condiciones simuladas.

---

## 📊 Resultados esperados

El modelo genera:

- **Dataset simulado (`Item_Master`)** con datos sintéticos del proceso NPI.  
- **Gráficas de funciones de membresía** (tiempo de entrega, gasto anual, puntualidad, etc.).  
- **Salida difusa defuzzificada** en forma de puntuación (`Score`) y recomendación (“Implement” o “Wait”).

---

## 🧩 Aplicaciones y adaptaciones

El modelo es totalmente parametrizable:
- Puede usarse para distintos materiales, divisiones o industrias.  
- Se pueden modificar las funciones de membresía y reglas en la clase `Fuzzy_Model`.  
- Permite integrar criterios adicionales (p. ej. sostenibilidad, riesgo, ESG).  

---

## 🧾 Citación académica

Si este repositorio se usa en publicaciones o trabajos académicos, por favor cite como:

> **Morales Velázquez, S. R. (2025).**  
> *NPI Supplier Selection Fuzzy Logic Simulation.*  
> [GitHub Repository](https://github.com/rubusarbaro/supplier-selection_fuzzy-logic_thesis)

### Formato BibTeX

```bibtex
@misc{Morales2025FuzzyNPI,
  author       = {Saúl R. Morales Velázquez},
  title        = {NPI Supplier Selection Fuzzy Logic Simulation},
  year         = {2025},
  howpublished = {\url{https://github.com/rubusarbaro/supplier-selection_fuzzy-logic_thesis}},
  note         = {Universidad Autónoma de Nuevo León, FIME}
}
```

---

## ⚖️ Licencia

Este proyecto está licenciado bajo la **MIT License**.  
Consulta el archivo `LICENSE` para más detalles.