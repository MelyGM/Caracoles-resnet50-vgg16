# Caracoles · Clasificador de familias con Transfer Learning

Aplicación web que ejecuta **4 modelos de clasificación de caracoles en el
navegador** mediante TensorFlow.js, sin backend ni envío de datos a servidores.

## Funcionalidades

- **Cámara en tiempo real**: apunta a un caracol y el modelo clasifica cada
  fotograma (~250 ms entre inferencias).
- **Selector doble de modelo**:
  - Arquitectura: **VGG16** / **ResNet50**
  - Aumento de datos: **SIN** / **CON** augmentation
- **Barras de probabilidad** para las 6 familias: *Cerithiidae*,
  *Embrionic_shell*, *Eulimidae*, *Litiopidae*, *Neritidae*, *Risooinidae*
  (orden alfabético, igual que en el entrenamiento).

## Estructura

```
.
├── index.html            # Vista única: cámara + resultados
├── css/style.css         # Estilos (misma paleta que la app de perros/gatos)
├── js/
│   ├── app.js            # UI: selectores, cámara, bucle de predicción
│   └── tfjs_models.js    # GestorModelos: carga, preproceso, softmax
├── modelos/
│   ├── vgg16_sin_aug/    # model.json + group1-shard*.bin
│   ├── vgg16_con_aug/
│   ├── resnet50_sin_aug/
│   └── resnet50_con_aug/
└── README.md
```

## Cómo ejecutarlo

La página es estática, pero **debe servirse por HTTP(S)** o `localhost`
(la cámara lo exige y los `.bin` no cargan desde `file://`).

### Opción A — servidor HTTP de Python (local)

```bash
cd caracoles-web
python -m http.server 8000
```

Abre `http://localhost:8000` en el navegador.

### Opción B — VS Code Live Server

Clic derecho en `index.html` → *Open with Live Server*.

## Detalles técnicos

- Los modelos se entrenaron en Python (Keras, transfer learning sobre
  ImageNet, fine-tuning) y se exportaron a SavedModel → `tensorflowjs_converter`
  (`--output_format=tfjs_graph_model --quantize_float16=*`).
- **Entrada**: 224×224×3 RGB con valores crudos (0-255). El `preprocess_input`
  de cada arquitectura (BGR + resta de medias) va **dentro** del grafo, así
  que en JS no se normaliza.
- **Salida**: 6 logits → `tf.softmax` en JS para las probabilidades.
- Los modelos CON augmentation se exportaron sin las capas `Random*` (que en
  inferencia son passthrough) para evitar el error del seed generator en el
  converter; predicciones verificadas idénticas al original (`max|diff| = 0`).
- Modelos en **float16 cuantizado**: VGG16 ≈ 60 MB, ResNet50 ≈ 110 MB.
  La primera carga de cada modelo tarda unos segundos.

## Nota sobre el dominio

El dataset original son imágenes **SEM** (microscopio electrónico). Con fotos
de webcam normales el acierto puede ser inferior al ~99 % de validación.
