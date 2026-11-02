/* =================================================================
 * tfjs_models.js
 * Carga de los cuatro modelos de caracoles (TF.js GraphModel) y
 * preprocesamiento + predicción. Todo se ejecuta en el cliente.
 *
 * Modelos (exportados con tensorflowjs_converter, cuantizados a
 * float16):
 *   modelos/<arquitectura>_<aug>_aug/model.json + group1-shard*.bin
 *
 * Detalles importantes:
 *  - La entrada espera 224x224x3 RGB con valores CRUDOS (0-255):
 *    el preprocess_input de cada arquitectura va dentro del grafo.
 *  - La salida son LOGITS (Dense lineal): se aplica softmax aquí.
 *  - Orden de clases: alfabético (sorted(os.listdir) en el entrenamiento).
 * ================================================================= */

const GestorModelos = (() => {
    'use strict';

    /* ---------------------------------------------------------------
     * Configuración
     * --------------------------------------------------------------- */
    const RUTAS = {
        vgg16_sin_aug:    'modelos/vgg16_sin_aug/model.json',
        vgg16_con_aug:    'modelos/vgg16_con_aug/model.json',
        resnet50_sin_aug: 'modelos/resnet50_sin_aug/model.json',
        resnet50_con_aug: 'modelos/resnet50_con_aug/model.json'
    };

    const CLASES = [
        'Cerithiidae',
        'Embrionic_shell',
        'Eulimidae',
        'Litiopidae',
        'Neritidae',
        'Risooinidae'
    ];

    const TAM = 224;   // tamaño de entrada de los modelos

    /* ---------------------------------------------------------------
     * Caché de modelos (y de promesas en vuelo para no cargar dos veces)
     * --------------------------------------------------------------- */
    const cache = {};
    const pendientes = {};

    function claveModelo(arquitectura, aumento) {
        return `${arquitectura}_${aumento}_aug`;
    }

    /**
     * Carga un modelo bajo demanda y lo guarda en caché.
     * @param {string} clave  p. ej. 'vgg16_con_aug'
     * @returns {Promise<tf.GraphModel|tf.LayersModel>}
     */
    async function cargarModelo(clave, onProgreso) {
        if (cache[clave]) return cache[clave];
        if (pendientes[clave]) return pendientes[clave];

        const ruta = RUTAS[clave];
        if (!ruta) throw new Error(`Clave de modelo desconocida: ${clave}`);

        pendientes[clave] = (async () => {
            try {
                const opciones = onProgreso ? { onProgress: onProgreso } : undefined;
                const modelo = await tf.loadGraphModel(ruta, opciones);
                cache[clave] = modelo;
                console.info(`[GestorModelos] Modelo (graph) cargado: ${clave}`);
                return modelo;
            } catch (errGraph) {
                // Fallback por si algún modelo se hubiera exportado como layers
                try {
                    const modelo = await tf.loadLayersModel(ruta);
                    cache[clave] = modelo;
                    console.info(`[GestorModelos] Modelo (layers) cargado: ${clave}`);
                    return modelo;
                } catch (errLayers) {
                    throw new Error(
                        `No se pudo cargar el modelo "${clave}". ` +
                        `Verifica que exista "${ruta}" y sus .bin. ` +
                        `Detalle: ${errGraph.message || errGraph}`
                    );
                }
            } finally {
                delete pendientes[clave];
            }
        })();
        return pendientes[clave];
    }

    function estaCargado(clave) {
        return !!cache[clave];
    }

    /**
     * Libera los pesos de todos los modelos en caché salvo `claveMantener`.
     * Pensado para móvil: 4 modelos en float32 pueden superar 600 MB de RAM.
     */
    function liberarOtros(claveMantener) {
        for (const clave of Object.keys(cache)) {
            if (clave === claveMantener) continue;
            try { cache[clave].dispose(); } catch (e) { /* noop */ }
            delete cache[clave];
            console.info(`[GestorModelos] Modelo liberado: ${clave}`);
        }
    }

    /* ---------------------------------------------------------------
     * Preprocesamiento — cámara
     * --------------------------------------------------------------- */

    /**
     * Convierte el fotograma actual del <video> en un tensor de entrada.
     * El modelo lleva el preprocess_input dentro del grafo, así que solo
     * hace falta RGB a 224x224 con valores 0-255 en float32.
     *
     * @param {HTMLVideoElement} video
     * @returns {tf.Tensor} tensor [1, 224, 224, 3]
     */
    function preprocesar(video) {
        return tf.tidy(() => {
            const rgb   = tf.browser.fromPixels(video, 3);            // [H,W,3]
            const redim = tf.image.resizeBilinear(rgb, [TAM, TAM]);   // float32
            return redim.expandDims(0).toFloat();                     // [1,224,224,3]
        });
    }

    /* ---------------------------------------------------------------
     * Predicción
     * --------------------------------------------------------------- */

    /**
     * Clasifica el fotograma actual con el modelo seleccionado.
     *
     * @param {HTMLVideoElement} video
     * @param {'vgg16'|'resnet50'} arquitectura
     * @param {'sin'|'con'} aumento
     * @returns {Promise<{probs:number[], indice:number, etiqueta:string, confianza:number}>}
     */
    async function predecir(video, arquitectura, aumento) {
        const clave = claveModelo(arquitectura, aumento);
        const modelo = await cargarModelo(clave);

        const entrada = preprocesar(video);
        let salida;
        try {
            salida = await modelo.executeAsync(entrada);
            if (Array.isArray(salida)) salida = salida[0];
            else if (!(salida instanceof tf.Tensor)) salida = salida[Object.keys(salida)[0]];
        } finally {
            entrada.dispose();
        }

        // Salida = logits [1,6] -> softmax -> probabilidades
        const probs = tf.tidy(() => tf.softmax(salida.reshape([-1])).dataSync());
        salida.dispose();

        const arr = Array.from(probs);
        let indice = 0;
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > arr[indice]) indice = i;
        }
        return {
            probs: arr,
            indice,
            etiqueta: CLASES[indice],
            confianza: arr[indice]
        };
    }

    /* ---------------------------------------------------------------
     * API pública
     * --------------------------------------------------------------- */
    return {
        cargarModelo,
        estaCargado,
        liberarOtros,
        claveModelo,
        predecir,
        CLASES,
        config: { TAM, RUTAS }
    };
})();
