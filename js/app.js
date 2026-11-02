/* =================================================================
 * app.js
 * Lógica de la interfaz:
 *   - Selectores de modelo (arquitectura × augmentation)
 *   - Activación de cámara (WebRTC) y cambio frontal/trasera
 *   - Bucle de predicción en tiempo real
 *   - Barras de probabilidad por familia (generadas dinámicamente)
 *
 * Depende de GestorModelos (tfjs_models.js) y de TensorFlow.js (tf).
 * ================================================================= */

(() => {
    'use strict';

    /* ---------------------------------------------------------------
     * Estado de la aplicación
     * --------------------------------------------------------------- */
    const estado = {
        arquitectura: 'vgg16',        // 'vgg16' | 'resnet50'
        aumento: 'con',               // 'sin' | 'con'
        camaraActiva: false,
        stream: null,                 // MediaStream actual
        facingMode: 'environment',    // 'user' | 'environment' (móvil)
        dispositivos: [],             // cámaras de vídeo enumeradas
        indiceDispositivo: 0,         // cámara activa dentro de 'dispositivos'
        bucleId: null,                // id del requestAnimationFrame
        prediciendo: false            // evita inferencias solapadas
    };

    /* ---------------------------------------------------------------
     * Atajos del DOM
     * --------------------------------------------------------------- */
    const $  = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));

    const video       = $('#video');
    const placeholder = $('#camera-placeholder');
    const btnCamara   = $('#btn-camara');
    const btnCam      = $('#btn-cambiar-camara');
    const contBarras  = $('#barras');
    const statusEl    = $('#status');

    /* ---------------------------------------------------------------
     * Utilidades
     * --------------------------------------------------------------- */
    function setStatus(texto, tipo = '') {
        statusEl.textContent = texto;
        statusEl.className = 'status-line' + (tipo ? ` status-line--${tipo}` : '');
    }

    function porcentaje(v) {
        return `${Math.round(v * 100)}%`;
    }

    function nombreModelo() {
        return `${estado.arquitectura.toUpperCase()} ${estado.aumento === 'con' ? 'CON' : 'SIN'} aug`;
    }

    /* ---------------------------------------------------------------
     * Barras de probabilidad (una por familia)
     * --------------------------------------------------------------- */
    const filas = [];   // {fill, value, row}

    function crearBarras() {
        contBarras.innerHTML = '';
        filas.length = 0;
        GestorModelos.CLASES.forEach(nombre => {
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <span class="bar-row__label" title="${nombre}">${nombre}</span>
                <div class="bar"><div class="bar__fill" style="width:0%"></div></div>
                <span class="bar-row__value">0%</span>`;
            contBarras.appendChild(row);
            filas.push({
                row,
                fill:  row.querySelector('.bar__fill'),
                value: row.querySelector('.bar-row__value')
            });
        });
    }

    function actualizarBarras(probs, indiceTop = -1) {
        filas.forEach((f, i) => {
            f.fill.style.width = porcentaje(probs[i]);
            f.value.textContent = porcentaje(probs[i]);
            f.row.classList.toggle('bar-row--top', i === indiceTop);
        });
    }

    function resetBarras() {
        actualizarBarras(GestorModelos.CLASES.map(() => 0));
    }

    /* ===============================================================
     * 1. Selectores de modelo (arquitectura × augmentation)
     * =============================================================== */
    $$('[data-toggle]').forEach(grupo => {
        const tipo = grupo.dataset.toggle;   // 'arquitectura' | 'aumento'
        grupo.querySelectorAll('.toggle').forEach(btn => {
            btn.addEventListener('click', () => {
                if (btn.classList.contains('toggle--active')) return;
                grupo.querySelectorAll('.toggle').forEach(b =>
                    b.classList.toggle('toggle--active', b === btn));
                if (tipo === 'arquitectura') estado.arquitectura = btn.dataset.valor;
                else estado.aumento = btn.dataset.valor;

                // Precarga el modelo seleccionado (los pesos son grandes)
                cargarModeloActual();
            });
        });
    });

    async function cargarModeloActual() {
        const clave = GestorModelos.claveModelo(estado.arquitectura, estado.aumento);
        if (GestorModelos.estaCargado(clave)) {
            setStatus(`Modelo listo: ${nombreModelo()}.`, 'ok');
            return;
        }
        setStatus(`Cargando ${nombreModelo()}… (puede tardar unos segundos)`);
        try {
            await GestorModelos.cargarModelo(clave);
            setStatus(`Modelo listo: ${nombreModelo()}.`, 'ok');
        } catch (err) {
            console.error(err);
            setStatus('Modelo no disponible. Coloca los archivos en /modelos.', 'error');
        }
    }

    /* ===============================================================
     * 2. Cámara (WebRTC)
     * ---------------------------------------------------------------
     * Funciona tanto en móvil (frontal/trasera vía facingMode) como
     * en escritorio (webcams USB/integradas vía enumeración de
     * dispositivos con deviceId). El botón "Cambiar cámara" recorre
     * las cámaras físicas realmente disponibles.
     * =============================================================== */
    btnCamara.addEventListener('click', () => {
        estado.camaraActiva ? detenerCamara() : iniciarCamara();
    });

    btnCam.addEventListener('click', cambiarCamara);

    /**
     * Enumera las cámaras de vídeo disponibles. Los nombres
     * (labels) solo aparecen tras conceder permiso de cámara.
     */
    async function obtenerDispositivos() {
        try {
            const todos = await navigator.mediaDevices.enumerateDevices();
            estado.dispositivos = todos.filter(d => d.kind === 'videoinput');
        } catch (e) {
            estado.dispositivos = [];
        }
        return estado.dispositivos;
    }

    /**
     * Detecta qué cámara está activa en el stream y la sincroniza
     * con estado.indiceDispositivo.
     */
    function sincronizarDispositivoActivo() {
        if (!estado.stream) return;
        const pista = estado.stream.getVideoTracks()[0];
        const idActivo = (pista && pista.getSettings().deviceId) || null;
        const idx = estado.dispositivos.findIndex(d => d.deviceId === idActivo);
        estado.indiceDispositivo = idx >= 0 ? idx : 0;
    }

    /**
     * Activa la cámara.
     *  - Con deviceId: abre esa cámara concreta (cambio manual).
     *  - Sin deviceId: abre la preferida (trasera en móvil, webcam
     *    por defecto en escritorio, ya que 'ideal' degrada con suavidad).
     */
    async function iniciarCamara(deviceId) {
        try {
            detenerStream();

            const videoConstraints = { width: { ideal: 640 }, height: { ideal: 480 } };
            if (deviceId) {
                videoConstraints.deviceId = { exact: deviceId };
            } else if (estado.facingMode) {
                videoConstraints.facingMode = { ideal: estado.facingMode };
            }
            const constraints = { video: videoConstraints, audio: false };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            estado.stream = stream;
            video.srcObject = stream;
            await video.play();

            // Tras conceder permiso, las etiquetas ya están disponibles.
            await obtenerDispositivos();
            sincronizarDispositivoActivo();

            estado.camaraActiva = true;
            placeholder.style.display = 'none';
            btnCamara.innerHTML = '<span class="btn__icon">■</span> Detener cámara';
            actualizarBotonCambiarCamara();
            iniciarBucleCamara();
        } catch (err) {
            console.error(err);
            setStatus(
                'No se pudo acceder a la cámara. Sirve la página por HTTPS o ' +
                'localhost (no abras el archivo con doble clic) y concede permiso.',
                'error');
        }
    }

    /**
     * Recorre las cámaras físicas disponibles (móvil y escritorio).
     */
    async function cambiarCamara() {
        if (!estado.camaraActiva) return;
        await obtenerDispositivos();
        if (estado.dispositivos.length < 2) {
            setStatus('Solo hay una cámara disponible.', 'error');
            return;
        }
        estado.indiceDispositivo =
            (estado.indiceDispositivo + 1) % estado.dispositivos.length;
        await iniciarCamara(estado.dispositivos[estado.indiceDispositivo].deviceId);
    }

    /**
     * Actualiza el botón "Cambiar cámara" según el nº de cámaras.
     */
    function actualizarBotonCambiarCamara() {
        const n = estado.dispositivos.length;
        if (n > 1) {
            btnCam.disabled = false;
            btnCam.textContent = `↻ Cambiar cámara (${estado.indiceDispositivo + 1}/${n})`;
        } else {
            btnCam.disabled = true;
            btnCam.textContent = '↻ Cambiar cámara';
        }
    }

    function detenerCamara() {
        detenerStream();
        estado.camaraActiva = false;
        video.srcObject = null;
        placeholder.style.display = 'flex';
        btnCamara.innerHTML = '<span class="btn__icon">▶</span> Activar cámara';
        btnCam.disabled = true;
        btnCam.textContent = '↻ Cambiar cámara';
        detenerBucleCamara();
        resetBarras();
        setStatus('Cámara detenida.');
    }

    function detenerStream() {
        if (estado.stream) {
            estado.stream.getTracks().forEach(t => t.stop());
            estado.stream = null;
        }
    }

    /* ---------------------------------------------------------------
     * 3. Bucle de predicción en tiempo real
     * --------------------------------------------------------------- */
    function iniciarBucleCamara() {
        detenerBucleCamara();
        let ultimo = 0;
        const INTERVALO = 250; // ms entre inferencias (rendimiento)

        const tick = async (t) => {
            if (!estado.camaraActiva) return;
            if (t - ultimo >= INTERVALO && video.readyState >= 2 && !estado.prediciendo) {
                ultimo = t;
                await predecirFotograma();
            }
            estado.bucleId = requestAnimationFrame(tick);
        };
        estado.bucleId = requestAnimationFrame(tick);
    }

    function detenerBucleCamara() {
        if (estado.bucleId) {
            cancelAnimationFrame(estado.bucleId);
            estado.bucleId = null;
        }
    }

    async function predecirFotograma() {
        estado.prediciendo = true;
        try {
            const r = await GestorModelos.predecir(video, estado.arquitectura, estado.aumento);
            actualizarBarras(r.probs, r.indice);
            setStatus(
                `Predicción (${nombreModelo()}): ${r.etiqueta} · ${porcentaje(r.confianza)}`,
                'ok');
        } catch (err) {
            console.error(err);
            setStatus(
                'Modelo no disponible. Coloca los archivos en /modelos.',
                'error');
            detenerBucleCamara();
        } finally {
            estado.prediciendo = false;
        }
    }

    /* ===============================================================
     * 4. Inicialización
     * =============================================================== */
    function init() {
        crearBarras();

        // Soporte de cámara
        if (!navigator.mediaDevices?.getUserMedia) {
            setStatus(
                'Tu navegador no soporta acceso a cámara (getUserMedia).',
                'error');
            btnCamara.disabled = true;
        }

        // TensorFlow listo
        if (typeof tf !== 'undefined') {
            console.info(`[app] TensorFlow.js backend: ${tf.getBackend()}`);
            cargarModeloActual();
        } else {
            setStatus('TensorFlow.js no se cargó. Revisa la conexión.', 'error');
        }
    }

    // Limpieza al cerrar
    window.addEventListener('beforeunload', () => {
        detenerBucleCamara();
        detenerStream();
    });

    document.addEventListener('DOMContentLoaded', init);
})();
