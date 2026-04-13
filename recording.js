// =============================================================================
// recording.js — Frame recording + video export for Petri Dish NCA
//
// Captures the live WebGL canvas via canvas.captureStream() + MediaRecorder
// and saves a .webm file. An optional "Convert to MP4" button lazy-loads
// ffmpeg.wasm (~30MB) from a CDN to transcode after the fact.
//
// Exposes a single global function: setupRecording(canvas, gui)
// =============================================================================
(function () {
    'use strict';

    const RECORDING_STATE = {
        FPS: 30,
        BITRATE_MBPS: 8,
        BLEND_ALPHA: 0.0,       // 0 = no blending (raw frames), 0.9 = heavy trail/smear
        recorder: null,
        chunks: [],
        stream: null,
        startTime: 0,
        bytesSoFar: 0,
        statusEl: null,
        statusTimer: null,
        lastBlob: null,
        lastBlobName: null,
        recordController: null,
        // EMA frame-blending state (lazily initialised on first recording with blend > 0)
        blendCanvas: null,      // offscreen 2D canvas for blended output
        blendCtx: null,
        blendAcc: null,         // Float32Array accumulator [W*H*4]
        blendTempU8: null,      // Uint8Array for readPixels
        blendAnimId: null,      // requestAnimationFrame handle for the blend tick
    };

    function ensureStatusElement() {
        if (RECORDING_STATE.statusEl) return RECORDING_STATE.statusEl;
        const el = document.createElement('div');
        el.id = 'recording-status';
        el.style.cssText = `
            position: fixed;
            top: 170px;
            right: 16px;
            z-index: 10000;
            padding: 8px 14px;
            background: rgba(20, 20, 20, 0.92);
            border: 1px solid #ff3b30;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #f0f6fc;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(8px);
            pointer-events: none;
            display: none;
        `;
        document.body.appendChild(el);
        RECORDING_STATE.statusEl = el;
        return el;
    }

    function setStatus(html, visible = true) {
        const el = ensureStatusElement();
        el.innerHTML = html;
        el.style.display = visible ? 'block' : 'none';
    }

    function fmtTime(ms) {
        const s = Math.floor(ms / 1000);
        const mm = String(Math.floor(s / 60)).padStart(2, '0');
        const ss = String(s % 60).padStart(2, '0');
        return `${mm}:${ss}`;
    }

    function fmtSize(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }

    function pickMimeType() {
        const candidates = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4', // some Safari builds
        ];
        for (const t of candidates) {
            if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(t)) return t;
        }
        return '';
    }

    function timestampString() {
        const d = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
    }

    // -------------------------------------------------------------------------
    // EMA frame-blending helpers
    // -------------------------------------------------------------------------
    /**
     * Compute the pixel-space bounding box of the grid within the GL canvas.
     * Mirrors the aspect-ratio logic in the render shader (main.js ~line 1458).
     */
    function getGridRect(glCanvas) {
        const cw = glCanvas.width;
        const ch = glCanvas.height;
        // CONFIG is a global from petri-dish.js
        const gw = (typeof CONFIG !== 'undefined') ? CONFIG.GRID_W : cw;
        const gh = (typeof CONFIG !== 'undefined') ? CONFIG.GRID_H : ch;
        const canvasAspect = cw / ch;
        const gridAspect = gw / gh;
        let w, h, x, y;
        if (canvasAspect > gridAspect) {
            // Canvas wider than grid → bars on left/right
            h = ch;
            w = h * gridAspect;
            x = (cw - w) / 2;
            y = 0;
        } else {
            // Canvas taller than grid → bars on top/bottom
            w = cw;
            h = w / gridAspect;
            x = 0;
            y = (ch - h) / 2;
        }
        return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
    }

    function initBlendState(glCanvas) {
        const rect = getGridRect(glCanvas);
        const w = rect.w;
        const h = rect.h;
        let bc = RECORDING_STATE.blendCanvas;
        if (!bc || bc.width !== w || bc.height !== h) {
            bc = document.createElement('canvas');
            bc.width = w;
            bc.height = h;
            RECORDING_STATE.blendCanvas = bc;
            RECORDING_STATE.blendCtx = bc.getContext('2d');
        }
        RECORDING_STATE.blendAcc = new Float32Array(w * h * 4);
        RECORDING_STATE.blendTempU8 = new Uint8Array(w * h * 4);
        RECORDING_STATE._gridRect = rect;
        // Seed accumulator with current frame (grid region only)
        const gl = glCanvas.getContext('webgl2') || glCanvas.getContext('webgl');
        if (gl) {
            gl.readPixels(rect.x, rect.y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, RECORDING_STATE.blendTempU8);
            for (let i = 0; i < RECORDING_STATE.blendTempU8.length; i++) {
                RECORDING_STATE.blendAcc[i] = RECORDING_STATE.blendTempU8[i];
            }
        }
    }

    /**
     * Initialise the crop-only offscreen canvas (no EMA blending) so the
     * recording captures only the grid region, never the grey letterbox bars.
     */
    function initCropState(glCanvas) {
        const rect = getGridRect(glCanvas);
        let bc = RECORDING_STATE.blendCanvas;
        if (!bc || bc.width !== rect.w || bc.height !== rect.h) {
            bc = document.createElement('canvas');
            bc.width = rect.w;
            bc.height = rect.h;
            RECORDING_STATE.blendCanvas = bc;
            RECORDING_STATE.blendCtx = bc.getContext('2d');
        }
        RECORDING_STATE._gridRect = rect;
    }

    function blendTick(glCanvas) {
        const alpha = RECORDING_STATE.BLEND_ALPHA;
        const rect = RECORDING_STATE._gridRect;
        if (!rect) return;
        const w = rect.w;
        const h = rect.h;
        const acc = RECORDING_STATE.blendAcc;
        const tmp = RECORDING_STATE.blendTempU8;
        const gl = glCanvas.getContext('webgl2') || glCanvas.getContext('webgl');
        if (!gl || !acc || !tmp) return;

        // Read only the grid region (skip grey letterbox bars)
        gl.readPixels(rect.x, rect.y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, tmp);

        // EMA: acc = α·acc + (1-α)·current
        const oneMinusAlpha = 1.0 - alpha;
        for (let i = 0; i < tmp.length; i++) {
            acc[i] = alpha * acc[i] + oneMinusAlpha * tmp[i];
        }

        // Write acc to the offscreen 2D canvas (WebGL readPixels is bottom-up → flip Y)
        const ctx = RECORDING_STATE.blendCtx;
        const imgData = ctx.createImageData(w, h);
        const dst = imgData.data;
        for (let row = 0; row < h; row++) {
            const srcRow = (h - 1 - row) * w * 4;
            const dstRow = row * w * 4;
            for (let col = 0; col < w * 4; col++) {
                dst[dstRow + col] = Math.min(255, Math.max(0, Math.round(acc[srcRow + col])));
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }

    function startBlendLoop(glCanvas) {
        stopBlendLoop();
        const targetMs = 1000 / Math.max(1, RECORDING_STATE.FPS);
        let lastTick = 0;
        function loop(now) {
            if (!RECORDING_STATE.recorder) return; // stopped
            RECORDING_STATE.blendAnimId = requestAnimationFrame(loop);
            if (now - lastTick < targetMs) return;
            lastTick = now;
            blendTick(glCanvas);
        }
        RECORDING_STATE.blendAnimId = requestAnimationFrame(loop);
    }

    function stopBlendLoop() {
        if (RECORDING_STATE.blendAnimId) {
            cancelAnimationFrame(RECORDING_STATE.blendAnimId);
            RECORDING_STATE.blendAnimId = null;
        }
    }

    function startRecording(canvas) {
        if (RECORDING_STATE.recorder) {
            // Already recording — warn and discard the request
            if (typeof showToast === 'function') {
                showToast('Recording already in progress — stop it first.');
            } else {
                console.warn('Recording already in progress.');
            }
            return false;
        }
        if (typeof MediaRecorder === 'undefined') {
            alert('MediaRecorder API not available in this browser.');
            return false;
        }

        const bitsPerSecond = Math.max(1, Math.min(20, RECORDING_STATE.BITRATE_MBPS)) * 1_000_000;
        const useBlend = RECORDING_STATE.BLEND_ALPHA > 0.001;

        // Always use an offscreen canvas (cropped to the grid region, no grey bars)
        if (useBlend) {
            initBlendState(canvas);
        } else {
            initCropState(canvas);
        }
        const streamCanvas = RECORDING_STATE.blendCanvas;
        RECORDING_STATE._useBlend = useBlend;
        RECORDING_STATE._sourceCanvas = canvas;

        // captureStream(fps) polls the offscreen canvas at the target framerate.
        // recordingCaptureFrame() (called per sim step) keeps the offscreen canvas
        // updated with the latest grid crop / EMA blend; captureStream samples it
        // at the right playback rate so the video plays at the correct speed.
        const fps = Math.max(1, Math.min(60, Math.round(RECORDING_STATE.FPS)));
        let stream;
        try {
            stream = streamCanvas.captureStream(fps);
        } catch (err) {
            console.error('captureStream failed:', err);
            alert('canvas.captureStream() failed: ' + err.message);
            return false;
        }
        RECORDING_STATE._videoTrack = stream.getVideoTracks()[0];

        const mimeType = pickMimeType();
        const opts = { videoBitsPerSecond: bitsPerSecond };
        if (mimeType) opts.mimeType = mimeType;

        let recorder;
        try {
            recorder = new MediaRecorder(stream, opts);
        } catch (err) {
            console.error('MediaRecorder constructor failed:', err);
            alert('Could not start MediaRecorder: ' + err.message);
            return false;
        }

        RECORDING_STATE.chunks = [];
        RECORDING_STATE.bytesSoFar = 0;
        RECORDING_STATE.startTime = performance.now();
        RECORDING_STATE.stream = stream;
        RECORDING_STATE.recorder = recorder;
        RECORDING_STATE.lastBlob = null;
        RECORDING_STATE.lastBlobName = null;

        recorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
                RECORDING_STATE.chunks.push(e.data);
                RECORDING_STATE.bytesSoFar += e.data.size;
            }
        };
        recorder.onerror = (e) => {
            console.error('MediaRecorder error:', e);
            if (typeof showToast === 'function') showToast('Recording error — see console.');
        };
        recorder.onstop = () => {
            const type = mimeType || (RECORDING_STATE.chunks[0] && RECORDING_STATE.chunks[0].type) || 'video/webm';
            const blob = new Blob(RECORDING_STATE.chunks, { type });
            const ext = type.includes('mp4') ? 'mp4' : 'webm';
            const filename = `petri-dish-${timestampString()}.${ext}`;
            RECORDING_STATE.lastBlob = blob;
            RECORDING_STATE.lastBlobName = filename;
            triggerDownload(blob, filename);
            // Clean up stream tracks
            try { stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
            RECORDING_STATE.recorder = null;
            RECORDING_STATE.stream = null;
            stopStatusTimer();
            setStatus(`Saved ${filename} (${fmtSize(blob.size)})`, true);
            setTimeout(() => setStatus('', false), 5000);
            if (typeof showToast === 'function') showToast(`Recording saved: ${filename}`);
            // Rename the start/stop button label back
            if (RECORDING_STATE.recordController) {
                RECORDING_STATE.recordController.name('Start recording');
            }
        };

        recorder.start(1000); // emit a chunk every 1s for live byte count
        RECORDING_STATE._frameCount = 0;
        startStatusTimer();
        if (RECORDING_STATE.recordController) {
            RECORDING_STATE.recordController.name('Stop recording');
        }
        if (typeof showToast === 'function') {
            showToast(`Recording at ${fps} fps, ${RECORDING_STATE.BITRATE_MBPS} Mbps`);
        }
        return true;
    }

    function stopRecording() {
        stopBlendLoop();
        const r = RECORDING_STATE.recorder;
        if (!r) return;
        if (r.state !== 'inactive') r.stop();
    }

    function startStatusTimer() {
        stopStatusTimer();
        const tick = () => {
            if (!RECORDING_STATE.recorder) return;
            const elapsed = performance.now() - RECORDING_STATE.startTime;
            const frames = RECORDING_STATE._frameCount || 0;
            setStatus(
                `<span style="color:#ff3b30;">&#9679; REC</span> ` +
                `${frames} frames &nbsp; ${fmtSize(RECORDING_STATE.bytesSoFar)}`,
                true
            );
        };
        tick();
        RECORDING_STATE.statusTimer = setInterval(tick, 250);
    }

    function stopStatusTimer() {
        if (RECORDING_STATE.statusTimer) {
            clearInterval(RECORDING_STATE.statusTimer);
            RECORDING_STATE.statusTimer = null;
        }
    }

    function triggerDownload(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    // MP4 conversion removed — use system ffmpeg post-hoc:
    // ffmpeg -i petri-dish-*.webm -c:v libx264 -pix_fmt yuv420p -crf 23 output.mp4

    // -------------------------------------------------------------------------
    // Public: setupRecording(canvas, gui)
    // -------------------------------------------------------------------------
    function setupRecording(canvas, gui) {
        if (!canvas) {
            console.warn('setupRecording: no canvas provided');
            return;
        }
        if (!gui || typeof gui.addFolder !== 'function') {
            console.warn('setupRecording: no dat.GUI instance provided');
            return;
        }

        const folder = gui.addFolder('Recording');

        const proxy = {
            FPS: RECORDING_STATE.FPS,
            BITRATE_MBPS: RECORDING_STATE.BITRATE_MBPS,
            BLEND_ALPHA: RECORDING_STATE.BLEND_ALPHA,
            toggleRecord: function () {
                if (RECORDING_STATE.recorder) {
                    stopRecording();
                } else {
                    startRecording(canvas);
                }
            },
        };

        const fpsCtrl = folder.add(proxy, 'FPS', 1, 60, 1).name('FPS').onChange((v) => {
            RECORDING_STATE.FPS = v;
        });
        const brCtrl = folder.add(proxy, 'BITRATE_MBPS', 1, 20, 1).name('Bitrate (Mbps)').onChange((v) => {
            RECORDING_STATE.BITRATE_MBPS = v;
        });
        const blendCtrl = folder.add(proxy, 'BLEND_ALPHA', 0, 0.98, 0.01).name('Frame blend (EMA)').onChange((v) => {
            RECORDING_STATE.BLEND_ALPHA = v;
        });
        try { blendCtrl.domElement.parentElement.parentElement.title = 'EMA blend factor. 0 = raw frames. 0.5 = moderate trail. 0.9+ = heavy smear. Only affects the exported video, not the live canvas.'; } catch (_) {}
        const recordCtrl = folder.add(proxy, 'toggleRecord').name('Start recording');
        RECORDING_STATE.recordController = recordCtrl;
        try {
            recordCtrl.domElement.parentElement.parentElement.classList.add('btn-export');
        } catch (_) {}

        folder.close();
        return folder;
    }

    // -------------------------------------------------------------------------
    // Per-step frame capture — called from the simulation loop in main.js
    // -------------------------------------------------------------------------
    function recordingCaptureFrame() {
        if (!RECORDING_STATE.recorder || RECORDING_STATE.recorder.state !== 'recording') return;
        const track = RECORDING_STATE._videoTrack;
        if (!track) return;

        const src = RECORDING_STATE._sourceCanvas;
        if (RECORDING_STATE._useBlend) {
            // EMA blend + crop to grid region
            blendTick(src);
        } else {
            // Crop-only: drawImage the grid region from the GL canvas to the offscreen 2D canvas
            const rect = RECORDING_STATE._gridRect;
            const ctx = RECORDING_STATE.blendCtx;
            if (rect && ctx && src) {
                ctx.drawImage(src, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);
            }
        }
        // Signal the stream to capture exactly this frame
        if (typeof track.requestFrame === 'function') {
            track.requestFrame();
        }
        RECORDING_STATE._frameCount = (RECORDING_STATE._frameCount || 0) + 1;
    }

    // Expose globally — no module system in this project
    window.setupRecording = setupRecording;
    window.recordingCaptureFrame = recordingCaptureFrame;
})();
