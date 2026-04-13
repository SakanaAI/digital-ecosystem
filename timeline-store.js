/**
 * Timeline Store - Data Layer for Timeline, Checkpoints, and Replay
 *
 * Provides: Seeded PRNG, MetricsBuffer, EventLog, RunManager, StorageManager
 */

// =============================================================================
// SEEDED PRNG (Mulberry32)
// =============================================================================

/**
 * Creates a seeded pseudo-random number generator using Mulberry32.
 * Returns a function that produces values in [0, 1).
 */
function createSeededRNG(seed) {
    let state = seed | 0;
    return function() {
        state = (state + 0x6D2B79F5) | 0;
        let t = Math.imul(state ^ (state >>> 15), 1 | state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Global seeded RNG instance — replaces Math.random() for reproducibility
let _seededRNG = null;
let _currentSeed = null;

function initSeededRNG(seed) {
    if (seed === undefined || seed === null) {
        seed = Date.now() ^ (Math.random() * 0xFFFFFFFF);
    }
    _currentSeed = seed;
    _seededRNG = createSeededRNG(seed);
    return seed;
}

function seededRandom() {
    if (!_seededRNG) initSeededRNG();
    return _seededRNG();
}

function getCurrentSeed() {
    return _currentSeed;
}

// =============================================================================
// UUID GENERATION
// =============================================================================

function generateUUID() {
    // Simple UUID v4 generator
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

// =============================================================================
// METRICS BUFFER (Ring Buffer, Struct-of-Arrays)
// =============================================================================

class MetricsBuffer {
    /**
     * @param {number} capacity - Max data points
     * @param {number} maxSpecies - Max species to track (allocates upfront)
     */
    constructor(capacity = 20000, maxSpecies = 25) {
        this.capacity = capacity;
        this.maxSpecies = maxSpecies;
        this.head = 0;
        this.length = 0;

        // Struct-of-arrays for efficient per-metric access
        this.steps = new Uint32Array(capacity);
        this.losses = new Float32Array(capacity);
        this.entropy = new Float32Array(capacity);

        // Per-species populations: flat array [capacity * (maxSpecies + 1)]
        // +1 for sun. Access: populations[(speciesIdx * capacity) + ringIdx]
        this.populationData = new Float32Array(capacity * (maxSpecies + 1));
        this.speciesCount = 0; // Current active species count (set externally)
    }

    push(step, loss, populations, shannonEntropy) {
        const idx = this.head;

        this.steps[idx] = step;
        this.losses[idx] = loss;
        this.entropy[idx] = shannonEntropy;

        // Store population data for sun + each species
        const nChannels = Math.min(populations.length, this.maxSpecies + 1);
        for (let i = 0; i < nChannels; i++) {
            this.populationData[i * this.capacity + idx] = populations[i];
        }

        this.head = (this.head + 1) % this.capacity;
        if (this.length < this.capacity) this.length++;
    }

    /**
     * Get the value at logical index i (0 = oldest, length-1 = newest).
     */
    _ringIndex(logicalIdx) {
        if (this.length < this.capacity) return logicalIdx;
        return (this.head + logicalIdx) % this.capacity;
    }

    getStep(i) { return this.steps[this._ringIndex(i)]; }
    getLoss(i) { return this.losses[this._ringIndex(i)]; }
    getEntropy(i) { return this.entropy[this._ringIndex(i)]; }

    getPopulation(speciesIdx, logicalIdx) {
        return this.populationData[speciesIdx * this.capacity + this._ringIndex(logicalIdx)];
    }

    /** Get all populations at a logical index as an array. */
    getPopulations(logicalIdx) {
        const idx = this._ringIndex(logicalIdx);
        const nChannels = this.speciesCount + 1; // +1 for sun
        const result = new Float32Array(nChannels);
        for (let i = 0; i < nChannels; i++) {
            result[i] = this.populationData[i * this.capacity + idx];
        }
        return result;
    }

    /** Get recent N entries as arrays (for timeline rendering). */
    getRecent(count) {
        const n = Math.min(count, this.length);
        const start = this.length - n;
        const steps = new Uint32Array(n);
        const losses = new Float32Array(n);
        const entropy = new Float32Array(n);

        for (let i = 0; i < n; i++) {
            const idx = this._ringIndex(start + i);
            steps[i] = this.steps[idx];
            losses[i] = this.losses[idx];
            entropy[i] = this.entropy[idx];
        }

        return { steps, losses, entropy, count: n, startLogical: start };
    }

    /** Get population timeseries for a specific species. */
    getSpeciesPopulationSeries(speciesIdx, count) {
        const n = Math.min(count, this.length);
        const start = this.length - n;
        const result = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            result[i] = this.populationData[speciesIdx * this.capacity + this._ringIndex(start + i)];
        }
        return result;
    }

    /** Compute population fractions at a logical index (for stacked area chart). */
    getPopulationFractions(logicalIdx) {
        const pops = this.getPopulations(logicalIdx);
        let total = 0;
        for (let i = 0; i < pops.length; i++) total += pops[i];
        if (total < 1e-8) return pops; // all dead
        for (let i = 0; i < pops.length; i++) pops[i] /= total;
        return pops;
    }

    /**
     * Truncate buffer to only keep entries at or before the given step.
     * Used when restoring a checkpoint to avoid timeline overlap.
     */
    truncateToStep(step) {
        if (this.length === 0) return;

        // Find how many entries to keep
        let keepCount = 0;
        for (let i = 0; i < this.length; i++) {
            if (this.getStep(i) <= step) {
                keepCount = i + 1;
            } else {
                break;
            }
        }

        if (keepCount === this.length) return; // Nothing to truncate

        // Adjust head and length
        this.length = keepCount;
        if (keepCount === 0) {
            this.head = 0;
        } else {
            this.head = this._ringIndex(keepCount - 1) + 1;
            if (this.head >= this.capacity) this.head = 0;
        }
    }

    clear() {
        this.head = 0;
        this.length = 0;
    }

    /** Export as serializable object (for .petri bundles). */
    serialize() {
        const n = this.length;
        const steps = new Uint32Array(n);
        const losses = new Float32Array(n);
        const entropy = new Float32Array(n);
        const nChannels = this.speciesCount + 1;
        const populations = new Float32Array(n * nChannels);

        for (let i = 0; i < n; i++) {
            const idx = this._ringIndex(i);
            steps[i] = this.steps[idx];
            losses[i] = this.losses[idx];
            entropy[i] = this.entropy[idx];
            for (let c = 0; c < nChannels; c++) {
                populations[i * nChannels + c] = this.populationData[c * this.capacity + idx];
            }
        }

        return {
            steps: steps.buffer,
            losses: losses.buffer,
            entropy: entropy.buffer,
            populations: populations.buffer,
            speciesCount: this.speciesCount,
            length: n,
        };
    }

    /** Restore from serialized data. */
    static deserialize(data) {
        const buf = new MetricsBuffer(Math.max(data.length * 2, 20000), 25);
        const steps = new Uint32Array(data.steps);
        const losses = new Float32Array(data.losses);
        const entropy = new Float32Array(data.entropy);
        const populations = new Float32Array(data.populations);
        const nChannels = data.speciesCount + 1;
        buf.speciesCount = data.speciesCount;

        for (let i = 0; i < data.length; i++) {
            const pops = [];
            for (let c = 0; c < nChannels; c++) {
                pops.push(populations[i * nChannels + c]);
            }
            buf.push(steps[i], losses[i], pops, entropy[i]);
        }
        return buf;
    }
}

// =============================================================================
// EVENT LOG
// =============================================================================

class EventLog {
    constructor() {
        this.events = [];
    }

    /**
     * Log a parameter change.
     * @param {number} step - Global simulation step
     * @param {string} param - Parameter name (e.g., "SOFTMAX_TEMP")
     * @param {*} oldValue - Previous value
     * @param {*} newValue - New value
     */
    logParamChange(step, param, oldValue, newValue) {
        this.events.push({
            type: 'param',
            step,
            param,
            oldValue,
            newValue,
            timestamp: Date.now(),
        });
    }

    /**
     * Log a drawing action (aggregated per stroke, not per pixel).
     */
    logDraw(step, tool, x, y, radius, species) {
        this.events.push({
            type: 'draw',
            step,
            tool,
            x, y, radius, species,
        });
    }

    /**
     * Log a reset or reseed event.
     */
    logReset(step, kind) {
        this.events.push({
            type: kind, // 'reset' | 'reseed'
            step,
            timestamp: Date.now(),
        });
    }

    /**
     * Get all events of a specific type.
     */
    getByType(type) {
        return this.events.filter(e => e.type === type);
    }

    /**
     * Get param changes for timeline markers.
     */
    getParamChanges() {
        return this.events.filter(e => e.type === 'param');
    }

    /**
     * Get events at or after a specific step.
     */
    getEventsFromStep(step) {
        return this.events.filter(e => e.step >= step);
    }

    /**
     * Get the next event after a given step (for replay).
     */
    getNextEvent(afterStep) {
        for (let i = 0; i < this.events.length; i++) {
            if (this.events[i].step > afterStep) return this.events[i];
        }
        return null;
    }

    clear() {
        this.events = [];
    }

    serialize() {
        return JSON.parse(JSON.stringify(this.events));
    }

    static deserialize(data) {
        const log = new EventLog();
        log.events = data || [];
        return log;
    }
}

// =============================================================================
// RUN MANAGER
// =============================================================================

class RunManager {
    constructor() {
        this.currentRun = null;
        this.metrics = new MetricsBuffer();
        this.eventLog = new EventLog();
        this.globalStep = 0;
        this.checkpoints = new Map(); // id -> checkpoint metadata (no heavy data)
        this.mode = 'LIVE'; // 'LIVE' | 'REPLAY' | 'TAKEOVER'

        // Replay state
        this.replayEventLog = null;
        this.replayEventIdx = 0;
        this.replayMaxStep = 0;
        this.replaySpeed = 1;

        // Auto-save slot
        this.autoSaveId = null;
        this.lastAutoSaveStep = 0;
    }

    /**
     * Start a new run.
     */
    startNewRun(seed, config, guiState) {
        this.currentRun = {
            id: generateUUID(),
            seed: seed,
            createdAt: new Date().toISOString(),
            initialConfig: {
                CONFIG: { ...config },
                GUI_STATE: { ...guiState },
            },
            checkpointIds: [],
            parentRunId: null,
            parentStep: null,
            formatVersion: 1,
            lastStep: 0,
            status: 'active',
        };

        this.globalStep = 0;
        this.metrics = new MetricsBuffer();
        this.metrics.speciesCount = config.N_NCAS;
        this.eventLog = new EventLog();
        this.checkpoints.clear();
        this.mode = 'LIVE';
        this.autoSaveId = null;
        this.lastAutoSaveStep = 0;

        return this.currentRun;
    }

    /**
     * Start a new run branching from a checkpoint.
     */
    startBranchRun(parentRunId, parentStep, seed, config, guiState) {
        const run = this.startNewRun(seed, config, guiState);
        run.parentRunId = parentRunId;
        run.parentStep = parentStep;
        return run;
    }

    /**
     * Record one simulation step's metrics.
     */
    recordStep(loss, populations, shannonEntropy) {
        this.globalStep++;
        this.metrics.push(this.globalStep, loss, populations, shannonEntropy);
        if (this.currentRun) {
            this.currentRun.lastStep = this.globalStep;
        }
    }

    /**
     * Check if it's time for an auto-save (every 500 steps).
     */
    shouldAutoSave() {
        return this.mode === 'LIVE' &&
               this.globalStep > 0 &&
               this.globalStep - this.lastAutoSaveStep >= 500;
    }

    /**
     * Register a checkpoint (metadata only — heavy data stored separately).
     */
    registerCheckpoint(checkpoint) {
        this.checkpoints.set(checkpoint.id, {
            id: checkpoint.id,
            step: checkpoint.step,
            label: checkpoint.label,
            trigger: checkpoint.trigger,
        });
        if (this.currentRun) {
            if (!this.currentRun.checkpointIds.includes(checkpoint.id)) {
                this.currentRun.checkpointIds.push(checkpoint.id);
            }
        }
        if (checkpoint.trigger === 'auto') {
            this.autoSaveId = checkpoint.id;
            this.lastAutoSaveStep = this.globalStep;
        }
    }

    /**
     * Get user (non-auto) checkpoint count.
     */
    getUserCheckpointCount() {
        let count = 0;
        for (const cp of this.checkpoints.values()) {
            if (cp.trigger !== 'auto') count++;
        }
        return count;
    }

    /**
     * Get all checkpoints sorted by step.
     */
    getCheckpointsSorted() {
        return Array.from(this.checkpoints.values())
            .sort((a, b) => a.step - b.step);
    }

    /**
     * Enter replay mode.
     */
    enterReplayMode(eventLog, maxStep) {
        this.mode = 'REPLAY';
        this.replayEventLog = eventLog;
        this.replayEventIdx = 0;
        this.replayMaxStep = maxStep;
        this.replaySpeed = 1;
    }

    /**
     * Take over from replay (branch into live mode).
     */
    takeover(parentRunId, parentStep, seed, config, guiState) {
        this.mode = 'TAKEOVER';
        // Start a branch run preserving current state
        this.startBranchRun(parentRunId, parentStep, seed, config, guiState);
        this.mode = 'LIVE';
    }

    /**
     * Exit replay mode back to live.
     */
    exitReplay() {
        this.mode = 'LIVE';
        this.replayEventLog = null;
        this.replayEventIdx = 0;
    }
}

// =============================================================================
// STORAGE MANAGER (IndexedDB)
// =============================================================================

const DB_NAME = 'petri-dish-db';
const DB_VERSION = 1;

class StorageManager {
    constructor() {
        this.db = null;
        this._initPromise = null;
    }

    /**
     * Initialize IndexedDB. Call once at app start.
     */
    async init() {
        if (this.db) return;
        if (this._initPromise) return this._initPromise;

        this._initPromise = new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onupgradeneeded = (event) => {
                const db = event.target.result;

                if (!db.objectStoreNames.contains('runs')) {
                    const runStore = db.createObjectStore('runs', { keyPath: 'id' });
                    runStore.createIndex('createdAt', 'createdAt');
                }
                if (!db.objectStoreNames.contains('checkpoints')) {
                    const cpStore = db.createObjectStore('checkpoints', { keyPath: 'id' });
                    cpStore.createIndex('runId', 'runId');
                    cpStore.createIndex('step', 'step');
                }
                if (!db.objectStoreNames.contains('timeseries')) {
                    db.createObjectStore('timeseries', { keyPath: ['runId', 'chunkIndex'] });
                }
                if (!db.objectStoreNames.contains('events')) {
                    db.createObjectStore('events', { keyPath: 'runId' });
                }
                if (!db.objectStoreNames.contains('meta')) {
                    db.createObjectStore('meta', { keyPath: 'key' });
                }
            };

            request.onsuccess = (event) => {
                this.db = event.target.result;
                resolve();
            };

            request.onerror = (event) => {
                console.error('IndexedDB open failed:', event.target.error);
                reject(event.target.error);
            };
        });

        return this._initPromise;
    }

    /**
     * Generic put into an object store.
     */
    async _put(storeName, data) {
        await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const request = store.put(data);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Generic get from an object store.
     */
    async _get(storeName, key) {
        await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction(storeName, 'readonly');
            const store = tx.objectStore(storeName);
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Generic delete from an object store.
     */
    async _delete(storeName, key) {
        await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const request = store.delete(key);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Get all records from a store.
     */
    async _getAll(storeName) {
        await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction(storeName, 'readonly');
            const store = tx.objectStore(storeName);
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    // --- Runs ---

    async saveRun(run) {
        return this._put('runs', run);
    }

    async getRun(id) {
        return this._get('runs', id);
    }

    async getAllRuns() {
        return this._getAll('runs');
    }

    async deleteRun(id) {
        // Also delete associated checkpoints, timeseries, events
        await this.init();
        const tx = this.db.transaction(['runs', 'checkpoints', 'events'], 'readwrite');

        // Delete checkpoints by index
        const cpStore = tx.objectStore('checkpoints');
        const cpIndex = cpStore.index('runId');
        const cpRequest = cpIndex.getAllKeys(id);
        cpRequest.onsuccess = () => {
            for (const key of cpRequest.result) {
                cpStore.delete(key);
            }
        };

        tx.objectStore('events').delete(id);
        tx.objectStore('runs').delete(id);

        return new Promise((resolve, reject) => {
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    }

    // --- Checkpoints ---

    async saveCheckpoint(checkpoint) {
        return this._put('checkpoints', checkpoint);
    }

    async getCheckpoint(id) {
        return this._get('checkpoints', id);
    }

    async getCheckpointsForRun(runId) {
        await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction('checkpoints', 'readonly');
            const store = tx.objectStore('checkpoints');
            const index = store.index('runId');
            const request = index.getAll(runId);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async deleteCheckpoint(id) {
        return this._delete('checkpoints', id);
    }

    // --- Events ---

    async saveEvents(runId, events) {
        return this._put('events', { runId, events });
    }

    async getEvents(runId) {
        const result = await this._get('events', runId);
        return result ? result.events : [];
    }

    // --- Meta ---

    async setMeta(key, value) {
        return this._put('meta', { key, value });
    }

    async getMeta(key) {
        const result = await this._get('meta', key);
        return result ? result.value : null;
    }

    // --- Storage estimate ---

    async getStorageEstimate() {
        if (navigator.storage && navigator.storage.estimate) {
            const est = await navigator.storage.estimate();
            return {
                usage: est.usage || 0,
                quota: est.quota || 0,
                percentUsed: est.quota ? ((est.usage / est.quota) * 100) : 0,
            };
        }
        return { usage: 0, quota: 0, percentUsed: 0 };
    }
}

// =============================================================================
// METRICS COMPUTATION HELPERS
// =============================================================================

/**
 * Compute Shannon entropy from population array.
 * @param {Float32Array|number[]} populations - Per-species populations (excluding sun)
 * @returns {number} Normalized entropy in [0, 1]
 */
function computeShannonEntropy(populations) {
    let total = 0;
    for (let i = 0; i < populations.length; i++) total += populations[i];
    if (total < 1e-8) return 0;

    let entropy = 0;
    for (let i = 0; i < populations.length; i++) {
        const p = populations[i] / total;
        if (p > 1e-8) {
            entropy -= p * Math.log(p);
        }
    }

    const maxEntropy = Math.log(populations.length);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
}

/**
 * Extract per-species population counts from the grid tensor (GPU-side reduce).
 * Returns a Promise resolving to Float32Array [sun, species0, species1, ...].
 */
async function computePopulations(grid, aliveDim) {
    const popTensor = tf.tidy(() => {
        const alive = grid.slice([0, 0, 0, 0], [-1, -1, -1, aliveDim]);
        return alive.mean([1, 2]).squeeze(); // [ALIVE_DIM]
    });
    const data = await popTensor.data();
    popTensor.dispose();
    return data; // Float32Array
}

// =============================================================================
// EXPORT/IMPORT: .petri BUNDLE FORMAT
// =============================================================================

const PETRI_MAGIC = 'PTRI';
const PETRI_FORMAT_VERSION = 1;

/**
 * Export a complete run as a .petri binary bundle.
 * Format: [PTRI][version:u16][headerLen:u32][headerJSON][binaryData]
 */
async function exportPetriBundleForRun(runId) {
    const run = await storageManager.getRun(runId);
    if (!run) throw new Error('Run not found');

    const events = await storageManager.getEvents(runId);
    const checkpoints = await storageManager.getCheckpointsForRun(runId);
    const metricsData = runManager.metrics.serialize();

    // Build binary sections: concatenate all ArrayBuffers
    const binarySections = [];
    let offset = 0;

    // Checkpoint sections
    const cpMeta = [];
    for (const cp of checkpoints) {
        const sections = {};
        for (const key of ['grid', 'wall', 'winRate', 'sunBase', 'sunParams']) {
            if (cp[key]) {
                sections[key] = { offset, length: cp[key].byteLength };
                binarySections.push(cp[key]);
                offset += cp[key].byteLength;
            }
        }
        // Model weights
        if (cp.modelWeights) {
            sections.modelWeights = cp.modelWeights.map(w => {
                const entry = { index: w.index, shape: w.shape, offset, length: w.data.byteLength };
                binarySections.push(w.data);
                offset += w.data.byteLength;
                return entry;
            });
        }
        cpMeta.push({
            id: cp.id,
            step: cp.step,
            label: cp.label,
            trigger: cp.trigger,
            config: cp.config,
            gridShape: cp.gridShape,
            wallShape: cp.wallShape,
            winRateShape: cp.winRateShape,
            sunBaseShape: cp.sunBaseShape,
            sunParamsShape: cp.sunParamsShape,
            sections,
        });
    }

    // Timeseries sections
    const tsMeta = {};
    for (const key of ['steps', 'losses', 'entropy', 'populations']) {
        if (metricsData[key]) {
            tsMeta[key] = { offset, length: metricsData[key].byteLength };
            binarySections.push(metricsData[key]);
            offset += metricsData[key].byteLength;
        }
    }
    tsMeta.speciesCount = metricsData.speciesCount;
    tsMeta.length = metricsData.length;

    // Build header
    const header = JSON.stringify({
        formatVersion: PETRI_FORMAT_VERSION,
        run,
        events,
        checkpoints: cpMeta,
        timeseries: tsMeta,
    });

    const headerBytes = new TextEncoder().encode(header);
    const totalBinaryLen = binarySections.reduce((sum, buf) => sum + buf.byteLength, 0);

    // Assemble bundle: magic(4) + version(2) + headerLen(4) + header + binary
    const bundle = new ArrayBuffer(4 + 2 + 4 + headerBytes.byteLength + totalBinaryLen);
    const view = new DataView(bundle);
    const uint8 = new Uint8Array(bundle);

    // Magic
    uint8[0] = 0x50; uint8[1] = 0x54; uint8[2] = 0x52; uint8[3] = 0x49; // PTRI
    // Version
    view.setUint16(4, PETRI_FORMAT_VERSION, true);
    // Header length
    view.setUint32(6, headerBytes.byteLength, true);
    // Header
    uint8.set(headerBytes, 10);
    // Binary sections
    let pos = 10 + headerBytes.byteLength;
    for (const buf of binarySections) {
        uint8.set(new Uint8Array(buf), pos);
        pos += buf.byteLength;
    }

    return bundle;
}

/**
 * Parse a .petri bundle.
 * Returns { header, binaryData } where binaryData is the raw binary after the header.
 */
function parsePetriBundle(buffer) {
    const view = new DataView(buffer);
    const uint8 = new Uint8Array(buffer);

    // Verify magic
    if (uint8[0] !== 0x50 || uint8[1] !== 0x54 || uint8[2] !== 0x52 || uint8[3] !== 0x49) {
        throw new Error('Not a valid .petri file');
    }

    const version = view.getUint16(4, true);
    if (version > PETRI_FORMAT_VERSION) {
        console.warn(`Petri format version ${version} is newer than supported ${PETRI_FORMAT_VERSION}`);
    }

    const headerLen = view.getUint32(6, true);
    const headerBytes = uint8.slice(10, 10 + headerLen);
    const header = JSON.parse(new TextDecoder().decode(headerBytes));
    const binaryStart = 10 + headerLen;
    const binaryData = buffer.slice(binaryStart);

    return { header, binaryData };
}

/**
 * Extract a checkpoint from a parsed .petri bundle.
 */
function extractCheckpointFromBundle(cpMeta, binaryData) {
    const result = {
        id: cpMeta.id,
        runId: cpMeta.run?.id,
        step: cpMeta.step,
        label: cpMeta.label,
        trigger: cpMeta.trigger,
        config: cpMeta.config,
        gridShape: cpMeta.gridShape,
        wallShape: cpMeta.wallShape,
        winRateShape: cpMeta.winRateShape,
        sunBaseShape: cpMeta.sunBaseShape,
        sunParamsShape: cpMeta.sunParamsShape,
        formatVersion: 1,
    };

    const sections = cpMeta.sections;
    for (const key of ['grid', 'wall', 'winRate', 'sunBase', 'sunParams']) {
        if (sections[key]) {
            result[key] = binaryData.slice(sections[key].offset, sections[key].offset + sections[key].length);
        }
    }

    if (sections.modelWeights) {
        result.modelWeights = sections.modelWeights.map(w => ({
            index: w.index,
            shape: w.shape,
            data: binaryData.slice(w.offset, w.offset + w.length),
        }));
    }

    return result;
}

// =============================================================================
// URL-ENCODED RECIPES
// =============================================================================

/**
 * Create a compact recipe from current run state.
 * Recipe = seed + config delta + param change events.
 */
function createRecipe(run, eventLog, guiStateDefaults) {
    // Delta config: only non-default values
    const cfg = {};
    const initial = run.initialConfig.GUI_STATE;
    for (const key of Object.keys(initial)) {
        if (initial[key] !== guiStateDefaults[key]) {
            cfg[key] = initial[key];
        }
    }

    // Compact events
    const events = eventLog.getParamChanges().map(e => ({
        t: e.step,
        k: e.param,
        v: e.newValue,
    }));

    return {
        v: run.formatVersion,
        seed: run.seed,
        cfg,
        events,
    };
}

/**
 * Encode a recipe as a URL fragment.
 */
function encodeRecipeToURL(recipe) {
    const json = JSON.stringify(recipe);
    // Base64url encode (no external compression needed for typical sizes)
    const base64 = btoa(unescape(encodeURIComponent(json)))
        .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    return `#r=${base64}`;
}

/**
 * Decode a recipe from a URL fragment.
 */
function decodeRecipeFromURL(hash) {
    if (!hash || !hash.startsWith('#r=')) return null;
    const base64 = hash.slice(3).replace(/-/g, '+').replace(/_/g, '/');
    try {
        const json = decodeURIComponent(escape(atob(base64)));
        return JSON.parse(json);
    } catch (e) {
        console.error('Failed to decode recipe URL:', e);
        return null;
    }
}

// =============================================================================
// SINGLETON INSTANCES
// =============================================================================

const runManager = new RunManager();
const storageManager = new StorageManager();
