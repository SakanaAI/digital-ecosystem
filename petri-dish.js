/**
 * Petri Dish NCA - Core Simulation Logic
 *
 * This file contains the Neural Cellular Automata simulation engine,
 * including model creation, competition mechanics, and training.
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const WALL_PENALTY = -1e9;      // Penalty for attempting to grow into walls
const GRADIENT_CLIP = 1.0;       // Gradient clipping norm
const EPSILON = 1e-8;            // Small value for numerical stability
const ALIVE_POOL_SIZE = 3;       // Kernel size for alive mask pooling
const MIN_PRESENCE_FOR_GROWTH = 0.05;  // Minimum pooled aliveness to participate in competition

// Performance timing flag - enable for detailed breakdown
const PERF_TIMING = true;

// =============================================================================
// PRE-CACHED TENSORS (for training efficiency)
// =============================================================================
// Cache commonly used scalars to avoid creating them on every forward pass
let CACHED_SCALARS = null;

function initCachedScalars() {
    if (CACHED_SCALARS) return;
    CACHED_SCALARS = {
        // Loss computation scalars
        one: tf.keep(tf.scalar(1.0)),
        negOne: tf.keep(tf.scalar(-1.0)),
        epsilon: tf.keep(tf.scalar(EPSILON)),
        // Commonly used penalties/thresholds
        absentPenalty: tf.keep(tf.scalar(1e9)),
        wallPenaltySteepness: tf.keep(tf.scalar(100.0)),
    };
}

function getCachedScalar(name) {
    if (!CACHED_SCALARS) initCachedScalars();
    return CACHED_SCALARS[name];
}

// Performance timing helpers
function perfStart(label) {
    if (PERF_TIMING) performance.mark(`${label}-start`);
}

function perfEnd(label) {
    if (!PERF_TIMING) return;
    performance.mark(`${label}-end`);
    try {
        performance.measure(label, `${label}-start`, `${label}-end`);
        const entries = performance.getEntriesByName(label, 'measure');
        if (entries.length > 0) {
            console.log(`[PERF] ${label}: ${entries[entries.length - 1].duration.toFixed(2)}ms`);
        }
    } catch (e) {
        // Ignore timing errors
    }
    performance.clearMarks(`${label}-start`);
    performance.clearMarks(`${label}-end`);
    performance.clearMeasures(label);
}

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
    // Species
    N_NCAS: 4,

    // Cell structure
    CELL_STATE_DIM: 16,
    CELL_HIDDEN_DIM: 0,
    CELL_CONC_DIM: 1,  // Concentration dimension (1 scalar per NCA)

    // Model architecture
    N_HIDDEN_LAYERS: 1,
    HIDDEN_DIM: 16,
    KERNEL_SIZE: 3,

    // Visibility
    ALIVE_VISIBLE: false,
    ALIVE_MASK_STEEPNESS: 10.0,

    // Grid (synced from GUI_STATE)
    GRID_W: 128,
    GRID_H: 128,

    // Derived values (computed in updateDerivedConfig)
    ALIVE_DIM: 5,
    CELL_WO_ALIVE_DIM: 16,
    CELL_DIM: 21,
    PACKED_PIXELS: 6,
    ATT_IDXS: [0, 1, 2, 3, 4, 5, 6, 7],
    DEF_IDXS: [8, 9, 10, 11, 12, 13, 14, 15],
    SUN_INIT_SCALE: 0.5,
};

const GUI_STATE = {
    // Simulation parameters (user-tested working values)
    SOFTMAX_TEMP: 1.0,          // Sharpness - lower = more brutal competition
    UPDATE_DELAY_MS: 0,
    OPTIMIZATION_PERCENT: 10,
    ASINH_SLOPE: 0.4,
    ALIVE_THRESHOLD: 0.5,       // Survival threshold
    SOFT_MIN_K: 8,              // Focus on weak species

    // Learning parameters
    LEARNING_BALANCE: 1,        // Steps per learning update
    OPTIMIZER_TYPE: 'SGD',
    LEARNING_RATE: 0.0007,      // Learning rate
    SUN_LR_SCALE: 1.0,          // Multiplier on sun's effective LR (0 = freeze sun, 1 = match species)
    GROWTH_GATE_STEEPNESS: 20,  // Sigmoid steepness in getAliveMask; lower = softer growth gate

    // Diversity loss: penalizes uneven population (0 = off, higher = stronger)
    DIVERSITY_WEIGHT: 0.4,      // Entropy-based diversity bonus

    // Sun penalty: handicaps sun in competition near NCAs (0 = off)
    SUN_PENALTY: 0,

    // Species preservation: minimum occupancy threshold (0 = off)
    MIN_OCCUPANCY: 20,

    // Competition similarity: true = cosine similarity, false = dot product
    USE_COSINE_SIM: true,

    // Model parameters
    N_NCAS: 5,
    CELL_STATE_DIM: 32,
    CELL_HIDDEN_DIM: 2,
    HIDDEN_DIM: 32,
    N_HIDDEN_LAYERS: 4,
    KERNEL_SIZE: 3,
    MODEL_DROPOUT_P: 0.3,
    SUN_INIT_SCALE: 0.6,

    // Grid parameters
    GRID_W: 128,
    GRID_H: 128,
    N_SEEDS: 10,

    // Drawing state
    DRAW_TOOL: 'none',
    DRAW_SIZE: 2,

    // Competition mode: 'full' (O(N²)) or 'global' (O(N))
    COMPETITION_MODE: 'full',

    // Stochastic update: probability each cell updates (1.0 = always, lower = more temporal noise)
    STOCHASTIC_UPDATE_PROB: 0.56,

    // Local win-rate modulation: boost update probability based on local success
    USE_LOCAL_WIN_RATE: true,
    LOCAL_WIN_RATE_ALPHA: 0.95,    // EMA smoothing (higher = longer memory)
    LOCAL_WIN_RATE_STRENGTH: 0.5,  // How much win rate affects update prob (0-2)

    // Color scheme for species
    COLOR_SCHEME: 'Vibrant',

    // Sun mode: 'global' (single vector), 'spatial' (per-cell learnable),
    // 'seasonal' (time-conditioned INR with learnable Fourier features).
    SUN_MODE: 'global',
    // How often (in sim steps) to refresh the sun divergence heatmap.
    HEATMAP_UPDATE_EVERY: 50,
    // Back-compat: read-through derived from SUN_MODE. Some code paths still
    // treat "local sun" as "anything spatially varying" (spatial OR seasonal).
    get USE_LOCAL_SUN() { return this.SUN_MODE !== 'global'; },
    set USE_LOCAL_SUN(v) { this.SUN_MODE = v ? 'spatial' : 'global'; },

    // Show FPS counter
    SHOW_FPS: false,

    // Concentration mechanism: scale attack/defense by spatial softmax
    USE_CONCENTRATION: true,
    CONCENTRATION_TEMP: 0.4,      // Temperature for spatial softmax (lower = sharper focus)
    CONC_BASELINE: 0.0,           // Minimum concentration weight (prevents complete suppression)
    CONCENTRATION_WINDOW: 4,      // Local window size for softmax
};

/**
 * Updates derived CONFIG values from base parameters.
 * Call this after changing N_NCAS, CELL_STATE_DIM, etc.
 */
function updateDerivedConfig() {
    CONFIG.ALIVE_DIM = CONFIG.N_NCAS + 1;
    CONFIG.CELL_WO_ALIVE_DIM = CONFIG.CELL_STATE_DIM + CONFIG.CELL_HIDDEN_DIM;
    CONFIG.CELL_DIM = CONFIG.CELL_WO_ALIVE_DIM + CONFIG.ALIVE_DIM;
    CONFIG.PACKED_PIXELS = Math.ceil(CONFIG.CELL_DIM / 4);

    const halfState = Math.floor(CONFIG.CELL_STATE_DIM / 2);
    CONFIG.ATT_IDXS = Array.from({ length: halfState }, (_, i) => i);
    CONFIG.DEF_IDXS = Array.from({ length: halfState }, (_, i) => i + halfState);
    CONFIG.SUN_INIT_SCALE = GUI_STATE.SUN_INIT_SCALE;
}

// Initialize derived values
updateDerivedConfig();


// =============================================================================
// MODEL ARCHITECTURE
// =============================================================================

/**
 * Creates an Inverted Residual Block (MobileNetV2 style).
 * Efficient and expressive for texture synthesis.
 */
function createSkipperBlock(config) {
    const { N_NCAS, HIDDEN_DIM, KERNEL_SIZE, MODEL_DROPOUT_P } = config;
    const input_dim = N_NCAS * HIDDEN_DIM;
    const expansion_factor = 2;
    const expanded_dim = input_dim * expansion_factor;

    class InvertedResidual extends tf.layers.Layer {
        constructor() {
            super({});
            this.expand = tf.layers.conv2d({
                filters: expanded_dim,
                kernelSize: 1,
                useBias: false,
                padding: 'same'
            });
            this.bn1 = tf.layers.batchNormalization({ axis: -1 });
            this.depthwise = tf.layers.depthwiseConv2d({
                kernelSize: KERNEL_SIZE,
                padding: 'same',
                depthMultiplier: 1,
                useBias: false,
            });
            this.project = tf.layers.conv2d({
                filters: input_dim,
                kernelSize: 1,
                useBias: false,
                padding: 'same'
            });
            this.activation = tf.layers.activation({ activation: 'gelu' });
            this.dropout = tf.layers.dropout({ rate: MODEL_DROPOUT_P });
        }

        call(inputs) {
            const x = Array.isArray(inputs) ? inputs[0] : inputs;
            let out = this.expand.apply(x);
            out = this.activation.apply(out);
            out = this.depthwise.apply(out);
            out = this.activation.apply(out);
            out = this.project.apply(out);
            out = this.dropout.apply(out);

            return tf.add(x, out);
        }

        get trainableWeights() {
            let weights = [];
            for (const sublayer of [this.expand, this.depthwise, this.project]) {
                if (sublayer.trainableWeights) {
                    weights = weights.concat(sublayer.trainableWeights);
                }
            }
            return weights;
        }

        static get className() { return 'InvertedResidual'; }
    }

    tf.serialization.registerClass(InvertedResidual);
    return new InvertedResidual();
}

/**
 * Creates the Neural CA model as a custom object.
 *
 * Returns a plain object with:
 * - predict(input) — manual forward pass
 * - trainableWeights — getter collecting weights from all sub-layers
 * - dispose() — disposes all sub-layer weights
 */
function createModel(config) {
    const input_dim = config.ALIVE_VISIBLE ? config.CELL_DIM : config.CELL_WO_ALIVE_DIM;
    const output_dim = config.CELL_WO_ALIVE_DIM;
    const hidden_dim = config.N_NCAS * config.HIDDEN_DIM;

    // --- Build layers individually ---

    // Encoder
    const encoderConv = tf.layers.conv2d({
        inputShape: [null, null, input_dim],
        filters: hidden_dim,
        kernelSize: config.KERNEL_SIZE,
        padding: 'same',
        useBias: false,
        kernelInitializer: tf.initializers.varianceScaling({
            scale: 0.1, mode: 'fanAvg', distribution: 'truncatedNormal'
        })
    });
    const encoderActivation = tf.layers.activation({ activation: 'gelu' });
    const encoderDropout = config.MODEL_DROPOUT_P > 0
        ? tf.layers.dropout({ rate: config.MODEL_DROPOUT_P })
        : null;

    // Hidden blocks (InvertedResidual)
    const hiddenBlocks = [];
    for (let i = 0; i < config.N_HIDDEN_LAYERS; i++) {
        hiddenBlocks.push(createSkipperBlock(config));
    }

    // Decoder
    const decoderConv = tf.layers.conv2d({
        filters: config.N_NCAS * output_dim,
        kernelSize: 1,
        groups: config.N_NCAS,
        useBias: false,
        kernelInitializer: tf.initializers.varianceScaling({
            scale: 0.1, mode: 'fanAvg', distribution: 'truncatedNormal'
        })
    });
    const decoderActivation = tf.layers.activation({ activation: 'tanh' });

    // Build all layers by running a dummy forward pass
    tf.tidy(() => {
        const dummyInput = tf.zeros([1, 1, 1, input_dim]);
        let x = encoderConv.apply(dummyInput);
        x = encoderActivation.apply(x);
        if (encoderDropout) x = encoderDropout.apply(x);
        for (const block of hiddenBlocks) {
            x = block.call(x);
        }
        x = decoderConv.apply(x);
        x = decoderActivation.apply(x);
    });

    // --- Model object ---
    const model = {
        // Store references for predict and dispose
        _encoderConv: encoderConv,
        _encoderActivation: encoderActivation,
        _encoderDropout: encoderDropout,
        _hiddenBlocks: hiddenBlocks,
        _decoderConv: decoderConv,
        _decoderActivation: decoderActivation,
        _config: config,

        predict(input) {
            let x = this._encoderConv.apply(input);
            x = this._encoderActivation.apply(x);
            if (this._encoderDropout) x = this._encoderDropout.apply(x);

            for (let i = 0; i < this._hiddenBlocks.length; i++) {
                x = this._hiddenBlocks[i].call(x);
            }

            x = this._decoderConv.apply(x);
            x = this._decoderActivation.apply(x);
            return x;
        },

        get trainableWeights() {
            const allLayers = [
                this._encoderConv,
                ...(this._encoderDropout ? [this._encoderDropout] : []),
                ...this._hiddenBlocks,
                this._decoderConv,
            ];
            let weights = [];
            for (const layer of allLayers) {
                if (layer.trainableWeights) {
                    weights = weights.concat(layer.trainableWeights);
                }
            }
            return weights;
        },

        dispose() {
            const allLayers = [
                this._encoderConv,
                this._encoderActivation,
                ...(this._encoderDropout ? [this._encoderDropout] : []),
                ...this._hiddenBlocks,
                this._decoderConv,
                this._decoderActivation,
            ];
            for (const layer of allLayers) {
                try { if (layer.dispose) layer.dispose(); } catch (e) { /* already disposed or not built */ }
            }
        },
    };

    return model;
}

// =============================================================================
// GRID INITIALIZATION
// =============================================================================

/**
 * Creates the initial simulation grid with seeded NCAs.
 */
function createGrid(config) {
    return tf.tidy(() => {
        const { GRID_H, GRID_W, CELL_DIM, ALIVE_DIM, N_NCAS, N_SEEDS, CELL_WO_ALIVE_DIM } = config;
        const buf = tf.buffer([1, GRID_H, GRID_W, CELL_DIM]);

        // Initialize all cells as "Sun"
        for (let y = 0; y < GRID_H; y++) {
            for (let x = 0; x < GRID_W; x++) {
                buf.set(1.0, 0, y, x, 0);
            }
        }

        // Generate per-species diversity factors
        // Each species gets a different scale (0.7 to 1.3) and direction bias
        const speciesScales = [];
        const speciesBiases = [];
        for (let nca_i = 0; nca_i < N_NCAS; nca_i++) {
            // Scale varies from 0.7 to 1.3 based on species index + randomness
            const baseScale = 0.7 + (0.6 * (nca_i + 0.5) / N_NCAS);
            const randomScale = baseScale * (0.9 + 0.2 * seededRandom());
            speciesScales.push(randomScale);

            // Each species gets a random directional bias vector
            const bias = tf.randomNormal([CELL_WO_ALIVE_DIM]).mul(0.3);
            speciesBiases.push(bias.dataSync());
            bias.dispose();
        }

        // Seed each NCA with species-specific initialization
        for (let nca_i = 0; nca_i < N_NCAS; nca_i++) {
            const scale = speciesScales[nca_i];
            const bias = speciesBiases[nca_i];

            for (let seed_j = 0; seed_j < N_SEEDS; seed_j++) {
                const x = Math.floor(seededRandom() * GRID_W);
                const y = Math.floor(seededRandom() * GRID_H);

                buf.set(0.0, 0, y, x, 0);
                buf.set(1.0, 0, y, x, nca_i + 1);

                // Random state + species-specific bias, then normalize and scale
                const randomState = tf.randomNormal([CELL_WO_ALIVE_DIM]);
                const biasedState = randomState.add(tf.tensor1d(bias));
                const normState = biasedState.div(biasedState.norm().add(EPSILON));
                const scaledState = normState.mul(scale);
                const stateData = scaledState.dataSync();

                for (let c = 0; c < CELL_WO_ALIVE_DIM; c++) {
                    buf.set(stateData[c], 0, y, x, ALIVE_DIM + c);
                }

                scaledState.dispose();
                normState.dispose();
                biasedState.dispose();
                randomState.dispose();
            }
        }

        return tf.keep(buf.toTensor());
    });
}

/**
 * Creates the sun's update vector with trainable parameters.
 * Returns { sun_base, sun_params, inr } where inr is non-null only in seasonal mode.
 */
function createSunUpdate(config) {
    const mode = GUI_STATE.SUN_MODE;
    const { sun_base, sun_params } = tf.tidy(() => {
        const { CELL_WO_ALIVE_DIM, CELL_STATE_DIM, CELL_HIDDEN_DIM } = config;
        let sun_vec = tf.randomNormal([CELL_WO_ALIVE_DIM]).mul(config.SUN_INIT_SCALE);

        // Zero out hidden channels
        if (CELL_HIDDEN_DIM > 0) {
            const hidden_indices = Array.from(
                { length: CELL_HIDDEN_DIM },
                (_, i) => [i + CELL_STATE_DIM]
            );
            const updates = tf.zeros([CELL_HIDDEN_DIM]);
            sun_vec = tf.tensorScatterUpdate(sun_vec, hidden_indices, updates);
        }

        sun_vec = sun_vec.div(sun_vec.norm().add(EPSILON));

        // Sun tensors are stored as 4D to avoid TF.js WebGL 5D shader issues.
        // Global: [1, 1, 1, CWA], Spatial/Seasonal: [1, H, W, CWA].
        if (mode !== 'global') {
            const H = config.GRID_H, W = config.GRID_W;
            const sun_base = tf.keep(
                sun_vec.reshape([1, 1, 1, CELL_WO_ALIVE_DIM]).tile([1, H, W, 1])
            );
            const sun_params = tf.keep(tf.variable(tf.zeros([1, H, W, CELL_WO_ALIVE_DIM]), true));
            return { sun_base, sun_params };
        } else {
            const sun_base = tf.keep(sun_vec.reshape([1, 1, 1, CELL_WO_ALIVE_DIM]));
            const sun_params = tf.keep(tf.variable(tf.zeros([1, 1, 1, CELL_WO_ALIVE_DIM]), true));
            return { sun_base, sun_params };
        }
    });
    const inr = (mode === 'seasonal') ? createInrParams(config) : null;
    return { sun_base, sun_params, inr };
}

// =============================================================================
// SEASONAL SUN — INR(x, y, t) with learnable time periods
// =============================================================================
//
// In seasonal mode the INR FULLY replaces sun_base+sun_params:
//   sun(x,y,t) = tanh( INR(x,y,t) )
//
// INR = MLP( [γ_xy(x,y), γ_t(t)] )
//   γ_xy uses Learnable Fourier Features (Li et al., NeurIPS 2021):
//     γ_xy = (1/√(D/2)) · [cos(p·W_r), sin(p·W_r)] with W_r ∈ R^{2×D/2} trainable.
//   γ_t uses K_T=8 *learnable* log_period values. Phase is maintained as a
//   non-trainable accumulator variable so ∂phase/∂log_period = O(1) rather
//   than O(t) — i.e., gradients stay bounded no matter how long the sim runs.
//
// MLP: 40 → 32 → 32 → CWA, ReLU hidden, linear output.
// The output layer is ZERO-initialized so Spatial→Seasonal promotion is
// bit-identical at the step of promotion (lossless) up to the bout reseed.
//
// Training: INR params live in their own var list and are updated via manual
// SGD scaled by SUN_LR_SCALE (same pattern as sun_params). This keeps the
// slider semantics "0 = fully frozen sun" exact.

const INR_GAMMA_XY_DIM = 24;           // output dim D of the LFF block
const INR_LFF_SIGMA = 5.0;             // init stddev of W_r — coords in [-1,1]
const INR_K_T  = 8;                    // temporal Fourier bands (learnable periods)
const INR_GAMMA_T_DIM  = 2 * INR_K_T;  // 16
const INR_INPUT_DIM = INR_GAMMA_XY_DIM + INR_GAMMA_T_DIM; // 40
const INR_HIDDEN = 32;
const INR_LOG_PERIOD_MIN = Math.log(25);
const INR_LOG_PERIOD_MAX = Math.log(10000);

/**
 * Heuristic Xavier-like init for a Dense layer.
 */
function _inrInitDense(inDim, outDim, scale = 1.0) {
    const limit = scale * Math.sqrt(6 / (inDim + outDim));
    return tf.keep(tf.variable(tf.randomUniform([inDim, outDim], -limit, limit), true));
}

/**
 * Allocates INR parameters (MLP + learnable periods + phase accumulator).
 * Final linear layer is zero-initialized so INR(x,y,t) ≡ 0 at promotion.
 */
function createInrParams(config) {
    const CWA = config.CELL_WO_ALIVE_DIM;

    // Learnable log-period, initialized log-uniform over [25, 1000] sim steps.
    const logPeriodInit = new Float32Array(INR_K_T);
    const lo = Math.log(25), hi = Math.log(1000);
    for (let k = 0; k < INR_K_T; k++) {
        logPeriodInit[k] = lo + (hi - lo) * (k / Math.max(1, INR_K_T - 1));
    }
    const inr_log_period = tf.keep(tf.variable(tf.tensor1d(logPeriodInit), true));
    // Phase accumulator — updated each sim step outside the tape, and
    // excluded from the optimizer varList (so effectively non-trainable).
    const inr_phase = tf.keep(tf.variable(tf.zeros([INR_K_T]), true));

    // Spatial LFF projection: W_r ∈ R^{2 × D/2}, init N(0, σ²).
    const inr_W_r = tf.keep(tf.variable(
        tf.randomNormal([2, INR_GAMMA_XY_DIM / 2], 0, INR_LFF_SIGMA), true
    ));

    // First layer split into spatial and temporal branches (algebraic identity:
    // [γ_xy; γ_t] @ W1 = γ_xy @ W1_xy + γ_t @ W1_t). Avoids tile+concat over
    // [1,H,W,40] textures each step.
    const inr_W1_xy = _inrInitDense(INR_GAMMA_XY_DIM, INR_HIDDEN);
    const inr_W1_t  = _inrInitDense(INR_GAMMA_T_DIM,  INR_HIDDEN);
    const inr_b1 = tf.keep(tf.variable(tf.zeros([INR_HIDDEN]), true));
    const inr_W2 = _inrInitDense(INR_HIDDEN, INR_HIDDEN);
    const inr_b2 = tf.keep(tf.variable(tf.zeros([INR_HIDDEN]), true));
    // Output layer zero-init; bout may be reassigned at promotion time to the
    // spatial mean of the previous sun (see setSunMode promote path).
    const inr_Wout = tf.keep(tf.variable(tf.zeros([INR_HIDDEN, CWA]), true));
    const inr_bout = tf.keep(tf.variable(tf.zeros([CWA]), true));

    return {
        inr_log_period, inr_phase,
        inr_W_r,
        inr_W1_xy, inr_W1_t,
        inr_b1, inr_W2, inr_b2, inr_Wout, inr_bout,
    };
}

function disposeInr(inr) {
    if (!inr) return;
    for (const k of Object.keys(inr)) {
        if (inr[k] && !inr[k].isDisposed) inr[k].dispose();
    }
}

/**
 * List the trainable INR variables (for inclusion in optimizer var list).
 * Excludes inr_phase, which is maintained outside the tape as a side-effect.
 */
function inrTrainableVars(inr) {
    if (!inr) return [];
    return [
        inr.inr_log_period,
        inr.inr_W_r,
        inr.inr_W1_xy, inr.inr_W1_t, inr.inr_b1,
        inr.inr_W2, inr.inr_b2,
        inr.inr_Wout, inr.inr_bout,
    ];
}

/**
 * Build the normalized coordinate grid ∈ [1, H, W, 2] with x,y ∈ [-1, 1].
 */
function buildCoordsCache(H, W) {
    return tf.tidy(() => {
        const xs = tf.linspace(-1, 1, W).reshape([1, 1, W, 1]).tile([1, H, 1, 1]);
        const ys = tf.linspace(-1, 1, H).reshape([1, H, 1, 1]).tile([1, 1, W, 1]);
        return tf.keep(tf.concat([xs, ys], 3)); // [1, H, W, 2]
    });
}

/**
 * Update the phase accumulator in place: phase ← (phase + 2π·exp(-log_period)) mod 2π.
 * Runs outside the gradient tape so ∂phase/∂log_period stays bounded across long sims.
 */
function stepInrPhase(inr, dt = 1) {
    if (!inr) return;
    tf.tidy(() => {
        const TWO_PI = 2 * Math.PI;
        const omega = tf.exp(inr.inr_log_period.neg()).mul(TWO_PI * dt);
        const newPhase = inr.inr_phase.add(omega);
        const wrapped = newPhase.sub(newPhase.div(TWO_PI).floor().mul(TWO_PI));
        inr.inr_phase.assign(wrapped);
    });
}

/**
 * Evaluate the INR MLP on the full grid. Returns [1, H, W, CWA].
 * coords: [1, H, W, 2] — normalized (x, y) grid cache.
 *
 * Phase is a stop-gradient accumulator: the tape sees phase as a constant
 * plus the current-step `omega·dt` increment (dt=1), bounding period grads.
 */
function evaluateInr(inr, coords, config) {
    const H = coords.shape[1], W = coords.shape[2];
    const N = H * W;
    const CWA = config.CELL_WO_ALIVE_DIM;
    const D2 = INR_GAMMA_XY_DIM / 2; // 12
    const INV_SQRT_D2 = 1 / Math.sqrt(D2);
    return tf.tidy(() => {
        const omega = tf.exp(inr.inr_log_period.neg()).mul(2 * Math.PI);
        const phase_eff = inr.inr_phase.add(omega); // dt=1
        const gamma_t_1d = tf.concat(
            [tf.sin(phase_eff), tf.cos(phase_eff)], 0
        ).reshape([1, INR_GAMMA_T_DIM]); // [1, 16]

        const coords_flat = coords.reshape([N, 2]);                       // [N, 2]
        const proj = tf.matMul(coords_flat, inr.inr_W_r);                  // [N, 12]
        const gamma_xy_flat = tf.concat(
            [tf.cos(proj), tf.sin(proj)], 1
        ).mul(INV_SQRT_D2);                                                // [N, 24]

        const preact_xy = tf.matMul(gamma_xy_flat, inr.inr_W1_xy);          // [N, HIDDEN]
        const preact_t  = tf.matMul(gamma_t_1d,    inr.inr_W1_t);            // [1, HIDDEN]
        const h1 = tf.relu(preact_xy.add(preact_t).add(inr.inr_b1));         // [N, HIDDEN]

        const h2 = tf.fused.matMul({
            a: h1, b: inr.inr_W2, bias: inr.inr_b2, activation: 'relu'
        });
        const out = tf.fused.matMul({
            a: h2, b: inr.inr_Wout, bias: inr.inr_bout
        }); // [N, CWA]

        return out.reshape([1, H, W, CWA]);
    });
}

/**
 * Read the learned periods as a JS number array (for UI display).
 */
async function getInrPeriods(inr) {
    if (!inr) return [];
    const data = await inr.inr_log_period.data();
    return Array.from(data, v => Math.exp(v));
}

// =============================================================================
// PARAMETER HELPERS
// =============================================================================

/**
 * Gets the current softmax temperature from GUI.
 */
function getEffectiveSoftmaxTemp() {
    return GUI_STATE.SOFTMAX_TEMP;
}

/**
 * Gets the current alive threshold from GUI.
 */
function getEffectiveAliveThreshold() {
    return GUI_STATE.ALIVE_THRESHOLD;
}

/**
 * Gets the sun penalty from GUI.
 * Applied spatially near NCAs to give them a competitive advantage.
 */
function getSunPenalty() {
    return GUI_STATE.SUN_PENALTY;
}

// =============================================================================
// COMPETITION MECHANICS
// =============================================================================

// Steepness for soft threshold (higher = more like hard threshold)
const SOFT_THRESHOLD_STEEPNESS = 20.0;  // Balanced: not too soft, not too harsh

/**
 * Soft threshold using sigmoid - differentiable alternative to greater().
 * softThreshold(x, t) ≈ 1 if x > t, ≈ 0 if x < t
 */
function softThreshold(x, threshold, steepness = SOFT_THRESHOLD_STEEPNESS) {
    return tf.sigmoid(x.sub(threshold).mul(steepness));
}

/**
 * L2 normalizes a tensor along a specified axis.
 */
function l2Normalize(tensor, axis) {
    return tf.tidy(() => {
        const squared = tensor.square();
        const summed = squared.sum(axis, true);
        const normed = summed.sqrt().add(EPSILON);
        return tensor.div(normed);
    });
}

/**
 * Computes local softmax concentration weights for attack/defense scaling.
 * Uses convolution-based local softmax to avoid global dilution.
 *
 * @param {tf.Tensor} attacks - Attack vectors [1, H, W, N+1, att_dim]
 * @param {tf.Tensor} defenses - Defense vectors [1, H, W, N+1, def_dim]
 * @returns {tf.Tensor} Concentration weights [1, H, W, N+1, 1]
 */
function computeLocalConcentration(attacks, defenses) {
    return tf.tidy(() => {
        const temp = GUI_STATE.CONCENTRATION_TEMP;
        const baseline = GUI_STATE.CONC_BASELINE;
        const windowSize = GUI_STATE.CONCENTRATION_WINDOW;

        // Compute "energy" at each cell as L2 norm of attack + defense
        // Shape: [1, H, W, N+1, att_dim] -> [1, H, W, N+1]
        const attackNorm = attacks.square().sum(4).sqrt();
        const defenseNorm = defenses.square().sum(4).sqrt();
        const energy = attackNorm.add(defenseNorm);  // [1, H, W, N+1]

        // Apply temperature scaling and exp for softmax
        const scaledEnergy = energy.div(temp);
        const expEnergy = tf.exp(scaledEnergy);  // [1, H, W, N+1]

        // Get dimensions
        const shape = expEnergy.shape;
        const h = shape[1];
        const w = shape[2];
        const numSpecies = shape[3];  // N+1

        // Create ones kernel for local sum (windowSize x windowSize)
        const onesKernel = tf.ones([windowSize, windowSize, 1, 1]);

        // Process each species channel separately to compute local sums
        // We need to do depthwise-style convolution per species
        const expEnergyChannels = tf.unstack(expEnergy, 3);  // List of [1, H, W] tensors

        const localWeightsList = expEnergyChannels.map(channelEnergy => {
            // Add channel dim: [1, H, W] -> [1, H, W, 1]
            const channelWithDim = channelEnergy.expandDims(3);

            // Compute local sum using convolution
            const localSum = tf.conv2d(channelWithDim, onesKernel, 1, 'same');

            // Local softmax: exp(x) / sum_local(exp(x))
            const localWeight = channelWithDim.div(localSum.add(EPSILON));

            // Apply baseline: final = baseline + (1 - baseline) * weight
            const finalWeight = tf.scalar(baseline).add(
                tf.scalar(1 - baseline).mul(localWeight)
            );

            return finalWeight.squeeze([3]);  // Back to [1, H, W]
        });

        // Stack back to [1, H, W, N+1]
        const localWeights = tf.stack(localWeightsList, 3);

        // Add dimension for broadcasting with attack/defense vectors
        // [1, H, W, N+1] -> [1, H, W, N+1, 1]
        return localWeights.expandDims(4);
    });
}

/**
 * Gets the aliveness mask based on max-pooled alive channels.
 * Uses soft threshold for differentiability.
 * Uses effective threshold during warmup for gentler competition.
 */
function getAliveMask(grid, config) {
    return tf.tidy(() => {
        const { ALIVE_DIM } = config;
        const threshold = getEffectiveAliveThreshold();

        const alive_channels = grid.slice([0, 0, 0, 0], [-1, -1, -1, ALIVE_DIM]);
        const alive_pooled = tf.maxPool(alive_channels, ALIVE_POOL_SIZE, 1, 'same');

        // Use soft threshold instead of hard greater(); steepness is GUI-tunable
        // (softer = wider transition zone, less likely to cascade-collapse at high thresholds)
        const float_mask = softThreshold(alive_pooled, threshold, GUI_STATE.GROWTH_GATE_STEEPNESS);

        const sun_mask_float = float_mask.slice([0, 0, 0, 0], [-1, -1, -1, 1]);
        const nca_masks_float = float_mask.slice([0, 0, 0, 1]);

        // Sun is always alive
        const sun_mask_forced_one = tf.onesLike(sun_mask_float);
        const final_mask_float = tf.concat([sun_mask_forced_one, nca_masks_float], 3);

        return final_mask_float;
    });
}

/**
 * Runs competition with FULL O(N²) matrix (original method).
 * All operations are differentiable for gradient computation.
 */
function runCompetitionFull(grid, all_updates, wallGrid, config) {
    return tf.tidy(() => {
        perfStart('competition-total');
        const { ALIVE_DIM, ATT_IDXS, DEF_IDXS, N_NCAS } = config;
        const SOFTMAX_TEMP = getEffectiveSoftmaxTemp();
        const ALIVE_THRESHOLD = getEffectiveAliveThreshold();

        perfStart('comp-alive-mask');
        const growth_mask = getAliveMask(grid, config);
        perfEnd('comp-alive-mask');

        // Extract attack and defense vectors
        // Note: Using gather instead of slice due to TF.js WebGL 5D tensor limitations
        perfStart('comp-gather');
        let all_attacks = all_updates.gather(tf.tensor1d(ATT_IDXS, 'int32'), 4);
        let all_defenses = all_updates.gather(tf.tensor1d(DEF_IDXS, 'int32'), 4);
        perfEnd('comp-gather');

        // --- CONCENTRATION SCALING ---
        // Apply local softmax concentration to focus energy spatially
        if (GUI_STATE.USE_CONCENTRATION) {
            perfStart('comp-concentration');
            const concWeights = computeLocalConcentration(all_attacks, all_defenses);
            // Scale attack/defense by concentration weights
            // concWeights shape: [1, H, W, N+1, 1] broadcasts with [1, H, W, N+1, dim]
            all_attacks = all_attacks.mul(concWeights);
            all_defenses = all_defenses.mul(concWeights);
            perfEnd('comp-concentration');
        }

        // Normalize for cosine similarity, or use raw dot product
        perfStart('comp-normalize');
        const norm_att = GUI_STATE.USE_COSINE_SIM ? l2Normalize(all_attacks, 4) : all_attacks;
        const norm_def = GUI_STATE.USE_COSINE_SIM ? l2Normalize(all_defenses, 4) : all_defenses;
        perfEnd('comp-normalize');

        // O(N) competition scoring - mathematically equivalent to O(N²) matmul
        // Uses distributive property: Σ_j (att_i · def_j) = att_i · (Σ_j def_j)
        // This avoids creating the [1, H, W, N+1, N+1] similarity matrix
        perfStart('comp-scoring');
        const def_sum = norm_def.sum(3);  // Sum defenders: [1, H, W, N+1, def_dim] -> [1, H, W, def_dim]
        const def_sum_expanded = def_sum.expandDims(3);  // [1, H, W, 1, def_dim] for broadcasting
        let raw_scores = norm_att.mul(def_sum_expanded).sum(4);  // [1, H, W, N+1]
        perfEnd('comp-scoring');

        // --- PRESENCE GATING (prevents species from "flashing" into disconnected locations) ---
        perfStart('comp-presence-gating');
        // Species must have meaningful presence nearby to participate in competition
        // Mask out walls before dilation to prevent presence leaking through walls
        const not_wall = wallGrid.less(0.5).cast('float32');
        const alive_channels = grid.slice([0, 0, 0, 0], [-1, -1, -1, ALIVE_DIM]);
        const alive_masked = alive_channels.mul(not_wall);
        const dilated_alive = tf.maxPool(alive_masked, ALIVE_POOL_SIZE, 1, 'same');
        const has_presence = dilated_alive.greater(MIN_PRESENCE_FOR_GROWTH).cast('float32');

        // Sun always present (index 0)
        const sun_presence = tf.onesLike(has_presence.slice([0, 0, 0, 0], [-1, -1, -1, 1]));
        const nca_presence = has_presence.slice([0, 0, 0, 1]);
        const full_presence_mask = tf.concat([sun_presence, nca_presence], 3);

        // Apply large negative penalty where species has no presence
        const ABSENT_PENALTY = 1e9;
        const presence_penalty = tf.sub(1, full_presence_mask).mul(ABSENT_PENALTY);
        raw_scores = raw_scores.sub(presence_penalty);
        perfEnd('comp-presence-gating');

        // --- SUN PENALTY ---
        perfStart('comp-penalties');
        // Apply sun penalty spatially near NCAs (makes it harder for sun to compete)
        const sunPenalty = getSunPenalty();
        if (sunPenalty > 0) {
            // Get NCA alivenesses, mask walls, then dilate for growth frontier
            // Use ALIVE_POOL_SIZE (3) instead of 5 so 1-pixel walls are effective
            const nca_alivenesses_penalty = grid.slice([0, 0, 0, 1], [-1, -1, -1, N_NCAS]);
            const nca_alive_masked_sp = nca_alivenesses_penalty.mul(not_wall);
            const dilated_nca = tf.maxPool(nca_alive_masked_sp, ALIVE_POOL_SIZE, 1, 'same');
            const max_nca_presence = dilated_nca.max(3, true);
            const nca_frontier_mask = tf.sigmoid(max_nca_presence.sub(0.01).mul(20.0));

            const spatial_penalty = nca_frontier_mask.mul(sunPenalty);
            const zeros_for_ncas = tf.zeros([1, grid.shape[1], grid.shape[2], N_NCAS]);
            const full_penalty = tf.concat([spatial_penalty, zeros_for_ncas], 3);
            raw_scores = raw_scores.sub(full_penalty);
        }

        // --- DIVERSITY BOOST FOR MINORITY SPECIES ---
        // Species with lower population get a score bonus during competition
        // This directly helps underrepresented species survive
        const diversityWeight = GUI_STATE.DIVERSITY_WEIGHT;
        if (diversityWeight > 0 && N_NCAS > 1) {
            // Get global population for each NCA (skip sun at index 0)
            const nca_alivenesses = grid.slice([0, 0, 0, 1], [-1, -1, -1, N_NCAS]);
            const total_per_nca = nca_alivenesses.sum([1, 2]);  // [1, N_NCAS]
            const total_all = total_per_nca.sum().add(EPSILON);
            const pop_fractions = total_per_nca.div(total_all);  // [1, N_NCAS]

            // Uniform distribution would be 1/N_NCAS for each
            const uniform_frac = 1.0 / N_NCAS;

            // Boost = how much BELOW uniform they are (clamped to positive)
            // More underrepresented = bigger boost
            const deficit = tf.scalar(uniform_frac).sub(pop_fractions);
            const boost = tf.relu(deficit);

            // Scale boost by diversity weight (10x multiplier for noticeable effect)
            const scaled_boost = boost.mul(diversityWeight * 10.0);

            // Prepend 0 for sun (no boost for sun)
            const sun_boost = tf.zeros([1, 1]);
            const full_boost = tf.concat([sun_boost, scaled_boost], 1);  // [1, ALIVE_DIM]

            // Add to raw scores (broadcast to [1, H, W, ALIVE_DIM])
            raw_scores = raw_scores.add(full_boost.reshape([1, 1, 1, ALIVE_DIM]));
        }

        // --- PRESERVATION BOOST FOR ENDANGERED SPECIES ---
        const minOccupancy = GUI_STATE.MIN_OCCUPANCY;
        if (minOccupancy > 0 && N_NCAS > 1) {
            const PRESERVATION_STRENGTH = 5.0;

            // Get NCA populations (absolute cell counts)
            const nca_alive_pres = grid.slice([0, 0, 0, 1], [-1, -1, -1, N_NCAS]);
            const cell_count_per_nca = nca_alive_pres.sum([1, 2]);  // [1, N_NCAS]

            // Compute deficit: how far below threshold in cells (0 if above)
            const pres_deficit = tf.relu(tf.scalar(minOccupancy).sub(cell_count_per_nca));
            const normalized_deficit = pres_deficit.div(minOccupancy + EPSILON);

            // sqrt scaling: gentle at first, stronger as extinction approaches
            const pres_boost = normalized_deficit.sqrt().mul(PRESERVATION_STRENGTH);

            // Apply boost only at species frontiers (where they can grow)
            // Mask walls before dilation to prevent boost leaking through walls
            const nca_alive_pres_masked = nca_alive_pres.mul(not_wall);
            const dilated_nca_pres = tf.maxPool(nca_alive_pres_masked, ALIVE_POOL_SIZE, 1, 'same');
            const frontier_mask = dilated_nca_pres.greater(MIN_PRESENCE_FOR_GROWTH).cast('float32');

            // Spatial boost: [1, N_NCAS] * [1, H, W, N_NCAS] -> [1, H, W, N_NCAS]
            const spatial_boost = pres_boost.reshape([1, 1, 1, N_NCAS]).mul(frontier_mask);

            // Add sun channel (zero boost)
            const sun_pres_boost = tf.zeros([1, grid.shape[1], grid.shape[2], 1]);
            const full_pres_boost = tf.concat([sun_pres_boost, spatial_boost], 3);

            raw_scores = raw_scores.add(full_pres_boost);
        }

        // Wall penalty - walls block all species from growing there
        // wallGrid shape: [1, H, W, 1], need to broadcast to [1, H, W, N+1]
        const wall_mask = softThreshold(wallGrid, 0.5, 100.0);  // Very sharp for walls
        // Set sun score to 1 in walls (sun fills walls), NCAs get huge penalty
        const wall_penalty = wall_mask.mul(1e9);  // Large penalty where wall=1
        const masked_scores = raw_scores.sub(wall_penalty);
        perfEnd('comp-penalties');

        // Softmax temperature
        perfStart('comp-softmax');
        const competition_weights = tf.softmax(masked_scores.div(SOFTMAX_TEMP), 3);
        perfEnd('comp-softmax');

        // Compute final update
        perfStart('comp-final-update');
        const weighted_updates = all_updates.mul(competition_weights.expandDims(4));
        const final_update = weighted_updates.sum(3);

        // Update hidden state
        const current_hidden = grid.slice([0, 0, 0, ALIVE_DIM]);
        const new_hidden = current_hidden.add(final_update.mul(0.5)).tanh();

        // Set alphas
        const new_alphas = competition_weights.mul(growth_mask);

        // Reconstruct grid
        let new_grid = tf.concat([new_alphas, new_hidden], 3);

        // Survival check - handles soft competition where weights are distributed
        // Cell survives if: sun dominates (>threshold) OR NCAs have meaningful presence
        const sun_alpha = new_alphas.slice([0, 0, 0, 0], [-1, -1, -1, 1]);
        const nca_alphas = new_alphas.slice([0, 0, 0, 1], [-1, -1, -1, -1]);
        const nca_total = nca_alphas.sum(3, true);

        // Sun-dominated: sun needs high alpha
        const sun_survives = softThreshold(sun_alpha, ALIVE_THRESHOLD);
        // NCA-present: combined NCA presence > 0.15 (lenient for distributed weights)
        const nca_survives = softThreshold(nca_total, 0.15);
        const is_alive = tf.maximum(sun_survives, nca_survives);

        perfEnd('comp-final-update');
        perfEnd('competition-total');
        return new_grid.mul(is_alive);
    });
}

/**
 * Runs competition with GLOBAL DEFENSE O(N) approximation.
 *
 * Instead of comparing each attacker against each defender (N² comparisons),
 * we compute a population-weighted global defense vector and compare each
 * attacker against this single vector (N comparisons).
 *
 * This is a mean-field approximation that significantly reduces computation
 * while preserving the essential competitive dynamics.
 * All operations are differentiable for gradient computation.
 */
function runCompetitionGlobal(grid, all_updates, wallGrid, config) {
    return tf.tidy(() => {
        const { ALIVE_DIM, ATT_IDXS, DEF_IDXS, N_NCAS } = config;
        const SOFTMAX_TEMP = getEffectiveSoftmaxTemp();
        const ALIVE_THRESHOLD = getEffectiveAliveThreshold();

        const growth_mask = getAliveMask(grid, config);

        // Extract attack and defense vectors
        // all_updates shape: [1, H, W, N+1, CELL_WO_ALIVE_DIM]
        // Note: Using gather instead of slice due to TF.js WebGL 5D tensor limitations
        let all_attacks = all_updates.gather(tf.tensor1d(ATT_IDXS, 'int32'), 4);
        let all_defenses = all_updates.gather(tf.tensor1d(DEF_IDXS, 'int32'), 4);

        // --- CONCENTRATION SCALING ---
        // Apply local softmax concentration to focus energy spatially
        if (GUI_STATE.USE_CONCENTRATION) {
            const concWeights = computeLocalConcentration(all_attacks, all_defenses);
            // Scale attack/defense by concentration weights
            all_attacks = all_attacks.mul(concWeights);
            all_defenses = all_defenses.mul(concWeights);
        }

        // Normalize for cosine similarity, or use raw dot product
        const norm_att = GUI_STATE.USE_COSINE_SIM ? l2Normalize(all_attacks, 4) : all_attacks;
        const norm_def = GUI_STATE.USE_COSINE_SIM ? l2Normalize(all_defenses, 4) : all_defenses;

        // --- GLOBAL DEFENSE APPROXIMATION ---
        // Get current population weights from aliveness
        const alive_channels = grid.slice([0, 0, 0, 0], [-1, -1, -1, ALIVE_DIM]);
        // Shape: [1, H, W, N+1]

        // Compute local population density for weighting
        // Use softmax to get smooth weights
        const pop_weights = tf.softmax(alive_channels.mul(2.0), 3);
        // Shape: [1, H, W, N+1]

        // Expand weights for broadcasting with defense vectors
        const pop_weights_expanded = pop_weights.expandDims(4);
        // Shape: [1, H, W, N+1, 1]

        // Compute weighted average defense vector
        // norm_def shape: [1, H, W, N+1, def_dim]
        const weighted_def = norm_def.mul(pop_weights_expanded);
        const global_defense = weighted_def.sum(3, true);
        // Shape: [1, H, W, 1, def_dim]

        // Normalize the global defense
        const global_defense_norm = l2Normalize(global_defense, 4);

        // --- O(N) SCORING ---
        // Each attacker competes against the global defense
        // norm_att shape: [1, H, W, N+1, att_dim]
        // global_defense_norm shape: [1, H, W, 1, def_dim]

        // Compute similarity using automatic broadcasting (no tile needed)
        // TF.js will broadcast global_defense_norm along axis 3 automatically
        const similarity = norm_att.mul(global_defense_norm).sum(4);
        // Shape: [1, H, W, N+1]

        // Add self-defense bonus (species that ARE the global defense get a bonus)
        // This preserves some of the winner-take-all dynamics
        const self_bonus = pop_weights.mul(0.5);
        let raw_scores = similarity.add(self_bonus);

        // --- PRESENCE GATING (prevents species from "flashing" into disconnected locations) ---
        // Species must have meaningful presence nearby to participate in competition
        // Mask out walls before dilation to prevent presence leaking through walls
        const not_wall = wallGrid.less(0.5).cast('float32');
        const alive_masked = alive_channels.mul(not_wall);
        const dilated_alive = tf.maxPool(alive_masked, ALIVE_POOL_SIZE, 1, 'same');
        const has_presence = dilated_alive.greater(MIN_PRESENCE_FOR_GROWTH).cast('float32');

        // Sun always present (index 0)
        const sun_presence = tf.onesLike(has_presence.slice([0, 0, 0, 0], [-1, -1, -1, 1]));
        const nca_presence = has_presence.slice([0, 0, 0, 1]);
        const full_presence_mask = tf.concat([sun_presence, nca_presence], 3);

        // Apply large negative penalty where species has no presence
        const ABSENT_PENALTY = 1e9;
        const presence_penalty = tf.sub(1, full_presence_mask).mul(ABSENT_PENALTY);
        raw_scores = raw_scores.sub(presence_penalty);

        // --- SUN PENALTY ---
        // Apply sun penalty spatially near NCAs (makes it harder for sun to compete)
        const sunPenalty = getSunPenalty();
        if (sunPenalty > 0) {
            // Get NCA alivenesses, mask walls, then dilate for growth frontier
            // Use ALIVE_POOL_SIZE (3) instead of 5 so 1-pixel walls are effective
            const nca_alivenesses_penalty = grid.slice([0, 0, 0, 1], [-1, -1, -1, N_NCAS]);
            const nca_alive_masked_sp = nca_alivenesses_penalty.mul(not_wall);
            const dilated_nca = tf.maxPool(nca_alive_masked_sp, ALIVE_POOL_SIZE, 1, 'same');
            const max_nca_presence = dilated_nca.max(3, true);
            const nca_frontier_mask = tf.sigmoid(max_nca_presence.sub(0.01).mul(20.0));

            const spatial_penalty = nca_frontier_mask.mul(sunPenalty);
            const zeros_for_ncas = tf.zeros([1, grid.shape[1], grid.shape[2], N_NCAS]);
            const full_penalty = tf.concat([spatial_penalty, zeros_for_ncas], 3);
            raw_scores = raw_scores.sub(full_penalty);
        }

        // --- DIVERSITY BOOST FOR MINORITY SPECIES ---
        // Species with lower population get a score bonus during competition
        // This directly helps underrepresented species survive
        const diversityWeight = GUI_STATE.DIVERSITY_WEIGHT;
        if (diversityWeight > 0 && N_NCAS > 1) {
            // Get global population for each NCA (skip sun at index 0)
            const nca_alivenesses = grid.slice([0, 0, 0, 1], [-1, -1, -1, N_NCAS]);
            const total_per_nca = nca_alivenesses.sum([1, 2]);  // [1, N_NCAS]
            const total_all = total_per_nca.sum().add(EPSILON);
            const pop_fractions_div = total_per_nca.div(total_all);  // [1, N_NCAS]

            // Uniform distribution would be 1/N_NCAS for each
            const uniform_frac_val = 1.0 / N_NCAS;

            // Boost = how much BELOW uniform they are (clamped to positive)
            const deficit_val = tf.scalar(uniform_frac_val).sub(pop_fractions_div);
            const boost_val = tf.relu(deficit_val);

            // Scale boost by diversity weight (10x multiplier for noticeable effect)
            const scaled_boost_val = boost_val.mul(diversityWeight * 10.0);

            // Prepend 0 for sun (no boost for sun)
            const sun_boost_val = tf.zeros([1, 1]);
            const full_boost_val = tf.concat([sun_boost_val, scaled_boost_val], 1);

            // Add to raw scores (broadcast to [1, H, W, ALIVE_DIM])
            raw_scores = raw_scores.add(full_boost_val.reshape([1, 1, 1, ALIVE_DIM]));
        }

        // --- PRESERVATION BOOST FOR ENDANGERED SPECIES ---
        const minOccupancy = GUI_STATE.MIN_OCCUPANCY;
        if (minOccupancy > 0 && N_NCAS > 1) {
            const PRESERVATION_STRENGTH = 5.0;

            // Get NCA populations (absolute cell counts)
            const nca_alive_pres = grid.slice([0, 0, 0, 1], [-1, -1, -1, N_NCAS]);
            const cell_count_per_nca = nca_alive_pres.sum([1, 2]);  // [1, N_NCAS]

            // Compute deficit: how far below threshold in cells (0 if above)
            const pres_deficit = tf.relu(tf.scalar(minOccupancy).sub(cell_count_per_nca));
            const normalized_deficit = pres_deficit.div(minOccupancy + EPSILON);

            // sqrt scaling: gentle at first, stronger as extinction approaches
            const pres_boost = normalized_deficit.sqrt().mul(PRESERVATION_STRENGTH);

            // Apply boost only at species frontiers (where they can grow)
            // Mask walls before dilation to prevent boost leaking through walls
            const nca_alive_pres_masked = nca_alive_pres.mul(not_wall);
            const dilated_nca_pres = tf.maxPool(nca_alive_pres_masked, ALIVE_POOL_SIZE, 1, 'same');
            const frontier_mask = dilated_nca_pres.greater(MIN_PRESENCE_FOR_GROWTH).cast('float32');

            // Spatial boost: [1, N_NCAS] * [1, H, W, N_NCAS] -> [1, H, W, N_NCAS]
            const spatial_boost = pres_boost.reshape([1, 1, 1, N_NCAS]).mul(frontier_mask);

            // Add sun channel (zero boost)
            const sun_pres_boost = tf.zeros([1, grid.shape[1], grid.shape[2], 1]);
            const full_pres_boost = tf.concat([sun_pres_boost, spatial_boost], 3);

            raw_scores = raw_scores.add(full_pres_boost);
        }

        // Wall penalty - walls block all species from growing there
        const wall_mask = softThreshold(wallGrid, 0.5, 100.0);  // Very sharp for walls
        const wall_penalty = wall_mask.mul(1e9);
        const masked_scores = raw_scores.sub(wall_penalty);

        // Softmax temperature
        const competition_weights = tf.softmax(masked_scores.div(SOFTMAX_TEMP), 3);

        // Compute final update
        const weighted_updates = all_updates.mul(competition_weights.expandDims(4));
        const final_update = weighted_updates.sum(3);

        // Update hidden state
        const current_hidden = grid.slice([0, 0, 0, ALIVE_DIM]);
        const new_hidden = current_hidden.add(final_update.mul(0.5)).tanh();

        // Set alphas based on competition weights and growth mask
        const new_alphas = competition_weights.mul(growth_mask);

        // Reconstruct grid
        let new_grid = tf.concat([new_alphas, new_hidden], 3);

        // Survival check - handles soft competition where weights are distributed
        // Cell survives if: sun dominates (>threshold) OR NCAs have meaningful presence
        const sun_alpha = new_alphas.slice([0, 0, 0, 0], [-1, -1, -1, 1]);
        const nca_alphas = new_alphas.slice([0, 0, 0, 1], [-1, -1, -1, -1]);
        const nca_total = nca_alphas.sum(3, true);

        // Sun-dominated: sun needs high alpha
        const sun_survives = softThreshold(sun_alpha, ALIVE_THRESHOLD);
        // NCA-present: combined NCA presence > 0.15 (lenient for distributed weights)
        const nca_survives = softThreshold(nca_total, 0.15);
        const is_alive = tf.maximum(sun_survives, nca_survives);

        return new_grid.mul(is_alive);
    });
}

/**
 * Main competition function - routes to appropriate implementation.
 * 'global' = O(N) mean-field approximation (fast)
 * 'full' = O(N²) all-pairs comparison (accurate)
 */
function runCompetition(grid, all_updates, wallGrid, config) {
    if (GUI_STATE.COMPETITION_MODE === 'global') {
        return runCompetitionGlobal(grid, all_updates, wallGrid, config);
    } else {
        return runCompetitionFull(grid, all_updates, wallGrid, config);
    }
}

// =============================================================================
// SIMULATION STEP
// =============================================================================

/**
 * Performs a single simulation step.
 * @param {tf.Tensor} localWinRate - Optional per-cell running win rate [1, H, W, 1]
 * @returns {Object} { grid, localWinRate } if localWinRate provided, otherwise just grid
 */
function simulationStep(grid, model, sun_base, sun_params, wallGrid, config, localWinRate = null, inr = null, coords = null, inr_out_override = null) {
    return tf.tidy(() => {
        perfStart('sim-step-total');
        const { N_NCAS, CELL_WO_ALIVE_DIM, ALIVE_DIM, ALIVE_VISIBLE } = config;
        const h = grid.shape[1];
        const w = grid.shape[2];

        // Prepare model input
        const model_input = ALIVE_VISIBLE ? grid : grid.slice([0, 0, 0, ALIVE_DIM]);

        // Model forward pass
        perfStart('sim-model-predict');
        const model_updates = model.predict(model_input);
        const model_updates_reshaped = model_updates.reshape([
            1, h, w, N_NCAS, CELL_WO_ALIVE_DIM
        ]);
        perfEnd('sim-model-predict');

        // Sun update — sun tensors are 4D [1,H,W,CWA] or [1,1,1,CWA]; compute
        // in 4D (avoiding TF.js WebGL 5D shader bugs) then expand for concat.
        // In seasonal mode, the INR FULLY describes the sun (no base+params add).
        // Callers may pass `inr_out_override` to short-circuit the MLP forward
        // (used by trainStep to reuse a precomputed full-grid inference output).
        perfStart('sim-sun-update');
        let sun_pre;
        if (inr && (coords || inr_out_override)) {
            sun_pre = inr_out_override
                ? inr_out_override
                : evaluateInr(inr, coords, config);
        } else {
            sun_pre = sun_base.add(sun_params);
        }
        const sun_update_4d = sun_pre.tanh(); // 4D
        // Broadcast global sun to spatial dims if needed
        const sun_spatial = (sun_update_4d.shape[1] === 1)
            ? sun_update_4d.mul(tf.ones([1, h, w, 1], 'float32'))
            : sun_update_4d;
        // Expand to 5D [1,H,W,1,CWA] for concat with model updates [1,H,W,N,CWA]
        const sun_update_5d = sun_spatial.expandDims(3);

        // Concatenate all updates
        const all_updates = tf.concat([sun_update_5d, model_updates_reshaped], 3);
        perfEnd('sim-sun-update');

        // Run competition
        perfStart('sim-competition');
        const new_grid = runCompetition(grid, all_updates, wallGrid, config);
        perfEnd('sim-competition');

        // --- Local Win Rate Update ---
        let newLocalWinRate = null;
        if (GUI_STATE.USE_LOCAL_WIN_RATE && localWinRate) {
            // Get alive channels from new grid (competition result)
            const alive_channels = new_grid.slice([0, 0, 0, 0], [-1, -1, -1, ALIVE_DIM]);
            // Win strength = max aliveness at each cell (how strongly did the winner win?)
            const win_strength = alive_channels.max(3, true);  // [1, H, W, 1]

            // EMA update: localWinRate = alpha * old + (1-alpha) * new
            const alpha = GUI_STATE.LOCAL_WIN_RATE_ALPHA;
            newLocalWinRate = localWinRate.mul(alpha).add(win_strength.mul(1 - alpha));
        }

        // --- Stochastic Update Mask ---
        perfStart('sim-stochastic');
        if (GUI_STATE.STOCHASTIC_UPDATE_PROB < 1.0 || (GUI_STATE.USE_LOCAL_WIN_RATE && localWinRate)) {
            let updateProb;

            if (GUI_STATE.USE_LOCAL_WIN_RATE && localWinRate) {
                // Positive feedback: higher local win rate → higher update probability
                // Base probability modulated by local success
                const strength = GUI_STATE.LOCAL_WIN_RATE_STRENGTH;
                const baseProb = GUI_STATE.STOCHASTIC_UPDATE_PROB;
                // win rate is ~0-1, center around 0.5 for modulation
                // updateProb = baseProb * (1 + strength * (winRate - 0.5))
                updateProb = localWinRate.sub(0.5).mul(strength).add(1).mul(baseProb);
                // Clamp to valid range
                updateProb = updateProb.clipByValue(0.1, 1.0);
            } else {
                updateProb = tf.scalar(GUI_STATE.STOCHASTIC_UPDATE_PROB);
            }

            const random = tf.randomUniform([1, h, w, 1]);
            const updateMask = random.less(updateProb).cast('float32');

            // Blend: cells that don't update keep their old state
            const blendedGrid = new_grid.mul(updateMask).add(grid.mul(tf.scalar(1).sub(updateMask)));

            perfEnd('sim-stochastic');
            perfEnd('sim-step-total');
            if (newLocalWinRate) {
                return { grid: blendedGrid, localWinRate: newLocalWinRate };
            }
            return blendedGrid;
        }
        perfEnd('sim-stochastic');

        perfEnd('sim-step-total');
        if (newLocalWinRate) {
            return { grid: new_grid, localWinRate: newLocalWinRate };
        }
        return new_grid;
    });
}

// =============================================================================
// TRAINING
// =============================================================================

/**
 * Creates an optimizer based on GUI_STATE settings.
 *
 * SGD+Momentum is often faster for NCA tasks than Adam,
 * with less memory overhead from adaptive moment tracking.
 */
function createOptimizer() {
    const lr = GUI_STATE.LEARNING_RATE;
    switch (GUI_STATE.OPTIMIZER_TYPE) {
        case 'SGD':
            return tf.train.sgd(lr);
        case 'SGD+Momentum':
            // Momentum SGD: often better for NCAs, less memory than Adam
            return tf.train.momentum(lr, 0.9);
        case 'RMSProp':
            return tf.train.rmsprop(lr, undefined, undefined, undefined, GRADIENT_CLIP);
        case 'Adam':
        default:
            return tf.train.adam(lr, undefined, undefined, undefined, GRADIENT_CLIP);
    }
}

/**
 * Computes loss using soft-min objective + entropy diversity bonus.
 * - Soft-min focuses optimization on the weakest species
 * - Entropy bonus rewards equal population distribution
 *
 * Optimized: Uses cached scalars and minimal tensor allocations.
 */
function computeLoss(grid, config) {
    return tf.tidy(() => {
        const { N_NCAS } = config;
        const { ASINH_SLOPE, SOFT_MIN_K, DIVERSITY_WEIGHT } = GUI_STATE;

        // Get NCA aliveness (skip sun at index 0)
        const nca_alivenesses = grid.slice([0, 0, 0, 1], [-1, -1, -1, N_NCAS]);
        const mean_alive_per_nca = nca_alivenesses.mean([1, 2]);

        // Compute log-like growth using asinh (stable at 0)
        const log_growth = mean_alive_per_nca.div(ASINH_SLOPE).asinh().mul(ASINH_SLOPE);

        // Soft-min operation: focuses on weakest species
        const k = SOFT_MIN_K || 10.0;
        const neg_k_growth = log_growth.mul(-k);
        const log_sum_exp = tf.logSumExp(neg_k_growth, 1);
        const soft_min_growth = log_sum_exp.div(-k);

        // Base loss: negative of minimum growth (want to maximize)
        let loss = soft_min_growth.neg().sum();

        // --- ENTROPY DIVERSITY BONUS ---
        // Encourages equal population distribution across species
        if (DIVERSITY_WEIGHT > 0) {
            // Get population fractions (add small epsilon for stability)
            const total_alive = mean_alive_per_nca.sum(1, true).add(EPSILON);
            const pop_fractions = mean_alive_per_nca.div(total_alive).add(EPSILON);

            // Compute entropy: H = -sum(p * log(p))
            const entropy = pop_fractions.mul(pop_fractions.log()).sum(1).neg();

            // Normalize by max entropy (log(N)) so bonus is in [0, 1]
            const max_entropy = Math.log(N_NCAS);
            const normalized_entropy = entropy.div(max_entropy);

            // Add diversity bonus (negative because we want to MAXIMIZE entropy)
            // Higher entropy = more diverse = lower loss
            const diversity_bonus = normalized_entropy.mul(DIVERSITY_WEIGHT).sum();
            loss = loss.sub(diversity_bonus);
        }

        return loss;
    });
}

/**
 * Emergency respawn for species with near-zero cells.
 * Only called when preservation is enabled and a species is nearly extinct.
 */
function emergencyRespawn(grid, wallGrid, config) {
    return tf.tidy(() => {
        const minOccupancy = GUI_STATE.MIN_OCCUPANCY;
        if (minOccupancy <= 0) return grid;

        const { N_NCAS, ALIVE_DIM, CELL_WO_ALIVE_DIM, CELL_DIM, GRID_W, GRID_H } = config;

        // Check cell count per species (sum of aliveness values)
        const nca_alive = grid.slice([0, 0, 0, 1], [-1, -1, -1, N_NCAS]);
        const cell_count_per_nca = nca_alive.sum([1, 2]).dataSync();

        const RESPAWN_COUNT = 5;  // Seeds to inject
        const wallData = wallGrid.dataSync();

        // Find valid (non-wall) locations
        const validLocs = [];
        for (let y = 0; y < GRID_H; y++) {
            for (let x = 0; x < GRID_W; x++) {
                if (wallData[y * GRID_W + x] < 0.5) {
                    validLocs.push([y, x]);
                }
            }
        }
        if (validLocs.length === 0) return grid;

        const indices = [];
        const updates = [];

        for (let i = 0; i < N_NCAS; i++) {
            if (cell_count_per_nca[i] < 1.0) {  // Less than 1 cell equivalent
                // Pick random locations for respawn
                for (let s = 0; s < RESPAWN_COUNT; s++) {
                    const [y, x] = validLocs[Math.floor(seededRandom() * validLocs.length)];

                    // Build cell update
                    const aliveState = new Array(ALIVE_DIM).fill(0);
                    aliveState[i + 1] = 1.0;  // This species alive

                    const randomState = [];
                    for (let j = 0; j < CELL_WO_ALIVE_DIM; j++) {
                        randomState.push((seededRandom() - 0.5) * 0.2);
                    }

                    indices.push([0, y, x]);
                    updates.push([...aliveState, ...randomState]);
                }
            }
        }

        if (indices.length === 0) return grid;

        const idxTensor = tf.tensor2d(indices, [indices.length, 3], 'int32');
        const updateTensor = tf.tensor2d(updates, [indices.length, CELL_DIM]);
        return tf.tensorScatterUpdate(grid, idxTensor, updateTensor);
    });
}

/**
 * Subsamples a random contiguous patch for efficient optimization.
 */
function subsampleGrid(grid, config) {
    const { GRID_H, GRID_W } = config;
    const { OPTIMIZATION_PERCENT } = GUI_STATE;

    if (OPTIMIZATION_PERCENT >= 100) {
        return { grid: grid.clone(), startY: 0, startX: 0, patchH: GRID_H, patchW: GRID_W };
    }

    const percent = OPTIMIZATION_PERCENT / 100.0;
    const patch_area = GRID_H * GRID_W * percent;
    const grid_aspect = GRID_W / GRID_H;

    let patchH = Math.round(Math.sqrt(patch_area / grid_aspect));
    let patchW = Math.round(patchH * grid_aspect);

    patchH = Math.max(1, Math.min(GRID_H, patchH));
    patchW = Math.max(1, Math.min(GRID_W, patchW));

    if (patchH > 1 && patchH % 2 !== 0) patchH -= 1;
    if (patchW > 1 && patchW % 2 !== 0) patchW -= 1;

    const startY = Math.floor(seededRandom() * (GRID_H - patchH + 1));
    const startX = Math.floor(seededRandom() * (GRID_W - patchW + 1));

    return {
        grid: grid.slice([0, startY, startX, 0], [1, patchH, patchW, -1]),
        startY, startX, patchH, patchW
    };
}

// Store latest loss for async retrieval (non-blocking)
let _lastTrainingLoss = 0;
function getLastTrainingLoss() {
    return _lastTrainingLoss;
}

/**
 * Performs a single training step (forward + backward + update).
 *
 * OPTIMIZATION: Uses one-step lag approach to eliminate double forward pass.
 * We run inference on the full grid FIRST, then compute gradients on a subsample.
 * The displayed state uses pre-update weights, but weight changes per step are
 * minimal so visual difference is negligible.
 *
 * OPTIMIZATION: Loss retrieval is async and non-blocking. The loss value is
 * stored in _lastTrainingLoss and updated when the GPU finishes. This prevents
 * the ~125ms GPU sync stall from blocking the main thread.
 */
async function trainStep(grid, model, sun_base, sun_params, wallGrid, optimizer, config, onProgress, inr = null, coords = null) {
    perfStart('train-step-total');

    // ONE-STEP LAG: Run inference on full grid FIRST (before weight update)
    // On the first step after a config change, each phase triggers WebGL shader
    // compilation (1-6s each). Yielding before each phase lets the browser:
    // (a) paint the loading spinner status text, and
    // (b) reset Chrome's "Page Unresponsive" timer.
    // The onProgress callback is only set on the first step, so yields only
    // happen when they're actually needed.
    if (onProgress) {
        onProgress('Forward pass → model predict + competition + stochastic update');
        await new Promise(r => setTimeout(r, 0));
    }
    // If the INR is frozen (SUN_LR_SCALE=0), its gradient path is wasted work.
    // Pre-compute its full-grid output ONCE and feed it as a detached constant
    // to both the inference pass and the loss closure.
    const sunFrozen = (GUI_STATE.SUN_LR_SCALE === 0);
    let inr_out_full = null;
    if (inr && coords && sunFrozen) {
        inr_out_full = tf.keep(evaluateInr(inr, coords, config));
    }

    perfStart('train-inference');
    let newGrid = tf.tidy(() =>
        simulationStep(grid, model, sun_base, sun_params, wallGrid, config, null, inr, coords, inr_out_full)
    );
    perfEnd('train-inference');

    if (onProgress) {
        onProgress('Backward pass → gradient computation on subsample');
        await new Promise(r => setTimeout(r, 0));
    }

    // Now compute gradients on a subsample (using current weights)
    perfStart('train-gradient');
    const { grid: optimizationGrid, startY, startX, patchH, patchW } = subsampleGrid(grid, config);
    const wallGridSample = wallGrid.slice([0, startY, startX, 0], [1, patchH, patchW, -1]);

    const isLocalSun = sun_params.shape[1] > 1;
    const lossFunction = (currentGrid) => tf.tidy(() => {
        // For local sun, slice 4D tensors to match the crop region.
        // TF.js tape records the slice; gradient back-propagates as a pad with zeros.
        let sun_base_eff = sun_base;
        let sun_params_eff = sun_params;
        let coords_eff = coords;
        let inr_out_eff = null;
        if (isLocalSun) {
            sun_base_eff = sun_base.slice([0, startY, startX, 0], [1, patchH, patchW, -1]);
            sun_params_eff = sun_params.slice([0, startY, startX, 0], [1, patchH, patchW, -1]);
            if (coords) {
                coords_eff = coords.slice([0, startY, startX, 0], [1, patchH, patchW, -1]);
            }
            if (inr_out_full) {
                inr_out_eff = inr_out_full.slice([0, startY, startX, 0], [1, patchH, patchW, -1]);
            }
        } else if (inr_out_full) {
            inr_out_eff = inr_out_full;
        }
        const next_grid = simulationStep(
            currentGrid, model, sun_base_eff, sun_params_eff, wallGridSample, config, null,
            inr, coords_eff, inr_out_eff
        );
        return computeLoss(next_grid, config);
    });

    const modelVars = model.trainableWeights.map(w => w.val);
    // When INR is frozen we feed its output as a detached constant, so none of
    // the INR vars participate in the tape. Excluding them from the varList
    // avoids computeGradients attempting to trace dead paths.
    const inrVars = sunFrozen ? [] : inrTrainableVars(inr);
    const varList = modelVars.concat([sun_params]).concat(inrVars);

    const { value, grads } = optimizer.computeGradients(() => lossFunction(optimizationGrid), varList);
    perfEnd('train-gradient');

    if (onProgress) {
        onProgress('Applying gradients → optimizer step + sun update');
        await new Promise(r => setTimeout(r, 0));
    }
    perfStart('train-apply-grads');
    // Strip the sun's gradient from the main optimizer's dict and apply it ourselves
    // as a manually-scaled SGD step. This sidesteps Adam's m/sqrt(v) cancellation,
    // so the user-facing semantics of SUN_LR_SCALE are exactly "multiplier on the LR".
    const sunGradName = sun_params.name;
    const sunGrad = grads[sunGradName];
    delete grads[sunGradName];

    // Strip INR gradients from the main optimizer's dict — same manual-SGD
    // pattern as sun_params so SUN_LR_SCALE=0 is a true hard freeze of the
    // entire INR (no Adam moments to leak through).
    const inrGradEntries = [];
    for (const v of inrVars) {
        const g = grads[v.name];
        if (g !== undefined) {
            inrGradEntries.push({ v, g });
            delete grads[v.name];
        }
    }

    optimizer.applyGradients(grads);

    const sunScale = GUI_STATE.SUN_LR_SCALE;
    const lr = GUI_STATE.LEARNING_RATE;
    if (sunScale > 0 && sunGrad) {
        tf.tidy(() => {
            sun_params.assign(sun_params.sub(sunGrad.mul(lr * sunScale)));
        });
    }
    if (sunGrad) sunGrad.dispose();

    if (sunScale > 0 && inrGradEntries.length > 0) {
        tf.tidy(() => {
            for (const { v, g } of inrGradEntries) {
                v.assign(v.sub(g.mul(lr * sunScale)));
            }
            if (inr) {
                // Clamp log_period to prevent drift to degenerate frequencies.
                inr.inr_log_period.assign(
                    inr.inr_log_period.clipByValue(INR_LOG_PERIOD_MIN, INR_LOG_PERIOD_MAX)
                );
            }
        });
    }
    for (const { g } of inrGradEntries) g.dispose();
    perfEnd('train-apply-grads');

    // NON-BLOCKING: Fire off async loss retrieval, don't wait for GPU sync
    // This prevents the ~125ms stall from blocking the simulation loop
    value.data().then(lossData => {
        _lastTrainingLoss = lossData[0];
        value.dispose();
    });

    for (const varName in grads) {
        grads[varName].dispose();
    }

    // Cleanup subsampled grids
    optimizationGrid.dispose();
    wallGridSample.dispose();
    if (inr_out_full) inr_out_full.dispose();

    perfEnd('train-step-total');
    return { newGrid, loss: _lastTrainingLoss };  // Return last known loss (may be from previous step)
}

// =============================================================================
// VISUALIZATION HELPERS
// =============================================================================

/**
 * Color schemes for species visualization.
 * Each scheme is designed for visual appeal and distinguishability.
 */
const COLOR_SCHEMES = {
    'Vibrant': [
        [0.349, 0.784, 0.980],  // Electric Blue
        [0.976, 0.341, 0.486],  // Coral Pink
        [0.200, 0.843, 0.659],  // Mint Green
        [0.992, 0.702, 0.180],  // Golden Yellow
        [0.647, 0.443, 0.976],  // Lavender Purple
        [1.000, 0.502, 0.314],  // Tangerine
        [0.000, 0.820, 0.878],  // Cyan
        [0.957, 0.263, 0.678],  // Hot Pink
    ],
    'Neon': [
        [0.000, 1.000, 0.875],  // Neon Cyan
        [1.000, 0.000, 0.500],  // Neon Pink
        [0.498, 1.000, 0.000],  // Neon Green
        [1.000, 0.800, 0.000],  // Neon Yellow
        [0.878, 0.000, 1.000],  // Neon Purple
        [1.000, 0.400, 0.000],  // Neon Orange
        [0.000, 0.600, 1.000],  // Neon Blue
        [1.000, 0.000, 0.200],  // Neon Red
    ],
    'Pastel': [
        [0.678, 0.847, 0.902],  // Light Blue
        [1.000, 0.714, 0.757],  // Light Pink
        [0.596, 0.984, 0.596],  // Light Green
        [1.000, 0.937, 0.678],  // Light Yellow
        [0.800, 0.600, 0.800],  // Light Purple
        [1.000, 0.800, 0.600],  // Light Orange
        [0.600, 0.800, 0.800],  // Light Teal
        [1.000, 0.600, 0.678],  // Light Coral
    ],
    'Earth': [
        [0.545, 0.353, 0.169],  // Saddle Brown
        [0.824, 0.706, 0.549],  // Tan
        [0.333, 0.420, 0.184],  // Olive
        [0.871, 0.722, 0.529],  // Wheat
        [0.627, 0.322, 0.176],  // Sienna
        [0.557, 0.537, 0.286],  // Khaki Dark
        [0.467, 0.533, 0.600],  // Slate
        [0.804, 0.522, 0.247],  // Peru
    ],
    'Ocean': [
        [0.000, 0.467, 0.745],  // Deep Blue
        [0.118, 0.565, 0.604],  // Teal
        [0.259, 0.808, 0.902],  // Sky Blue
        [0.000, 0.392, 0.392],  // Dark Cyan
        [0.275, 0.510, 0.706],  // Steel Blue
        [0.498, 0.753, 0.753],  // Light Sea
        [0.125, 0.698, 0.667],  // Turquoise
        [0.282, 0.239, 0.545],  // Slate Blue
    ],
    'Sunset': [
        [0.988, 0.369, 0.259],  // Tomato
        [1.000, 0.549, 0.000],  // Dark Orange
        [0.988, 0.753, 0.286],  // Golden
        [0.933, 0.510, 0.510],  // Light Coral
        [0.867, 0.267, 0.467],  // Rose
        [0.800, 0.200, 0.200],  // Firebrick
        [1.000, 0.388, 0.278],  // Coral
        [0.941, 0.502, 0.502],  // Light Salmon
    ],
    'Candy': [
        [1.000, 0.412, 0.706],  // Hot Pink
        [0.678, 0.847, 0.902],  // Powder Blue
        [0.933, 0.510, 0.933],  // Violet
        [0.498, 1.000, 0.831],  // Aquamarine
        [1.000, 0.753, 0.796],  // Pink
        [0.686, 0.933, 0.933],  // Pale Turquoise
        [0.855, 0.439, 0.839],  // Orchid
        [0.596, 0.984, 0.596],  // Pale Green
    ],
    'Forest': [
        [0.133, 0.545, 0.133],  // Forest Green
        [0.604, 0.804, 0.196],  // Yellow Green
        [0.420, 0.557, 0.137],  // Olive Drab
        [0.545, 0.271, 0.075],  // Saddle Brown
        [0.498, 0.498, 0.000],  // Olive
        [0.333, 0.420, 0.184],  // Dark Olive
        [0.180, 0.545, 0.341],  // Sea Green
        [0.565, 0.933, 0.565],  // Light Green
    ],
    'Retro': [
        [1.000, 0.078, 0.576],  // Deep Pink (Synthwave)
        [0.000, 1.000, 1.000],  // Cyan
        [1.000, 0.843, 0.000],  // Gold
        [0.580, 0.000, 0.827],  // Dark Violet
        [1.000, 0.271, 0.000],  // Orange Red
        [0.118, 0.565, 1.000],  // Dodger Blue
        [0.698, 0.133, 0.133],  // Firebrick
        [0.000, 0.808, 0.820],  // Dark Cyan
    ],
    'Mono Blue': [
        [0.118, 0.565, 1.000],  // Dodger Blue
        [0.255, 0.412, 0.882],  // Royal Blue
        [0.000, 0.749, 1.000],  // Deep Sky Blue
        [0.275, 0.510, 0.706],  // Steel Blue
        [0.529, 0.808, 0.922],  // Sky Blue
        [0.000, 0.502, 0.502],  // Teal
        [0.282, 0.239, 0.545],  // Slate Blue
        [0.392, 0.584, 0.929],  // Cornflower Blue
    ],
    'Rainbow': [
        [1.000, 0.000, 0.000],  // Red
        [1.000, 0.498, 0.000],  // Orange
        [1.000, 1.000, 0.000],  // Yellow
        [0.000, 1.000, 0.000],  // Green
        [0.000, 1.000, 1.000],  // Cyan
        [0.000, 0.000, 1.000],  // Blue
        [0.545, 0.000, 0.545],  // Purple
        [1.000, 0.000, 1.000],  // Magenta
    ],
    'High Contrast': [
        [1.000, 0.000, 0.000],  // Pure Red
        [0.000, 0.000, 1.000],  // Pure Blue
        [1.000, 1.000, 0.000],  // Pure Yellow
        [0.000, 1.000, 0.000],  // Pure Green
        [1.000, 0.000, 1.000],  // Magenta
        [0.000, 1.000, 1.000],  // Cyan
        [1.000, 0.500, 0.000],  // Orange
        [0.500, 0.000, 1.000],  // Violet
    ],
};

// Current active palette (reference to one of the schemes)
let ACTIVE_PALETTE = COLOR_SCHEMES['Vibrant'];

/**
 * Sets the active color scheme by name.
 */
function setColorScheme(schemeName) {
    if (COLOR_SCHEMES[schemeName]) {
        ACTIVE_PALETTE = COLOR_SCHEMES[schemeName];
        GUI_STATE.COLOR_SCHEME = schemeName;
    }
}

/**
 * Legacy alias for backwards compatibility
 */
const MODERN_PALETTE = [
    // Primary vibrant colors (first 8 - most distinct)
    [0.349, 0.784, 0.980],  // Electric Blue (#59C8FA)
    [0.976, 0.341, 0.486],  // Coral Pink (#F9577C)
    [0.200, 0.843, 0.659],  // Mint Green (#33D7A8)
    [0.992, 0.702, 0.180],  // Golden Yellow (#FDB32E)
    [0.647, 0.443, 0.976],  // Lavender Purple (#A571F9)
    [1.000, 0.502, 0.314],  // Tangerine (#FF8050)
    [0.000, 0.820, 0.878],  // Cyan (#00D1E0)
    [0.957, 0.263, 0.678],  // Hot Pink (#F443AD)

    // Secondary colors (next 8 - still distinct)
    [0.467, 0.933, 0.467],  // Lime (#77EE77)
    [0.557, 0.647, 0.996],  // Periwinkle (#8EA5FE)
    [1.000, 0.839, 0.373],  // Amber (#FFD65F)
    [0.373, 0.612, 0.627],  // Teal (#5F9CA0)
    [0.906, 0.553, 0.780],  // Rose (#E78DC7)
    [0.745, 0.902, 0.420],  // Yellow-Green (#BEE66B)
    [0.686, 0.478, 0.667],  // Mauve (#AF7AAA)
    [0.996, 0.600, 0.557],  // Salmon (#FE998E)

    // Extended colors (for 17-24 species)
    [0.204, 0.659, 0.863],  // Sky Blue (#34A8DC)
    [0.863, 0.431, 0.549],  // Dusty Rose (#DC6E8C)
    [0.494, 0.745, 0.502],  // Sage (#7EBE80)
    [0.933, 0.765, 0.502],  // Peach (#EEC380)
    [0.537, 0.467, 0.745],  // Iris (#8977BE)
    [0.698, 0.537, 0.388],  // Terracotta (#B28963)
    [0.431, 0.749, 0.749],  // Aqua (#6EBFBF)
    [0.835, 0.659, 0.890],  // Lilac (#D5A8E3)
];

/**
 * Generates N distinct colors for NCA visualization.
 * Uses a curated modern palette for visual appeal.
 */
function generateNCAColors(n_ncas) {
    const colors = [];

    // Sun color (very dark - background)
    colors.push([0.05, 0.05, 0.06]);

    const paletteSize = ACTIVE_PALETTE.length;

    if (n_ncas <= paletteSize) {
        // Use pure palette colors for 8 or fewer species
        for (let i = 0; i < n_ncas; i++) {
            colors.push(ACTIVE_PALETTE[i]);
        }
    } else {
        // For 8+ species, use pure colors first, then generate variations
        // First pass: all pure palette colors
        for (let i = 0; i < paletteSize; i++) {
            colors.push(ACTIVE_PALETTE[i]);
        }

        // Generate variations for remaining species
        const remaining = n_ncas - paletteSize;
        const variations = generateColorVariations(ACTIVE_PALETTE, remaining);
        colors.push(...variations);
    }

    return colors;
}

/**
 * Generate color variations from a base palette.
 * Uses HSL adjustments: saturation, lightness, and slight hue shifts.
 */
function generateColorVariations(basePalette, count) {
    const variations = [];
    const paletteSize = basePalette.length;

    // Variation strategies (applied in rounds)
    const strategies = [
        { satMult: 0.7, lightAdd: 0.15, hueShift: 0 },      // Lighter, less saturated
        { satMult: 1.0, lightAdd: -0.15, hueShift: 0 },     // Darker
        { satMult: 0.85, lightAdd: 0, hueShift: 0.05 },     // Slight hue shift +
        { satMult: 0.85, lightAdd: 0, hueShift: -0.05 },    // Slight hue shift -
        { satMult: 0.6, lightAdd: 0.25, hueShift: 0 },      // Very light pastel
        { satMult: 1.0, lightAdd: -0.25, hueShift: 0.03 },  // Dark with hue shift
    ];

    let strategyIndex = 0;
    let paletteIndex = 0;

    for (let i = 0; i < count; i++) {
        const baseColor = basePalette[paletteIndex];
        const strategy = strategies[strategyIndex];

        // Convert RGB to HSL
        const [h, s, l] = rgbToHsl(baseColor[0], baseColor[1], baseColor[2]);

        // Apply variation
        let newH = (h + strategy.hueShift + 1) % 1;
        let newS = Math.max(0.2, Math.min(1, s * strategy.satMult));
        let newL = Math.max(0.15, Math.min(0.85, l + strategy.lightAdd));

        // Convert back to RGB
        const [r, g, b] = hslToRgb(newH, newS, newL);
        variations.push([r, g, b]);

        // Cycle through palette, then strategies
        paletteIndex = (paletteIndex + 1) % paletteSize;
        if (paletteIndex === 0) {
            strategyIndex = (strategyIndex + 1) % strategies.length;
        }
    }

    return variations;
}

/**
 * RGB to HSL conversion
 */
function rgbToHsl(r, g, b) {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h, s;
    const l = (max + min) / 2;

    if (max === min) {
        h = s = 0;
    } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
            case g: h = ((b - r) / d + 2) / 6; break;
            case b: h = ((r - g) / d + 4) / 6; break;
        }
    }

    return [h, s, l];
}

/**
 * HSL to RGB conversion
 */
function hslToRgb(h, s, l) {
    let r, g, b;

    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }

    return [r, g, b];
}

/**
 * HSV to RGB conversion (fallback for extended palettes)
 */
function hsvToRgb(h, s, v) {
    let r, g, b;
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return [r, g, b];
}

// =============================================================================
// CHECKPOINT SERIALIZATION
// =============================================================================

/**
 * Serialize all model weights to an array of {index, shape, data} objects.
 * Uses async tensor.data() to avoid blocking the main thread.
 */
async function serializeModelWeights(model) {
    const weights = model.trainableWeights;
    const promises = weights.map((w, i) =>
        w.val.data().then(data => ({
            index: i,
            shape: w.val.shape.slice(),
            data: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
        }))
    );
    return Promise.all(promises);
}

/**
 * Restore model weights from serialized data (by positional index).
 */
function restoreModelWeights(model, serializedWeights) {
    const weights = model.trainableWeights;
    for (const sw of serializedWeights) {
        if (sw.index < weights.length) {
            const tensor = tf.tensor(new Float32Array(sw.data), sw.shape);
            weights[sw.index].val.assign(tensor);
            tensor.dispose();
        }
    }
}

/**
 * Serialize the full grid state to ArrayBuffers.
 * Returns a Promise resolving to {grid, wall, winRate, sunBase, sunParams}.
 */
async function serializeGridState(simStateObj) {
    const [gridData, wallData, winRateData, sunBaseData, sunParamsData] = await Promise.all([
        simStateObj.grid.data(),
        simStateObj.wallGrid.data(),
        simStateObj.localWinRate ? simStateObj.localWinRate.data() : Promise.resolve(null),
        simStateObj.sun_base.data(),
        simStateObj.sun_params.data(),
    ]);

    const copyBuffer = (typedArray) =>
        typedArray.buffer.slice(typedArray.byteOffset, typedArray.byteOffset + typedArray.byteLength);

    const out = {
        grid: copyBuffer(gridData),
        gridShape: simStateObj.grid.shape.slice(),
        wall: copyBuffer(wallData),
        wallShape: simStateObj.wallGrid.shape.slice(),
        winRate: winRateData ? copyBuffer(winRateData) : null,
        winRateShape: simStateObj.localWinRate ? simStateObj.localWinRate.shape.slice() : null,
        sunBase: copyBuffer(sunBaseData),
        sunBaseShape: simStateObj.sun_base.shape.slice(),
        sunParams: copyBuffer(sunParamsData),
        sunParamsShape: simStateObj.sun_params.shape.slice(),
    };

    // Seasonal sun: serialize every INR tensor's data + shape.
    if (simStateObj.inr) {
        const inr = simStateObj.inr;
        const keys = Object.keys(inr);
        const datas = await Promise.all(keys.map(k => inr[k].data()));
        out.inr = {};
        keys.forEach((k, i) => {
            out.inr[k] = {
                data: copyBuffer(datas[i]),
                shape: inr[k].shape.slice(),
            };
        });
    }

    return out;
}

/**
 * Restore grid state from checkpoint data.
 * Disposes old tensors and creates new ones.
 */
function restoreGridState(simStateObj, checkpointData) {
    // Dispose old tensors
    if (simStateObj.grid) simStateObj.grid.dispose();
    if (simStateObj.wallGrid) simStateObj.wallGrid.dispose();
    if (simStateObj.localWinRate) simStateObj.localWinRate.dispose();
    if (simStateObj.sun_base) simStateObj.sun_base.dispose();
    if (simStateObj.sun_params) simStateObj.sun_params.dispose();
    if (simStateObj.inr) { disposeInr(simStateObj.inr); simStateObj.inr = null; }
    if (simStateObj.inr_coords) { simStateObj.inr_coords.dispose(); simStateObj.inr_coords = null; }

    // Restore grid (validate buffer size matches shape)
    const gridArray = new Float32Array(checkpointData.grid);
    const expectedGridSize = checkpointData.gridShape.reduce((a, b) => a * b, 1);
    if (gridArray.length !== expectedGridSize) {
        console.error(`Grid size mismatch: buffer has ${gridArray.length} values, shape ${checkpointData.gridShape} expects ${expectedGridSize}`);
        throw new Error('Checkpoint grid dimensions do not match current configuration. Try clearing browser data and saving a new checkpoint.');
    }
    simStateObj.grid = tf.keep(
        tf.tensor(gridArray, checkpointData.gridShape)
    );

    // Restore wall grid
    simStateObj.wallGrid = tf.keep(
        tf.tensor(new Float32Array(checkpointData.wall), checkpointData.wallShape)
    );

    // Restore local win rate
    if (checkpointData.winRate && checkpointData.winRateShape) {
        simStateObj.localWinRate = tf.keep(
            tf.tensor(new Float32Array(checkpointData.winRate), checkpointData.winRateShape)
        );
    } else {
        simStateObj.localWinRate = tf.keep(
            tf.fill(simStateObj.grid.shape.slice(0, 3).concat([1]), 0.5)
        );
    }

    // Restore sun tensors (migrate old 5D shapes to 4D if needed)
    let sunBaseShape = checkpointData.sunBaseShape;
    let sunParamsShape = checkpointData.sunParamsShape;
    // Old checkpoints used 5D [1,H,W,1,CWA] or [1,1,1,1,CWA]; squeeze dim 3 to get 4D
    if (sunBaseShape.length === 5) {
        sunBaseShape = [sunBaseShape[0], sunBaseShape[1], sunBaseShape[2], sunBaseShape[4]];
    }
    if (sunParamsShape.length === 5) {
        sunParamsShape = [sunParamsShape[0], sunParamsShape[1], sunParamsShape[2], sunParamsShape[4]];
    }
    simStateObj.sun_base = tf.keep(
        tf.tensor(new Float32Array(checkpointData.sunBase), sunBaseShape)
    );
    simStateObj.sun_params = tf.keep(
        tf.variable(
            tf.tensor(new Float32Array(checkpointData.sunParams), sunParamsShape),
            true
        )
    );

    // Restore INR if present
    if (checkpointData.inr) {
        simStateObj.inr = createInrParams({ CELL_WO_ALIVE_DIM: simStateObj.sun_base.shape[3] });
        tf.tidy(() => {
            for (const k of Object.keys(checkpointData.inr)) {
                if (!(k in simStateObj.inr)) continue;
                const { data, shape } = checkpointData.inr[k];
                const t = tf.tensor(new Float32Array(data), shape);
                simStateObj.inr[k].assign(t);
            }
        });
        simStateObj.inr_coords = buildCoordsCache(
            simStateObj.sun_base.shape[1], simStateObj.sun_base.shape[2]
        );
        GUI_STATE.SUN_MODE = 'seasonal';
    } else {
        GUI_STATE.SUN_MODE = (simStateObj.sun_base.shape[1] > 1) ? 'spatial' : 'global';
    }
}

/**
 * Capture a 64x64 thumbnail from the main canvas.
 * Returns a Promise resolving to a Blob (PNG).
 */
function captureThumbnail(sourceCanvas, size = 64) {
    return new Promise((resolve) => {
        const thumb = document.createElement('canvas');
        thumb.width = size;
        thumb.height = size;
        const ctx = thumb.getContext('2d');

        // Crop center square from source canvas
        const sw = sourceCanvas.width;
        const sh = sourceCanvas.height;
        const cropSize = Math.min(sw, sh);
        const sx = (sw - cropSize) / 2;
        const sy = (sh - cropSize) / 2;

        ctx.drawImage(sourceCanvas, sx, sy, cropSize, cropSize, 0, 0, size, size);
        thumb.toBlob(blob => resolve(blob), 'image/png');
    });
}
