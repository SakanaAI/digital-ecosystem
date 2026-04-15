/**
 * Petri Dish NCA - Main Application
 *
 * This file handles the UI, rendering loop, and user interactions.
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const LOG_PERFORMANCE = true;  // Enable performance logging for optimization verification
const MAX_GRID_DIM_DEFAULT = 2048;

// =============================================================================
// GLOBAL STATE
// =============================================================================

const simState = {
    // Simulation state
    isTraining: false,
    isDrawing: false,
    stepCount: 0,
    globalStep: 0,

    // WebGL
    glsl: null,
    gridTex: null,
    wallTex: null,
    renderShader: null,

    // TensorFlow
    model: null,
    sun_base: null,
    sun_params: null,
    inr: null,              // Seasonal INR params (null when SUN_MODE !== 'seasonal')
    inr_coords: null,       // Normalized (x,y) coordinate cache for the INR
    optimizer: null,
    grid: null,
    wallGrid: null,
    localWinRate: null,  // Per-cell running success rate for update modulation

    // Rendering
    nca_colors: null,
    color_uniform: null,
    TEX_W: 0,
    TEX_H: 0,
    maxGridDim: MAX_GRID_DIM_DEFAULT,

    // GUI
    gui: null,
    startPauseController: null,
    toolControllers: {},
    drawFolder: null,

    // Drawing interpolation
    lastDrawX: null,
    lastDrawY: null,
};

// =============================================================================
// BUTTON HANDLERS
// =============================================================================

const GUI_BUTTONS = {
    startPause: () => {
        simState.isTraining = !simState.isTraining;
        const btnEl = simState.startPauseController.domElement.parentElement.parentElement;
        if (simState.isTraining) {
            simState.startPauseController.name('Pause');
            btnEl.classList.remove('btn-start');
            btnEl.classList.add('btn-pause');
            simulationLoop();
        } else {
            hideLoadingSpinner();
            simState.startPauseController.name('Start');
            btnEl.classList.remove('btn-pause');
            btnEl.classList.add('btn-start');
        }
    },
    hardReset: () => resetSimulation(false),
    reseed: () => addSeeds(),
    selectPencil: () => setTool('pencil'),
    selectEraser: () => setTool('eraser'),
    exportParams: () => exportParamsToFile(),
    loadParams: () => triggerLoadParams(),
    exportRun: () => exportRunAsPetri(),
    importRun: () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.petri';
        input.onchange = (e) => {
            if (e.target.files[0]) importPetriFile(e.target.files[0]);
        };
        input.click();
    },
    loadShowcase: async () => {
        try {
            if (window.stopShowcaseGlow) window.stopShowcaseGlow();
            showLoadingSpinner('Downloading preset...');
            const resp = await fetch('presets/default-ecosystem.petri');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const buffer = await resp.arrayBuffer();

            showLoadingSpinner('Building model...');
            await new Promise(r => setTimeout(r, 0)); // yield so browser paints
            const { header, binaryData } = parsePetriBundle(buffer);
            if (!header.checkpoints || header.checkpoints.length === 0) throw new Error('No checkpoints');
            const cpMeta = header.checkpoints[header.checkpoints.length - 1];
            const checkpoint = extractCheckpointFromBundle(cpMeta, binaryData);

            showLoadingSpinner('Restoring ecosystem...');
            await new Promise(r => setTimeout(r, 0));
            // Load checkpoint directly — skip IndexedDB, timeline, checkpoint tray
            await loadCheckpointData(checkpoint);
            // Reset step counter to 0 so the timeline starts fresh
            simState.globalStep = 0;
            hideLoadingSpinner();
            showToast('Showcase ecosystem loaded');
            // Auto-start the simulation
            if (!simState.isTraining) {
                GUI_BUTTONS.startPause();
            }
        } catch (e) {
            hideLoadingSpinner();
            console.error('Failed to load showcase preset:', e);
            showToast('Showcase load failed: ' + e.message);
        }
    },
    copyRecipe: () => copyRecipeURL(),
};

/**
 * Export current parameters to a downloadable .txt file
 */
function exportParamsToFile() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `petri-dish-params-${timestamp}.txt`;

    const params = {
        // Simulation
        SOFTMAX_TEMP: GUI_STATE.SOFTMAX_TEMP,
        ALIVE_THRESHOLD: GUI_STATE.ALIVE_THRESHOLD,
        LEARNING_BALANCE: GUI_STATE.LEARNING_BALANCE,
        LEARNING_RATE: GUI_STATE.LEARNING_RATE,
        SUN_LR_SCALE: GUI_STATE.SUN_LR_SCALE,
        GROWTH_GATE_STEEPNESS: GUI_STATE.GROWTH_GATE_STEEPNESS,
        UPDATE_DELAY_MS: GUI_STATE.UPDATE_DELAY_MS,
        COMPETITION_MODE: GUI_STATE.COMPETITION_MODE,
        STOCHASTIC_UPDATE_PROB: GUI_STATE.STOCHASTIC_UPDATE_PROB,
        USE_LOCAL_WIN_RATE: GUI_STATE.USE_LOCAL_WIN_RATE,
        LOCAL_WIN_RATE_ALPHA: GUI_STATE.LOCAL_WIN_RATE_ALPHA,
        LOCAL_WIN_RATE_STRENGTH: GUI_STATE.LOCAL_WIN_RATE_STRENGTH,

        // Training
        OPTIMIZATION_PERCENT: GUI_STATE.OPTIMIZATION_PERCENT,
        ASINH_SLOPE: GUI_STATE.ASINH_SLOPE,
        SOFT_MIN_K: GUI_STATE.SOFT_MIN_K,
        DIVERSITY_WEIGHT: GUI_STATE.DIVERSITY_WEIGHT,
        SUN_PENALTY: GUI_STATE.SUN_PENALTY,
        MIN_OCCUPANCY: GUI_STATE.MIN_OCCUPANCY,
        USE_COSINE_SIM: GUI_STATE.USE_COSINE_SIM,

        // Model
        N_NCAS: GUI_STATE.N_NCAS,
        CELL_STATE_DIM: GUI_STATE.CELL_STATE_DIM,
        CELL_HIDDEN_DIM: GUI_STATE.CELL_HIDDEN_DIM,
        HIDDEN_DIM: GUI_STATE.HIDDEN_DIM,
        N_HIDDEN_LAYERS: GUI_STATE.N_HIDDEN_LAYERS,
        KERNEL_SIZE: GUI_STATE.KERNEL_SIZE,
        MODEL_DROPOUT_P: GUI_STATE.MODEL_DROPOUT_P,
        OPTIMIZER_TYPE: GUI_STATE.OPTIMIZER_TYPE,
        SUN_INIT_SCALE: GUI_STATE.SUN_INIT_SCALE,

        // Concentration
        USE_CONCENTRATION: GUI_STATE.USE_CONCENTRATION,
        CONCENTRATION_TEMP: GUI_STATE.CONCENTRATION_TEMP,
        CONC_BASELINE: GUI_STATE.CONC_BASELINE,
        CONCENTRATION_WINDOW: GUI_STATE.CONCENTRATION_WINDOW,

        // Grid
        GRID_W: GUI_STATE.GRID_W,
        GRID_H: GUI_STATE.GRID_H,
        N_SEEDS: GUI_STATE.N_SEEDS,

        // Visual
        COLOR_SCHEME: GUI_STATE.COLOR_SCHEME,
    };

    let content = `# Petri Dish NCA Parameters\n`;
    content += `# Exported: ${new Date().toLocaleString()}\n`;
    content += `# ================================\n\n`;

    content += `[Simulation]\n`;
    content += `SOFTMAX_TEMP = ${params.SOFTMAX_TEMP}\n`;
    content += `ALIVE_THRESHOLD = ${params.ALIVE_THRESHOLD}\n`;
    content += `LEARNING_BALANCE = ${params.LEARNING_BALANCE}\n`;
    content += `LEARNING_RATE = ${params.LEARNING_RATE}\n`;
    content += `SUN_LR_SCALE = ${params.SUN_LR_SCALE}\n`;
    content += `GROWTH_GATE_STEEPNESS = ${params.GROWTH_GATE_STEEPNESS}\n`;
    content += `UPDATE_DELAY_MS = ${params.UPDATE_DELAY_MS}\n`;
    content += `COMPETITION_MODE = ${params.COMPETITION_MODE}\n`;
    content += `STOCHASTIC_UPDATE_PROB = ${params.STOCHASTIC_UPDATE_PROB}\n`;
    content += `USE_LOCAL_WIN_RATE = ${params.USE_LOCAL_WIN_RATE}\n`;
    content += `LOCAL_WIN_RATE_ALPHA = ${params.LOCAL_WIN_RATE_ALPHA}\n`;
    content += `LOCAL_WIN_RATE_STRENGTH = ${params.LOCAL_WIN_RATE_STRENGTH}\n\n`;

    content += `[Training]\n`;
    content += `OPTIMIZATION_PERCENT = ${params.OPTIMIZATION_PERCENT}\n`;
    content += `ASINH_SLOPE = ${params.ASINH_SLOPE}\n`;
    content += `SOFT_MIN_K = ${params.SOFT_MIN_K}\n`;
    content += `DIVERSITY_WEIGHT = ${params.DIVERSITY_WEIGHT}\n`;
    content += `SUN_PENALTY = ${params.SUN_PENALTY}\n`;
    content += `MIN_OCCUPANCY = ${params.MIN_OCCUPANCY}\n`;
    content += `USE_COSINE_SIM = ${params.USE_COSINE_SIM}\n\n`;

    content += `[Model]\n`;
    content += `N_NCAS = ${params.N_NCAS}\n`;
    content += `CELL_STATE_DIM = ${params.CELL_STATE_DIM}\n`;
    content += `CELL_HIDDEN_DIM = ${params.CELL_HIDDEN_DIM}\n`;
    content += `HIDDEN_DIM = ${params.HIDDEN_DIM}\n`;
    content += `N_HIDDEN_LAYERS = ${params.N_HIDDEN_LAYERS}\n`;
    content += `KERNEL_SIZE = ${params.KERNEL_SIZE}\n`;
    content += `MODEL_DROPOUT_P = ${params.MODEL_DROPOUT_P}\n`;
    content += `OPTIMIZER_TYPE = ${params.OPTIMIZER_TYPE}\n`;
    content += `SUN_INIT_SCALE = ${params.SUN_INIT_SCALE}\n\n`;

    content += `[Concentration]\n`;
    content += `USE_CONCENTRATION = ${params.USE_CONCENTRATION}\n`;
    content += `CONCENTRATION_TEMP = ${params.CONCENTRATION_TEMP}\n`;
    content += `CONC_BASELINE = ${params.CONC_BASELINE}\n`;
    content += `CONCENTRATION_WINDOW = ${params.CONCENTRATION_WINDOW}\n\n`;

    content += `[Grid]\n`;
    content += `GRID_W = ${params.GRID_W}\n`;
    content += `GRID_H = ${params.GRID_H}\n`;
    content += `N_SEEDS = ${params.N_SEEDS}\n\n`;

    content += `[Visual]\n`;
    content += `COLOR_SCHEME = ${params.COLOR_SCHEME}\n\n`;

    content += `# JSON (for programmatic use):\n`;
    content += `# ${JSON.stringify(params)}\n`;

    // Create and trigger download
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`Parameters exported to ${filename}`);
}

/**
 * Triggers file input for loading params
 */
function triggerLoadParams() {
    let fileInput = document.getElementById('params-file-input');
    if (!fileInput) {
        fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.id = 'params-file-input';
        fileInput.accept = '.txt,.json';
        fileInput.style.display = 'none';
        fileInput.addEventListener('change', handleParamsFileSelect);
        document.body.appendChild(fileInput);
    }
    fileInput.click();
}

/**
 * Handles file selection for params loading
 */
function handleParamsFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target.result;
        loadParamsFromContent(content);
    };
    reader.readAsText(file);

    // Reset input so same file can be selected again
    event.target.value = '';
}

/**
 * Parse and apply params from file content
 */
function loadParamsFromContent(content) {
    let params = null;

    // Try to find JSON line in the file (comment line starting with # {)
    const jsonMatch = content.match(/^#\s*(\{.*\})\s*$/m);
    if (jsonMatch) {
        try {
            params = JSON.parse(jsonMatch[1]);
        } catch (e) {
            console.error('Failed to parse JSON from params file:', e);
        }
    }

    // Fallback: parse key = value lines
    if (!params) {
        params = {};
        const lines = content.split('\n');
        for (const line of lines) {
            const match = line.match(/^(\w+)\s*=\s*(.+)$/);
            if (match) {
                const key = match[1];
                let value = match[2].trim();
                // Parse value type
                if (value === 'true') value = true;
                else if (value === 'false') value = false;
                else if (!isNaN(parseFloat(value)) && isFinite(value)) value = parseFloat(value);
                params[key] = value;
            }
        }
    }

    if (!params || Object.keys(params).length === 0) {
        console.error('No valid parameters found in file');
        return;
    }

    // Check if architecture params changed (requires full reset)
    const architectureParams = ['N_NCAS', 'CELL_STATE_DIM', 'CELL_HIDDEN_DIM', 'HIDDEN_DIM', 'N_HIDDEN_LAYERS'];
    let needsReset = false;
    for (const key of architectureParams) {
        if (params[key] !== undefined && params[key] !== GUI_STATE[key]) {
            needsReset = true;
            break;
        }
    }

    // Apply params to GUI_STATE
    const appliedParams = [];
    for (const key in params) {
        if (key in GUI_STATE) {
            GUI_STATE[key] = params[key];
            appliedParams.push(key);
        }
    }

    console.log(`Loaded params: ${appliedParams.join(', ')}`);

    // Refresh GUI to show new values
    if (simState.gui) {
        simState.gui.destroy();
        simState.guiInitialized = false;
    }

    if (needsReset) {
        console.log('Architecture params changed - resetting simulation');
        resetSimulation(true);
    } else {
        // Just rebuild GUI without full reset
        setupGUI();
        setupNCAButtons();
        // Recreate optimizer with new learning rate
        simState.optimizer = createOptimizer();
    }
}

// =============================================================================
// TOOL MANAGEMENT
// =============================================================================

function setTool(toolName) {
    if (GUI_STATE.DRAW_TOOL === toolName) {
        GUI_STATE.DRAW_TOOL = 'none';
    } else {
        GUI_STATE.DRAW_TOOL = toolName;
    }
    updateToolStyles();
}

function updateToolStyles() {
    // dat.GUI buttons (Wall Pencil, Eraser)
    for (const toolName in simState.toolControllers) {
        const controller = simState.toolControllers[toolName];
        if (controller.domElement) {
            const row = controller.domElement.parentElement.parentElement;
            if (toolName === GUI_STATE.DRAW_TOOL) {
                row.classList.add('active-tool');
            } else {
                row.classList.remove('active-tool');
            }
        }
    }
    // Custom HTML buttons (Draw NCA species)
    const buttons = document.getElementsByClassName('nca-grid-button');
    for (let i = 0; i < buttons.length; i++) {
        const btn = buttons[i];
        if (btn.dataset.toolName === GUI_STATE.DRAW_TOOL) {
            btn.classList.add('selected-tool');
        } else {
            btn.classList.remove('selected-tool');
        }
    }
}

// =============================================================================
// CONTROL TOOLTIPS
// =============================================================================

const CONTROL_TOOLTIPS = {
    // Showcase
    'Load Showcase': 'Load a pre-trained ecosystem with 5 species in a stable, interesting equilibrium. Great starting point for exploration.',
    // Quick Settings
    'Species': 'Number of competing NCA species. Each has its own neural network. More species = richer dynamics but slower.',
    'Grid Size': 'Width and height of the simulation grid in cells. Larger grids allow more complex spatial patterns.',
    'Competition': 'global = O(N) mean-field approximation (fast, scales to many species). full = O(N²) pairwise competition (accurate, slow for N>10).',
    'Color Scheme': 'Visual color palette for species. Does not affect simulation dynamics.',

    // Simulation
    'Sharpness': 'Softmax temperature for competition. Low (<0.3) = winner-take-all, aggressive. High (>2) = permissive coexistence. Default 0.57 balances competition and diversity.',
    'Survival': 'Alive threshold. Cells with aliveness below this are considered dead. Low = fuzzy boundaries, High = crisp territories. Interacts with sharpness.',
    'Learn Interval': 'Simulation steps between gradient updates. 1 = learn every step (slow but adaptive). >100 = inference only (fast, no learning). Higher = less adaptation.',
    'Learn Rate': 'Step size for neural network weight updates. Too low = slow adaptation. Too high = unstable/chaotic. Default 0.0007.',
    'Sun LR Scale': "Multiplier on the learning rate applied to the sun's update parameters. 0 = freeze the sun (constant background, like the original PDNCA). 1 = train at the same rate as the species. >1 = train the sun faster. Use this to dial the synchronized 'breathing' oscillation up/down at high survival thresholds.",
    'Local Sun': 'Per-cell learnable sun. Each grid location gets its own sun parameter vector, creating a learnable spatial environment. OFF = single global sun (default). Switching ON tiles the current global sun to all cells; switching OFF averages local values back to a single global vector.',
    'Growth Gate Steepness': 'Sharpness of the soft alive-threshold gate inside getAliveMask. Default 20 (very sharp, like a step function). Lower (5-10) widens the transition zone and reduces cascade collapses at high survival thresholds. Use to test whether the threshold cliff drives the breathing oscillation.',
    'Frame Delay': 'Milliseconds between simulation steps. 0 = maximum speed. Increase to slow down visualization.',
    'Update Prob': 'Probability each cell updates per step. <1.0 adds stochastic noise, breaking synchronous artifacts. Lower = more temporal variation.',
    'Local Win Boost': 'Enable per-cell win-rate tracking. Cells where one species dominates update more often, stabilizing territories.',
    'Win Boost Strength': 'How strongly the local win rate affects update probability. 0 = no effect. Higher = stronger positive feedback for dominant cells.',
    'Win Rate Memory': 'EMA decay for win rate tracking. Higher (0.99) = long memory, stable. Lower (0.8) = responsive to recent changes.',

    // Advanced
    'Model Depth': 'Number of InvertedResidual blocks in the neural network. More = higher capacity but slower. Requires reset.',
    'Model Width': 'Hidden dimension of the neural network. Wider = more parameters, richer strategies. Requires reset.',
    'Cell State Dim': 'Dimensions for attack + defense vectors (split 50/50). Larger = more expressive combat strategies. Requires reset.',
    'Cell Memory': 'Extra persistent hidden dimensions per cell. 0 = memoryless (state determined purely by neighbors). >0 = cells remember. Requires reset.',
    'Kernel Size': 'Convolution kernel size. 1 = self only. 3 = immediate neighbors. 5+ = wider perception. Odd values only. Requires reset.',
    'Sun Strength': 'Initial magnitude of the sun\'s (background) attack/defense vectors. Higher = stronger baseline competitor. Requires reset.',
    'Dropout': 'Training regularization. 0 = off. Higher = more noise during learning, may prevent overfitting but slows convergence.',
    'Optimizer': 'Training algorithm. SGD = simple, low memory. Adam = adaptive learning rates, faster convergence. SGD+Momentum = balanced.',
    'Focus Weak': 'Soft-min sharpness (k). Higher = loss focuses more on the weakest species. Prevents strong species from ignoring weak ones.',
    'Growth Curve': 'Asinh slope parameter (α). Controls how population size maps to reward. Lower = diminishing returns for large populations.',
    'Diversity': 'Diversity incentive strength. Adds entropy bonus to loss + competition score boost for minority species. 0 = pure competition. Higher = forces coexistence.',
    'Train Region': 'Percentage of grid used for gradient computation. 10% = fast but noisy. 100% = accurate but slow.',
    'Sun Penalty': 'Handicap applied to sun near NCA cells. Higher = NCAs expand more easily into empty space. 0 = sun competes fairly.',
    'Min Occupancy': 'Emergency respawn threshold (cell count). Species below this get fresh seeds injected. 0 = no safety net, species can go fully extinct.',
    'Cosine Similarity': 'ON = normalized cosine similarity for attack vs defense scoring (direction matters). OFF = raw dot product (magnitude matters too).',

    // Concentration
    'Concentration': 'Enable spatial concentration mechanism. Species focus competitive energy on high-activity cells via local softmax, creating territorial focus.',
    'Conc. Temp': 'Concentration temperature. Low (<0.3) = sharp focus on strongest cells. High (>2) = diffuse, nearly uniform. Controls territorial compactness.',
    'Conc. Baseline': 'Minimum concentration weight. 0 = pure softmax (weak cells fully suppressed). Higher = ensures all cells retain some competitive presence.',
    'Conc. Window': 'Size of the local window for concentration softmax. Smaller = local focus. Larger = broader spatial averaging.',

    // Grid
    'Seeds per species': 'Number of initial random seed points for each species. More seeds = faster initial coverage.',
    'Add Seeds': 'Inject new random seed points for all species into the current grid.',

    // Drawing
    'Wall Pencil': 'Draw walls that permanently block species growth. Creates spatial barriers and niches.',
    'Eraser': 'Erase walls and clear all NCA state in the brush area. Cells revert to sun (empty).',
    'Tool Size': 'Brush radius for drawing tools (in grid cells).',

    // Buttons
    'Start': 'Begin or resume the simulation.',
    'Pause': 'Pause the simulation. Grid state is preserved.',
    'Reset': 'Reset grid, model weights, and optimizer. Keeps current settings.',
    'Export Params': 'Download current hyperparameter settings as a text file.',
    'Load Params': 'Load hyperparameter settings from a previously exported text file.',
    'Export Run (.petri)': 'Download the full run state (grid, weights, metrics, checkpoints) as a .petri file for sharing.',
    'Import Run (.petri)': 'Load a previously exported .petri run file, restoring the full simulation state.',
    'Copy Recipe URL': 'Copy a compact URL encoding the seed, settings, and parameter changes for approximate reproduction.',
};

/**
 * Adds a tooltip icon to a dat.GUI controller (inserted at LEFT of label)
 */
function addTooltip(controller, tooltipText) {
    if (!controller || !tooltipText) return;

    // Find the row element
    const propertyRow = controller.domElement?.closest('.cr') || controller.__li;
    if (!propertyRow) return;

    // Skip function buttons (Start, Reset, etc.) - they don't need tooltips on the label
    if (propertyRow.classList.contains('function')) return;

    // Find the property-name element
    const propertyName = propertyRow.querySelector('.property-name');
    if (!propertyName) return;

    // Check if tooltip already exists
    if (propertyName.querySelector('.tooltip-icon')) return;

    const tooltip = document.createElement('span');
    tooltip.className = 'tooltip-icon';
    tooltip.textContent = '?';

    const tooltipTextEl = document.createElement('span');
    tooltipTextEl.className = 'tooltip-text';
    tooltipTextEl.textContent = tooltipText;
    tooltip.appendChild(tooltipTextEl);

    // Position tooltip above the control row on hover
    tooltip.addEventListener('mouseenter', () => {
        const rowRect = propertyRow.getBoundingClientRect();
        tooltipTextEl.style.bottom = `${window.innerHeight - rowRect.top + 8}px`;
        tooltipTextEl.style.left = `${rowRect.left + 10}px`;
    });

    // Insert at the BEGINNING of property-name (left side)
    propertyName.insertBefore(tooltip, propertyName.firstChild);
}

/**
 * Updates the color scheme preview bar with colored squares
 */
function updateColorSchemePreview(schemeName) {
    const previewBar = document.getElementById('color-scheme-preview');
    if (!previewBar) return;

    // Get the color scheme from petri-dish.js
    const schemes = {
        'Vibrant': [[0.349, 0.784, 0.980], [0.976, 0.341, 0.486], [0.200, 0.843, 0.659], [0.992, 0.702, 0.180], [0.647, 0.443, 0.976], [1.000, 0.502, 0.314], [0.000, 0.820, 0.878], [0.957, 0.263, 0.678]],
        'Neon': [[0.000, 1.000, 0.875], [1.000, 0.000, 0.500], [0.498, 1.000, 0.000], [1.000, 0.800, 0.000], [0.878, 0.000, 1.000], [1.000, 0.400, 0.000], [0.000, 0.600, 1.000], [1.000, 0.000, 0.200]],
        'Pastel': [[0.678, 0.847, 0.902], [1.000, 0.714, 0.757], [0.596, 0.984, 0.596], [1.000, 0.937, 0.678], [0.800, 0.600, 0.800], [1.000, 0.800, 0.600], [0.600, 0.800, 0.800], [1.000, 0.600, 0.678]],
        'Earth': [[0.545, 0.353, 0.169], [0.824, 0.706, 0.549], [0.333, 0.420, 0.184], [0.871, 0.722, 0.529], [0.627, 0.322, 0.176], [0.557, 0.537, 0.286], [0.467, 0.533, 0.600], [0.804, 0.522, 0.247]],
        'Ocean': [[0.000, 0.467, 0.745], [0.118, 0.565, 0.604], [0.259, 0.808, 0.902], [0.000, 0.392, 0.392], [0.275, 0.510, 0.706], [0.498, 0.753, 0.753], [0.125, 0.698, 0.667], [0.282, 0.239, 0.545]],
        'Sunset': [[0.988, 0.369, 0.259], [1.000, 0.549, 0.000], [0.988, 0.753, 0.286], [0.933, 0.510, 0.510], [0.867, 0.267, 0.467], [0.800, 0.200, 0.200], [1.000, 0.388, 0.278], [0.941, 0.502, 0.502]],
        'Candy': [[1.000, 0.412, 0.706], [0.678, 0.847, 0.902], [0.933, 0.510, 0.933], [0.498, 1.000, 0.831], [1.000, 0.753, 0.796], [0.686, 0.933, 0.933], [0.855, 0.439, 0.839], [0.596, 0.984, 0.596]],
        'Forest': [[0.133, 0.545, 0.133], [0.604, 0.804, 0.196], [0.420, 0.557, 0.137], [0.545, 0.271, 0.075], [0.498, 0.498, 0.000], [0.333, 0.420, 0.184], [0.180, 0.545, 0.341], [0.565, 0.933, 0.565]],
        'Retro': [[1.000, 0.078, 0.576], [0.000, 1.000, 1.000], [1.000, 0.843, 0.000], [0.580, 0.000, 0.827], [1.000, 0.271, 0.000], [0.118, 0.565, 1.000], [0.698, 0.133, 0.133], [0.000, 0.808, 0.820]],
        'Mono Blue': [[0.118, 0.565, 1.000], [0.255, 0.412, 0.882], [0.000, 0.749, 1.000], [0.275, 0.510, 0.706], [0.529, 0.808, 0.922], [0.000, 0.502, 0.502], [0.282, 0.239, 0.545], [0.392, 0.584, 0.929]],
        'Rainbow': [[1.000, 0.000, 0.000], [1.000, 0.498, 0.000], [1.000, 1.000, 0.000], [0.000, 1.000, 0.000], [0.000, 1.000, 1.000], [0.000, 0.000, 1.000], [0.545, 0.000, 0.545], [1.000, 0.000, 1.000]],
        'High Contrast': [[1.000, 0.000, 0.000], [0.000, 0.000, 1.000], [1.000, 1.000, 0.000], [0.000, 1.000, 0.000], [1.000, 0.000, 1.000], [0.000, 1.000, 1.000], [1.000, 0.500, 0.000], [0.500, 0.000, 1.000]],
    };

    const colors = schemes[schemeName] || schemes['Vibrant'];
    previewBar.innerHTML = '';

    colors.forEach(rgb => {
        const swatch = document.createElement('div');
        swatch.style.cssText = `
            width: 18px;
            height: 12px;
            border-radius: 2px;
            background: rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)});
            flex-shrink: 0;
        `;
        previewBar.appendChild(swatch);
    });
}

/**
 * Adds tooltips to all controllers in a folder
 */
/**
 * For every numeric controller in a dat.GUI folder, inject a pair of small
 * up/down arrow buttons next to the number input for fine-grained adjustment.
 * The step size defaults to the controller's own step; if no step is set we
 * fall back to 0.001 for [0,1]-ranged controls, 1 for integer-looking ones.
 */
function addStepArrowsToFolder(folder) {
    if (!folder || !folder.__controllers) return;
    folder.__controllers.forEach(controller => {
        // Only numeric controllers. dat.GUI sets __min/__max on Number/NumberSlider.
        if (typeof controller.getValue !== 'function') return;
        if (typeof controller.__min !== 'number' || typeof controller.__max !== 'number') return;
        const sample = controller.getValue();
        if (typeof sample !== 'number') return;

        const row = controller.__li || controller.domElement?.closest('.cr');
        if (!row) return;
        if (row.querySelector('.step-arrows')) return;  // already attached

        const rawStep = controller.__step;
        const min = controller.__min;
        const max = controller.__max;
        const step = (typeof rawStep === 'number' && rawStep > 0)
            ? rawStep
            : (max - min <= 1.0 ? 0.001 : 1);

        const arrows = document.createElement('span');
        arrows.className = 'step-arrows';
        // Vertically anchor to the number input's midline by computing its offset
        // inside the row on first layout.
        arrows.style.cssText = 'position:absolute;right:2px;display:flex;flex-direction:column;gap:1px;z-index:10;pointer-events:auto;';

        const makeBtn = (label, delta, title) => {
            const b = document.createElement('button');
            b.type = 'button';
            b.className = 'step-arrow';
            b.textContent = label;
            b.title = title;
            b.style.cssText = 'background:#1a1a1a;border:1px solid #555;color:#ddd;border-radius:2px;width:14px;height:10px;line-height:1;font-size:7px;padding:0;margin:0;cursor:pointer;display:flex;align-items:center;justify-content:center;font-family:monospace;';
            b.addEventListener('mousedown', (e) => { e.preventDefault(); e.stopPropagation(); });
            b.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                let v = controller.getValue();
                if (typeof v !== 'number') return;
                v += delta;
                v = Math.max(min, Math.min(max, v));
                const decimals = Math.max(0, -Math.floor(Math.log10(step)));
                v = parseFloat(v.toFixed(decimals));
                controller.setValue(v);
            });
            return b;
        };
        arrows.appendChild(makeBtn('\u25B2', +step, `+${step}`));
        arrows.appendChild(makeBtn('\u25BC', -step, `-${step}`));

        // Make room: shrink the number-input box and make the row a positioning context.
        // Also give the row right-padding so the absolute-positioned arrows don't overlap the number text.
        if (getComputedStyle(row).position === 'static') row.style.position = 'relative';
        row.style.paddingRight = '20px';
        row.appendChild(arrows);

        // Vertically centre the arrows on the number input box (not on the whole row).
        // Run after layout so offsetTop / offsetHeight are valid.
        requestAnimationFrame(() => {
            const input = row.querySelector('input[type="text"], input[type="number"]');
            const target = input || row.querySelector('.c');
            if (!target) return;
            const inputTop = target.offsetTop;
            const inputHeight = target.offsetHeight;
            const arrowsHeight = arrows.offsetHeight;
            arrows.style.top = `${inputTop + (inputHeight - arrowsHeight) / 2}px`;
        });
    });
}

function addTooltipsToFolder(folder) {
    if (!folder || !folder.__controllers) return;

    folder.__controllers.forEach(controller => {
        // Get the row element
        const row = controller.__li || controller.domElement?.closest('.cr');
        if (!row) return;

        // Try to find the display name from property-name element
        const propertyNameEl = row.querySelector('.property-name');
        const displayName = propertyNameEl?.textContent?.trim() || '';

        if (!displayName) return;

        // Try to find tooltip - check exact match first, then partial matches
        let tooltipText = CONTROL_TOOLTIPS[displayName];

        if (!tooltipText) {
            // Try matching by prefix (handles dynamic labels like "Steps per Learn: 5")
            for (const key in CONTROL_TOOLTIPS) {
                if (displayName.startsWith(key) || key.startsWith(displayName.split(':')[0]?.trim())) {
                    tooltipText = CONTROL_TOOLTIPS[key];
                    break;
                }
            }
        }

        if (tooltipText) {
            addTooltip(controller, tooltipText);
        }
    });
}

// =============================================================================
// GUI SETUP
// =============================================================================

function setupGUI() {
    if (simState.gui) {
        simState.gui.destroy();
    }
    simState.gui = new dat.GUI({ width: 280, autoPlace: false });  // Manual placement
    // Append to gui-container instead of left-panel directly
    const guiContainer = document.getElementById('gui-container');
    if (guiContainer) {
        guiContainer.appendChild(simState.gui.domElement);
    }

    // Calculate max grid dimension based on GPU limits
    try {
        const gl = document.getElementById('c').getContext('webgl2');
        if (gl) {
            const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
            const packedPixels = Math.ceil((GUI_STATE.CELL_STATE_DIM + GUI_STATE.CELL_HIDDEN_DIM + GUI_STATE.N_NCAS + 1) / 4);
            simState.maxGridDim = Math.floor(maxTexSize / packedPixels);
        }
    } catch (e) {
        console.error("Could not get WebGL context.", e);
    }

    const resetSimOnChange = () => resetSimulation(true);

    // --- Main Controls (paper-relevant, default open) ---
    const mainFolder = simState.gui.addFolder('Main');
    simState.startPauseController = mainFolder.add(GUI_BUTTONS, 'startPause').name('Start');
    const resetController = mainFolder.add(GUI_BUTTONS, 'hardReset').name('Reset');
    const showcaseController = mainFolder.add(GUI_BUTTONS, 'loadShowcase').name('LOAD SHOWCASE');
    simState.startPauseController.domElement.parentElement.parentElement.classList.add('btn-start');
    resetController.domElement.parentElement.parentElement.classList.add('btn-reset');
    const showcaseRow = showcaseController.domElement.parentElement.parentElement;
    showcaseRow.classList.add('btn-showcase', 'btn-showcase-glow');

    // Kill showcase glow on ANY user interaction
    function stopShowcaseGlow() {
        const el = document.querySelector('.btn-showcase-glow');
        if (el) {
            el.classList.remove('btn-showcase-glow');
            // Revert to normal text style
            const nameEl = el.querySelector('.property-name');
            if (nameEl) nameEl.textContent = 'Load Showcase';
        }
    }
    window.stopShowcaseGlow = stopShowcaseGlow; // expose for GUI_BUTTONS
    document.addEventListener('keydown', stopShowcaseGlow, { once: true });
    document.addEventListener('mousedown', stopShowcaseGlow, { once: true });

    mainFolder.add(GUI_STATE, 'N_NCAS', 2, 6).step(1).name('Species')
        .onFinishChange(() => resetSimulation(true));
    mainFolder.add(GUI_STATE, 'GRID_W', 32, 256).step(8).name('Grid Size')
        .onFinishChange((val) => {
            GUI_STATE.GRID_H = val;
            resetSimulation(false);
        });
    const sharpnessCtrl = mainFolder.add(GUI_STATE, 'SOFTMAX_TEMP', 0.1, 6).step(0.001).name('Sharpness');
    sharpnessCtrl.domElement.parentElement.parentElement.classList.add('key-param');
    const survivalCtrl = mainFolder.add(GUI_STATE, 'ALIVE_THRESHOLD', 0.0, 1.0).step(0.001).name('Survival');
    survivalCtrl.domElement.parentElement.parentElement.classList.add('key-param');
    const growthGateCtrl = mainFolder.add(GUI_STATE, 'GROWTH_GATE_STEEPNESS', 1, 25).step(0.1).name('Growth Gate');
    growthGateCtrl.domElement.parentElement.parentElement.classList.add('key-param');
    mainFolder.add(GUI_STATE, 'N_HIDDEN_LAYERS', 0, 10).step(1)
        .name('Model Depth').onFinishChange(resetSimOnChange);
    mainFolder.add(GUI_STATE, 'HIDDEN_DIM', 8, 256).step(8)
        .name('Model Width').onFinishChange(resetSimOnChange);
    mainFolder.add(GUI_STATE, 'CELL_STATE_DIM', 4, 128).step(2)
        .name('Cell State Dim').onFinishChange(resetSimOnChange);
    mainFolder.add(GUI_STATE, 'KERNEL_SIZE', [3, 5, 7])
        .name('Kernel Size').onChange(resetSimOnChange);
    mainFolder.add(GUI_STATE, 'OPTIMIZER_TYPE', ['Adam', 'SGD', 'SGD+Momentum', 'RMSProp'])
        .name('Optimizer')
        .onFinishChange(() => { simState.optimizer = createOptimizer(); });
    mainFolder.add(GUI_STATE, 'LEARNING_RATE', 0.0001, 0.01).step(0.00005).name('Learn Rate');
    mainFolder.open();

    // --- Advanced (everything else, collapsed) ---
    const advancedFolder = simState.gui.addFolder('Advanced');

    // Diagnostic / oscillation knobs (top of Advanced)
    advancedFolder.add(GUI_STATE, 'SUN_LR_SCALE', 0, 10).step(0.001).name('Sun LR Scale');
    const localSunController = advancedFolder.add(GUI_STATE, 'SUN_MODE', ['global', 'spatial', 'seasonal']).name('Sun Mode')
        .onChange(newMode => {
            setSunMode(newMode);
            const isLocal = newMode !== 'global';
            updateSunHeatmapVisibility(isLocal);
            updateSeasonalPanelVisibility(newMode === 'seasonal');
            if (isLocal) updateSunHeatmap();
            if (newMode === 'seasonal') updateSeasonalPanel();
        });
    // Red tint + EXPERIMENTAL warning on the Sun Mode row
    {
        const sunModeLi = localSunController.domElement.parentElement.parentElement;
        sunModeLi.id = 'sun-mode-li';
        sunModeLi.classList.add('sun-tinted-row');
        const propertyRow = localSunController.domElement.closest('.cr') || sunModeLi;
        const nameCell = propertyRow.querySelector('.property-name');
        if (nameCell && !nameCell.querySelector('.sun-warn-badge')) {
            const warnText = 'EXPERIMENTAL: spatial/seasonal sun modes are untested research features and may destabilise the simulation or fail to converge.';
            const warn = document.createElement('span');
            warn.className = 'tooltip-icon sun-warn-badge';
            warn.textContent = '!';
            const warnTextEl = document.createElement('span');
            warnTextEl.className = 'tooltip-text';
            warnTextEl.textContent = warnText;
            warn.appendChild(warnTextEl);
            warn.addEventListener('mouseenter', () => {
                const rowRect = propertyRow.getBoundingClientRect();
                warnTextEl.style.bottom = `${window.innerHeight - rowRect.top + 8}px`;
                warnTextEl.style.left = `${rowRect.left + 10}px`;
            });
            nameCell.appendChild(warn);
        }
    }

    // --- Sun divergence heatmap (visible only when Local Sun is on) ---
    {
        const heatmapLi = document.createElement('li');
        heatmapLi.style.cssText = 'padding:4px 8px; background:transparent; height:auto; overflow:visible; border:none;';
        heatmapLi.id = 'sun-heatmap-li';
        heatmapLi.classList.add('sun-tinted-row');
        heatmapLi.style.display = (GUI_STATE.SUN_MODE !== 'global') ? 'block' : 'none';

        const label = document.createElement('div');
        label.style.cssText = 'font-size:10px; color:#888; margin-bottom:3px; font-family:"JetBrains Mono",monospace;';
        label.textContent = 'Sun divergence from mean';
        heatmapLi.appendChild(label);

        const heatmapCanvas = document.createElement('canvas');
        heatmapCanvas.id = 'sun-heatmap-canvas';
        heatmapCanvas.style.cssText = 'width:100%; image-rendering:pixelated; border:1px solid #333; border-radius:3px; display:block; cursor:crosshair;';
        heatmapCanvas.addEventListener('mouseenter', showSunOverlay);
        heatmapCanvas.addEventListener('mouseleave', hideSunOverlay);
        heatmapLi.appendChild(heatmapCanvas);

        // Stats line under the heatmap
        const statsLine = document.createElement('div');
        statsLine.id = 'sun-heatmap-stats';
        statsLine.style.cssText = 'font-size:9px; color:#666; margin-top:2px; font-family:"JetBrains Mono",monospace;';
        statsLine.textContent = 'variance: —';
        heatmapLi.appendChild(statsLine);

        // Insert after the Local Sun controller
        const localSunLi = localSunController.domElement.parentElement.parentElement;
        const innerUl = advancedFolder.domElement.querySelector('ul');
        if (localSunLi.nextSibling) {
            innerUl.insertBefore(heatmapLi, localSunLi.nextSibling);
        } else {
            innerUl.appendChild(heatmapLi);
        }
    }

    // --- Heatmap refresh rate slider (visible only with heatmap) ---
    const heatmapRateCtrl = advancedFolder.add(GUI_STATE, 'HEATMAP_UPDATE_EVERY', 1, 200).step(1).name('Heatmap Every');
    {
        const rowLi = heatmapRateCtrl.domElement.parentElement.parentElement;
        rowLi.id = 'sun-heatmap-rate-li';
        rowLi.classList.add('sun-tinted-row');
        const heatmapLi = document.getElementById('sun-heatmap-li');
        const innerUl = advancedFolder.domElement.querySelector('ul');
        if (heatmapLi && heatmapLi.nextSibling) {
            innerUl.insertBefore(rowLi, heatmapLi.nextSibling);
        } else if (heatmapLi) {
            innerUl.appendChild(rowLi);
        }
        rowLi.style.display = GUI_STATE.SUN_MODE !== 'global' ? 'block' : 'none';
    }

    // --- Overlay blend options (visible only with heatmap) ---
    {
        const blendLi = document.createElement('li');
        blendLi.id = 'sun-overlay-blend-li';
        blendLi.classList.add('sun-tinted-row');
        blendLi.style.cssText = 'padding:6px 8px; background:transparent; height:auto; overflow:visible; border:none; display:' + (GUI_STATE.SUN_MODE !== 'global' ? 'block' : 'none') + ';';

        const label = document.createElement('div');
        label.style.cssText = 'font-size:10px; color:#888; margin-bottom:3px; font-family:"JetBrains Mono",monospace;';
        label.textContent = 'Overlay blend (on hover)';
        blendLi.appendChild(label);

        const modeRow = document.createElement('div');
        modeRow.style.cssText = 'display:flex; align-items:center; gap:6px; margin-bottom:4px;';
        const modeLabel = document.createElement('span');
        modeLabel.style.cssText = 'font-size:10px; color:#aaa; width:52px; font-family:"JetBrains Mono",monospace;';
        modeLabel.textContent = 'mode';
        const modeSel = document.createElement('select');
        modeSel.id = 'sun-overlay-blend-mode';
        modeSel.style.cssText = 'flex:1; font-size:10px; background:#111; color:#ddd; border:1px solid #333; border-radius:2px; padding:2px 4px; font-family:"JetBrains Mono",monospace;';
        ['screen', 'normal', 'multiply', 'overlay', 'hard-light', 'soft-light', 'lighten', 'difference'].forEach(m => {
            const opt = document.createElement('option');
            opt.value = m; opt.textContent = m;
            if (m === 'screen') opt.selected = true;
            modeSel.appendChild(opt);
        });
        modeSel.addEventListener('change', () => {
            const ov = document.getElementById('sun-overlay-canvas');
            if (ov) ov.style.mixBlendMode = modeSel.value;
        });
        modeRow.appendChild(modeLabel);
        modeRow.appendChild(modeSel);
        blendLi.appendChild(modeRow);

        const opRow = document.createElement('div');
        opRow.style.cssText = 'display:flex; align-items:center; gap:6px;';
        const opLabel = document.createElement('span');
        opLabel.style.cssText = 'font-size:10px; color:#aaa; width:52px; font-family:"JetBrains Mono",monospace;';
        opLabel.textContent = 'opacity';
        const opInput = document.createElement('input');
        opInput.type = 'range'; opInput.min = '0'; opInput.max = '1'; opInput.step = '0.01'; opInput.value = '0.75';
        opInput.id = 'sun-overlay-blend-opacity';
        opInput.style.cssText = 'flex:1;';
        const opVal = document.createElement('span');
        opVal.style.cssText = 'font-size:10px; color:#ccc; width:34px; text-align:right; font-family:"JetBrains Mono",monospace;';
        opVal.textContent = '0.75';
        opInput.addEventListener('input', () => {
            const ov = document.getElementById('sun-overlay-canvas');
            if (ov) ov.style.opacity = opInput.value;
            opVal.textContent = parseFloat(opInput.value).toFixed(2);
        });
        opRow.appendChild(opLabel);
        opRow.appendChild(opInput);
        opRow.appendChild(opVal);
        blendLi.appendChild(opRow);

        const innerUl = advancedFolder.domElement.querySelector('ul');
        const rateLi = document.getElementById('sun-heatmap-rate-li');
        if (rateLi && rateLi.nextSibling) {
            innerUl.insertBefore(blendLi, rateLi.nextSibling);
        } else if (rateLi) {
            innerUl.appendChild(blendLi);
        }
    }

    // --- Seasonal period read-out (visible only when Sun Mode == seasonal) ---
    {
        const seasonLi = document.createElement('li');
        seasonLi.id = 'sun-seasonal-li';
        seasonLi.style.cssText = 'padding:4px 8px; background:transparent; height:auto; overflow:visible; border:none;';
        seasonLi.classList.add('sun-tinted-row');
        seasonLi.style.display = (GUI_STATE.SUN_MODE === 'seasonal') ? 'block' : 'none';

        const label = document.createElement('div');
        label.style.cssText = 'font-size:10px; color:#888; margin-bottom:3px; font-family:"JetBrains Mono",monospace;';
        label.textContent = 'Learned seasonal periods (sim steps)';
        seasonLi.appendChild(label);

        const list = document.createElement('pre');
        list.id = 'sun-seasonal-periods';
        list.style.cssText = 'font-size:10px; color:#ccc; margin:0; padding:6px 8px; background:#0a0a0a; border:1px solid #333; border-radius:3px; font-family:"JetBrains Mono",monospace; line-height:1.3;';
        list.textContent = '—';
        seasonLi.appendChild(list);

        const innerUl = advancedFolder.domElement.querySelector('ul');
        const anchor = document.getElementById('sun-overlay-blend-li')
            || document.getElementById('sun-heatmap-li')
            || localSunController.domElement.parentElement.parentElement;
        if (anchor.nextSibling) {
            innerUl.insertBefore(seasonLi, anchor.nextSibling);
        } else {
            innerUl.appendChild(seasonLi);
        }
    }

    advancedFolder.add(GUI_STATE, 'COMPETITION_MODE', ['global', 'full']).name('Competition')
        .onFinishChange(() => {
            console.log(`Competition mode: ${GUI_STATE.COMPETITION_MODE}`);
        });
    const colorSchemeController = advancedFolder.add(GUI_STATE, 'COLOR_SCHEME', ['Vibrant', 'Neon', 'Pastel', 'Earth', 'Ocean', 'Sunset', 'Candy', 'Forest', 'Retro', 'Mono Blue', 'Rainbow', 'High Contrast'])
        .name('Color Scheme')
        .onFinishChange((scheme) => {
            setColorScheme(scheme);
            simState.nca_colors = generateNCAColors(CONFIG.N_NCAS);
            simState.color_uniform = simState.nca_colors.flat();
            simState.renderShader.colors = simState.color_uniform;
            setupNCAButtons();
            updateColorSchemePreview(scheme);
        });

    // Add color preview bar below the color scheme selector
    setTimeout(() => {
        const row = colorSchemeController.domElement?.closest('.cr');
        if (row) {
            const previewBar = document.createElement('div');
            previewBar.id = 'color-scheme-preview';
            previewBar.style.cssText = `
                display: flex;
                gap: 2px;
                padding: 4px 8px;
                margin: 4px 8px 8px 8px;
                background: var(--bg-dark);
                border-radius: 4px;
            `;
            row.parentNode.insertBefore(previewBar, row.nextSibling);
            updateColorSchemePreview(GUI_STATE.COLOR_SCHEME);
        }
    }, 50);

    // Learning Balance: sim steps between gradient updates
    const balanceController = advancedFolder.add(GUI_STATE, 'LEARNING_BALANCE', 1, 101).step(1)
        .name('Learn Interval');
    const updateBalanceLabel = (value) => {
        if (value > 100) {
            balanceController.name('Learn: OFF');
        } else {
            balanceController.name(`Learn Interval: ${value}`);
        }
    };
    balanceController.onChange(updateBalanceLabel);
    updateBalanceLabel(GUI_STATE.LEARNING_BALANCE);

    advancedFolder.add(GUI_STATE, 'UPDATE_DELAY_MS', 0, 500).step(10).name('Frame Delay');
    advancedFolder.add(GUI_STATE, 'STOCHASTIC_UPDATE_PROB', 0.1, 1.0).step(0.001).name('Update Prob');
    advancedFolder.add(GUI_STATE, 'USE_LOCAL_WIN_RATE').name('Local Win Boost');
    advancedFolder.add(GUI_STATE, 'LOCAL_WIN_RATE_STRENGTH', 0, 2).step(0.01).name('Win Boost Strength');
    advancedFolder.add(GUI_STATE, 'LOCAL_WIN_RATE_ALPHA', 0.8, 0.99).step(0.001).name('Win Rate Memory');

    advancedFolder.add(GUI_STATE, 'CELL_HIDDEN_DIM', 0, 64).step(2)
        .name('Cell Memory').onFinishChange(resetSimOnChange);
    advancedFolder.add(GUI_STATE, 'SUN_INIT_SCALE', 0.1, 1.0).step(0.01)
        .name('Sun Strength').onFinishChange(resetSimOnChange);
    advancedFolder.add(GUI_STATE, 'MODEL_DROPOUT_P', 0.0, 0.5).step(0.01).name('Dropout');
    advancedFolder.add(GUI_STATE, 'SOFT_MIN_K', 1, 50).step(0.1).name('Focus Weak');
    advancedFolder.add(GUI_STATE, 'ASINH_SLOPE', 0.1, 5.0).step(0.01).name('Growth Curve');
    advancedFolder.add(GUI_STATE, 'DIVERSITY_WEIGHT', 0, 2.0).step(0.01).name('Diversity');
    advancedFolder.add(GUI_STATE, 'OPTIMIZATION_PERCENT', 10, 100).step(10).name('Train Region');
    advancedFolder.add(GUI_STATE, 'SUN_PENALTY', 0, 10).step(0.01).name('Sun Penalty');
    advancedFolder.add(GUI_STATE, 'MIN_OCCUPANCY', 0, 100).step(1).name('Min Occupancy');
    advancedFolder.add(GUI_STATE, 'USE_COSINE_SIM').name('Cosine Similarity');
    advancedFolder.add(GUI_STATE, 'USE_CONCENTRATION').name('Concentration');
    advancedFolder.add(GUI_STATE, 'CONCENTRATION_TEMP', 0.1, 5.0).step(0.01).name('Conc. Temp');
    advancedFolder.add(GUI_STATE, 'CONC_BASELINE', 0.0, 0.5).step(0.01).name('Conc. Baseline');
    advancedFolder.add(GUI_STATE, 'CONCENTRATION_WINDOW', 3, 21).step(2).name('Conc. Window');

    advancedFolder.add(GUI_STATE, 'N_SEEDS', 1, 20).step(1).name('Seeds per species');
    const reseedController = advancedFolder.add(GUI_BUTTONS, 'reseed').name('Add Seeds');
    reseedController.domElement.parentElement.parentElement.classList.add('btn-reseed');

    // --- Drawing Controls (in Advanced) ---
    // The drawFolder reference is kept for setupNCAButtons() but points into Advanced
    simState.drawFolder = advancedFolder;
    simState.toolControllers = {};
    simState.toolControllers['pencil'] = advancedFolder.add(GUI_BUTTONS, 'selectPencil').name('Wall Pencil');
    simState.toolControllers['eraser'] = advancedFolder.add(GUI_BUTTONS, 'selectEraser').name('Eraser');
    simState.toolControllers['pencil'].domElement.parentElement.parentElement.classList.add('btn-tool');
    simState.toolControllers['eraser'].domElement.parentElement.parentElement.classList.add('btn-tool');
    // NCA draw buttons will be inserted before Tool Size by setupNCAButtons()
    simState.toolSizeController = advancedFolder.add(GUI_STATE, 'DRAW_SIZE', 1, 20).step(1).name('Tool Size');

    // --- Export/Import buttons (in Advanced) ---
    const exportController = advancedFolder.add(GUI_BUTTONS, 'exportParams').name('Export Params');
    exportController.domElement.parentElement.parentElement.classList.add('btn-export');
    const loadController = advancedFolder.add(GUI_BUTTONS, 'loadParams').name('Load Params');
    loadController.domElement.parentElement.parentElement.classList.add('btn-tool');
    const exportRunCtrl = advancedFolder.add(GUI_BUTTONS, 'exportRun').name('Export Run (.petri)');
    exportRunCtrl.domElement.parentElement.parentElement.classList.add('btn-export');
    const importRunCtrl = advancedFolder.add(GUI_BUTTONS, 'importRun').name('Import Run (.petri)');
    importRunCtrl.domElement.parentElement.parentElement.classList.add('btn-tool');
    const recipeCtrl = advancedFolder.add(GUI_BUTTONS, 'copyRecipe').name('Copy Recipe URL');
    recipeCtrl.domElement.parentElement.parentElement.classList.add('btn-tool');

    advancedFolder.add(GUI_STATE, 'SHOW_FPS').name('Show FPS')
        .onChange(val => {
            const el = document.getElementById('fps-counter');
            if (el) el.style.display = val ? 'block' : 'none';
        });

    advancedFolder.close();

    // Add tooltips + fine-grained step arrows to all folders after a short delay
    setTimeout(() => {
        addTooltipsToFolder(mainFolder);
        addTooltipsToFolder(advancedFolder);
        addStepArrowsToFolder(mainFolder);
        addStepArrowsToFolder(advancedFolder);
    }, 100);

    // Recording / video export folder (recording.js)
    if (typeof setupRecording === 'function') {
        try {
            const recCanvas = document.getElementById('c');
            setupRecording(recCanvas, simState.gui);
        } catch (e) {
            console.warn('setupRecording failed:', e);
        }
    }

    // Hook all GUI controllers for event logging
    hookGUIForEventLog();
}

/**
 * Attach onChange listeners to all dat.GUI controllers to record param changes.
 */
function hookGUIForEventLog() {
    if (!simState.gui) return;

    const trackableParams = new Set([
        'SOFTMAX_TEMP', 'ALIVE_THRESHOLD', 'LEARNING_BALANCE', 'LEARNING_RATE',
        'SUN_LR_SCALE', 'GROWTH_GATE_STEEPNESS',
        'DIVERSITY_WEIGHT', 'SUN_PENALTY', 'MIN_OCCUPANCY', 'STOCHASTIC_UPDATE_PROB',
        'USE_LOCAL_WIN_RATE', 'LOCAL_WIN_RATE_ALPHA', 'LOCAL_WIN_RATE_STRENGTH',
        'COMPETITION_MODE', 'MODEL_DROPOUT_P', 'OPTIMIZER_TYPE', 'USE_COSINE_SIM',
        'USE_CONCENTRATION', 'SUN_MODE', 'CONCENTRATION_TEMP', 'CONC_BASELINE', 'CONCENTRATION_WINDOW',
        'OPTIMIZATION_PERCENT', 'ASINH_SLOPE', 'SOFT_MIN_K', 'SUN_INIT_SCALE',
        'N_NCAS', 'CELL_STATE_DIM', 'CELL_HIDDEN_DIM', 'HIDDEN_DIM',
        'N_HIDDEN_LAYERS', 'KERNEL_SIZE', 'GRID_W', 'GRID_H', 'N_SEEDS',
    ]);

    function hookFolder(folder) {
        for (const controller of folder.__controllers) {
            const prop = controller.property;
            if (trackableParams.has(prop)) {
                const originalOnChange = controller.__onChange;
                let lastValue = GUI_STATE[prop];
                controller.onChange((value) => {
                    if (value !== lastValue) {
                        runManager.eventLog.logParamChange(
                            simState.globalStep, prop, lastValue, value
                        );
                        lastValue = value;
                        // Trigger timeline redraw
                        if (timelineDashboard) timelineDashboard.needsRedraw = true;
                    }
                    if (originalOnChange) originalOnChange.call(controller, value);
                });
            }
        }
        for (const subfolder of Object.values(folder.__folders)) {
            hookFolder(subfolder);
        }
    }

    hookFolder(simState.gui);
}

function setupNCAButtons() {
    if (!simState.drawFolder) return;

    const oldContainer = document.getElementById('nca-grid-container');
    if (oldContainer) {
        oldContainer.remove();
    }

    const container = document.createElement('div');
    container.id = 'nca-grid-container';
    container.className = 'nca-grid-container';

    const n_cols = Math.min(CONFIG.N_NCAS, 3);
    container.style.gridTemplateColumns = `repeat(${n_cols}, 1fr)`;

    for (let i = 1; i < simState.nca_colors.length; i++) {
        const nca_idx = i - 1;
        const toolName = `nca_${nca_idx}`;
        const color = simState.nca_colors[i];

        const btn = document.createElement('button');
        btn.className = 'nca-grid-button modern-pointer';
        btn.title = `Draw Species ${nca_idx}`;
        btn.innerText = `Draw ${nca_idx}`;
        btn.dataset.toolName = toolName;
        const r = Math.round(color[0] * 255), g = Math.round(color[1] * 255), b = Math.round(color[2] * 255);
        btn.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        btn.style.setProperty('--species-glow', `rgba(${r}, ${g}, ${b}, 0.8)`);
        btn.style.setProperty('--species-glow-dim', `rgba(${r}, ${g}, ${b}, 0.3)`);

        btn.addEventListener('click', () => {
            setTool(toolName);
        });

        simState.toolControllers[toolName] = btn;
        container.appendChild(btn);
    }

    // Insert NCA buttons immediately above "Tool Size" in the folder
    const innerUl = simState.drawFolder.domElement.querySelector('ul');
    const wrapper = document.createElement('li');
    wrapper.style.cssText = 'padding:0; background:transparent; height:auto; overflow:visible; border:none;';
    wrapper.appendChild(container);

    if (innerUl && simState.toolSizeController) {
        // Find the Tool Size <li> and insert before it
        const toolSizeLi = simState.toolSizeController.domElement.parentElement.parentElement;
        if (toolSizeLi && toolSizeLi.parentElement === innerUl) {
            innerUl.insertBefore(wrapper, toolSizeLi);
        } else {
            innerUl.appendChild(wrapper);
        }
    } else if (innerUl) {
        innerUl.appendChild(wrapper);
    } else {
        simState.drawFolder.domElement.appendChild(container);
    }
    updateToolStyles();
}

// =============================================================================
// MOUSE HANDLING
// =============================================================================

function setupMouseListeners() {
    const canvas = document.getElementById('c');

    canvas.addEventListener('mousedown', (e) => {
        simState.isDrawing = true;
        simState.lastDrawX = null;
        simState.lastDrawY = null;
        drawOnGrid(e);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (simState.isDrawing) {
            drawOnGrid(e);
        }
    });

    canvas.addEventListener('mouseup', () => {
        simState.isDrawing = false;
        simState.lastDrawX = null;
        simState.lastDrawY = null;
    });

    canvas.addEventListener('mouseleave', () => {
        simState.isDrawing = false;
        simState.lastDrawX = null;
        simState.lastDrawY = null;
    });
}

function drawOnGrid(e) {
    const tool = GUI_STATE.DRAW_TOOL;
    if (tool === 'none') return;

    const canvas = document.getElementById('c');
    const rect = canvas.getBoundingClientRect();

    // Get canvas UV (0,0 at bottom-left)
    let u = (e.clientX - rect.left) / canvas.clientWidth;
    let v = 1.0 - ((e.clientY - rect.top) / canvas.clientHeight);

    // Convert canvas UV to grid UV (aspect ratio correction)
    const gridAspect = CONFIG.GRID_W / CONFIG.GRID_H;
    const canvasAspect = canvas.clientWidth / canvas.clientHeight;
    const aspect = gridAspect / canvasAspect;

    if (aspect > 1.0) {
        v = (v - 0.5) * aspect + 0.5;
    } else {
        u = (u - 0.5) / aspect + 0.5;
    }

    const gridX = Math.floor(u * CONFIG.GRID_W);
    const gridY = Math.floor(v * CONFIG.GRID_H);

    // Get points to draw (interpolate from last position if available)
    const points = getLinePoints(simState.lastDrawX, simState.lastDrawY, gridX, gridY);

    // Draw at all interpolated points
    if (tool === 'eraser') {
        // Batch eraser: collect all cells, do one scatter update
        for (const [px, py] of points) {
            drawWall(px, py, 0.0);
        }
        eraseNCABatch(points, GUI_STATE.DRAW_SIZE);
    } else {
        for (const [px, py] of points) {
            if (tool === 'pencil') {
                drawWall(px, py, 1.0);
            } else if (tool.startsWith('nca_')) {
                const nca_idx = parseInt(tool.split('_')[1]);
                drawNCA(px, py, nca_idx, GUI_STATE.DRAW_SIZE);
            }
        }
    }

    // Store current position for next interpolation
    simState.lastDrawX = gridX;
    simState.lastDrawY = gridY;
}

/**
 * Get all points along a line using Bresenham's algorithm.
 * Returns array of [x, y] pairs.
 */
function getLinePoints(x0, y0, x1, y1) {
    // If no previous point, just return current point
    if (x0 === null || y0 === null) {
        return [[x1, y1]];
    }

    const points = [];
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;

    let x = x0;
    let y = y0;

    while (true) {
        points.push([x, y]);

        if (x === x1 && y === y1) break;

        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }

    return points;
}

function drawWall(gridX, gridY, value) {
    const size = GUI_STATE.DRAW_SIZE;
    const halfSize = Math.floor(size / 2);

    tf.tidy(() => {
        const indices = [];
        const updates = [];

        for (let i = -halfSize; i <= halfSize; i++) {
            for (let j = -halfSize; j <= halfSize; j++) {
                const cX = gridX + i;
                const cY = gridY + j;
                if (cX >= 0 && cX < CONFIG.GRID_W && cY >= 0 && cY < CONFIG.GRID_H) {
                    indices.push([0, cY, cX, 0]);
                    updates.push(value);
                }
            }
        }

        if (indices.length === 0) return;

        const indicesTensor = tf.tensor2d(indices, [indices.length, 4], 'int32');
        const updatesTensor = tf.tensor1d(updates, 'float32');
        const newWallGrid = tf.tensorScatterUpdate(simState.wallGrid, indicesTensor, updatesTensor);

        simState.wallGrid.dispose();
        simState.wallGrid = tf.keep(newWallGrid);
    });
}

function drawNCA(gridX, gridY, nca_idx, size) {
    const halfSize = Math.floor(size / 2);

    const aliveUpdate = new Array(CONFIG.ALIVE_DIM).fill(0.0);
    aliveUpdate[0] = 0.0;
    aliveUpdate[nca_idx + 1] = 1.0;

    tf.tidy(() => {
        const randomState = tf.randomNormal([CONFIG.CELL_WO_ALIVE_DIM]);
        const normState = randomState.div(randomState.norm());
        const normStateData = normState.dataSync();

        const cellUpdate = [...aliveUpdate, ...normStateData];

        const indices = [];
        const updates = [];

        for (let i = -halfSize; i <= halfSize; i++) {
            for (let j = -halfSize; j <= halfSize; j++) {
                const cX = gridX + i;
                const cY = gridY + j;
                if (cX >= 0 && cX < CONFIG.GRID_W && cY >= 0 && cY < CONFIG.GRID_H) {
                    indices.push([0, cY, cX]);
                    updates.push(cellUpdate);
                }
            }
        }

        if (indices.length === 0) return;

        const indicesTensor = tf.tensor2d(indices, [indices.length, 3], 'int32');
        const updatesTensor = tf.tensor2d(updates, [indices.length, CONFIG.CELL_DIM]);

        const newGrid = tf.tensorScatterUpdate(simState.grid, indicesTensor, updatesTensor);

        simState.grid.dispose();
        simState.grid = tf.keep(newGrid);
    });
}

/**
 * Batch erase NCAs at multiple points in a single tensor operation.
 * Prevents rapid tensor create/dispose cycles that cause instability.
 */
function eraseNCABatch(points, size) {
    const halfSize = Math.floor(size / 2);
    const aliveUpdate = new Array(CONFIG.ALIVE_DIM).fill(0.0);
    aliveUpdate[0] = 1.0; // Set to sun
    const stateUpdate = new Array(CONFIG.CELL_WO_ALIVE_DIM).fill(0.0);
    const cellUpdate = [...aliveUpdate, ...stateUpdate];

    tf.tidy(() => {
        const indices = [];
        const updates = [];
        const seen = new Set();

        for (const [px, py] of points) {
            for (let i = -halfSize; i <= halfSize; i++) {
                for (let j = -halfSize; j <= halfSize; j++) {
                    const cX = px + i;
                    const cY = py + j;
                    if (cX >= 0 && cX < CONFIG.GRID_W && cY >= 0 && cY < CONFIG.GRID_H) {
                        const key = cY * CONFIG.GRID_W + cX;
                        if (!seen.has(key)) {
                            seen.add(key);
                            indices.push([0, cY, cX]);
                            updates.push(cellUpdate);
                        }
                    }
                }
            }
        }

        if (indices.length === 0) return;

        const indicesTensor = tf.tensor2d(indices, [indices.length, 3], 'int32');
        const updatesTensor = tf.tensor2d(updates, [indices.length, CONFIG.CELL_DIM]);
        const newGrid = tf.tensorScatterUpdate(simState.grid, indicesTensor, updatesTensor);
        simState.grid.dispose();
        simState.grid = tf.keep(newGrid);
    });
}

function eraseNCA(gridX, gridY, size) {
    const halfSize = Math.floor(size / 2);

    const aliveUpdate = new Array(CONFIG.ALIVE_DIM).fill(0.0);
    aliveUpdate[0] = 1.0;

    const stateUpdate = new Array(CONFIG.CELL_WO_ALIVE_DIM).fill(0.0);
    const cellUpdate = [...aliveUpdate, ...stateUpdate];

    tf.tidy(() => {
        const indices = [];
        const updates = [];

        for (let i = -halfSize; i <= halfSize; i++) {
            for (let j = -halfSize; j <= halfSize; j++) {
                const cX = gridX + i;
                const cY = gridY + j;
                if (cX >= 0 && cX < CONFIG.GRID_W && cY >= 0 && cY < CONFIG.GRID_H) {
                    indices.push([0, cY, cX]);
                    updates.push(cellUpdate);
                }
            }
        }

        if (indices.length === 0) return;

        const indicesTensor = tf.tensor2d(indices, [indices.length, 3], 'int32');
        const updatesTensor = tf.tensor2d(updates, [indices.length, CONFIG.CELL_DIM]);

        const newGrid = tf.tensorScatterUpdate(simState.grid, indicesTensor, updatesTensor);

        simState.grid.dispose();
        simState.grid = tf.keep(newGrid);
    });
}

function addSeeds() {
    const { N_SEEDS } = GUI_STATE;
    const { N_NCAS, GRID_W, GRID_H } = CONFIG;

    tf.tidy(() => {
        const indices = [];
        const updates = [];

        for (let nca_i = 0; nca_i < N_NCAS; nca_i++) {
            const randomState = tf.randomNormal([CONFIG.CELL_WO_ALIVE_DIM]);
            const normState = randomState.div(randomState.norm());
            const normStateData = normState.dataSync();

            const aliveUpdate = new Array(CONFIG.ALIVE_DIM).fill(0.0);
            aliveUpdate[0] = 0.0;
            aliveUpdate[nca_i + 1] = 1.0;

            const cellUpdate = [...aliveUpdate, ...normStateData];

            for (let seed_j = 0; seed_j < N_SEEDS; seed_j++) {
                const x = Math.floor(seededRandom() * GRID_W);
                const y = Math.floor(seededRandom() * GRID_H);

                indices.push([0, y, x]);
                updates.push(cellUpdate);
            }
        }

        if (indices.length === 0) return;

        const indicesTensor = tf.tensor2d(indices, [indices.length, 3], 'int32');
        const updatesTensor = tf.tensor(updates);

        const newGrid = tf.tensorScatterUpdate(simState.grid, indicesTensor, updatesTensor);

        simState.grid.dispose();
        simState.grid = tf.keep(newGrid);
    });
}

// =============================================================================
// SIMULATION INITIALIZATION
// =============================================================================

async function initializeSimulation(keepState = false, oldGrid, oldWallGrid, oldSunBase, oldSunParams) {
    simState.stepCount = 0;
    simState.globalStep = 0;

    // Initialize seeded RNG and run manager for new runs
    if (!keepState) {
        const seed = initSeededRNG();
        runManager.startNewRun(seed, CONFIG, GUI_STATE);
        runManager.metrics.speciesCount = GUI_STATE.N_NCAS;
    }

    // Dispose old tensors. INR weights are grid-size-invariant so we keep them
    // across keepState resizes; only the coords cache (which IS grid-sized)
    // is always rebuilt below.
    if (simState.sun_base) simState.sun_base.dispose();
    if (simState.sun_params) simState.sun_params.dispose();
    if (simState.inr_coords) { simState.inr_coords.dispose(); simState.inr_coords = null; }
    if (!keepState && simState.inr) { disposeInr(simState.inr); simState.inr = null; }
    if (simState.model) simState.model.dispose();

    // Update CONFIG from GUI
    CONFIG.GRID_W = GUI_STATE.GRID_W;
    CONFIG.GRID_H = GUI_STATE.GRID_H;
    CONFIG.N_SEEDS = GUI_STATE.N_SEEDS;
    CONFIG.N_NCAS = GUI_STATE.N_NCAS;
    CONFIG.CELL_STATE_DIM = GUI_STATE.CELL_STATE_DIM;
    CONFIG.CELL_HIDDEN_DIM = GUI_STATE.CELL_HIDDEN_DIM;
    CONFIG.N_HIDDEN_LAYERS = GUI_STATE.N_HIDDEN_LAYERS;
    CONFIG.HIDDEN_DIM = GUI_STATE.HIDDEN_DIM;
    CONFIG.KERNEL_SIZE = parseInt(GUI_STATE.KERNEL_SIZE, 10);
    CONFIG.MODEL_DROPOUT_P = GUI_STATE.MODEL_DROPOUT_P;
    CONFIG.SUN_INIT_SCALE = GUI_STATE.SUN_INIT_SCALE;
    CONFIG.ALIVE_THRESHOLD = GUI_STATE.ALIVE_THRESHOLD;
    // Update derived values
    updateDerivedConfig();

    // Create model (layers are built internally with dummy forward pass)
    simState.model = createModel(CONFIG);

    // Create sun (preserve local sun state on keepState resize if possible)
    if (keepState && oldSunBase && oldSunParams && oldSunBase.shape[1] > 1) {
        // Local sun: resize spatially (pad or crop dims 1,2)
        simState.sun_base = tf.keep(resizeSunTensor(oldSunBase, CONFIG.GRID_H, CONFIG.GRID_W));
        const resizedParams = resizeSunTensor(oldSunParams, CONFIG.GRID_H, CONFIG.GRID_W);
        simState.sun_params = tf.keep(tf.variable(resizedParams, true));
        resizedParams.dispose();
        oldSunBase.dispose();
        oldSunParams.dispose();
    } else if (keepState && oldSunBase && oldSunParams && oldSunBase.shape[1] === 1) {
        // Global sun: shape doesn't depend on grid size, keep as-is
        simState.sun_base = oldSunBase;
        simState.sun_params = oldSunParams;
    } else {
        if (oldSunBase) oldSunBase.dispose();
        if (oldSunParams) oldSunParams.dispose();
        const { sun_base, sun_params, inr } = createSunUpdate(CONFIG);
        simState.sun_base = sun_base;
        simState.sun_params = sun_params;
        simState.inr = inr;
    }
    // Always (re)build the coords cache when in seasonal mode — its shape
    // depends on GRID_H/W, so a resize makes it stale.
    if (GUI_STATE.SUN_MODE === 'seasonal') {
        if (!simState.inr) simState.inr = createInrParams(CONFIG);
        simState.inr_coords = buildCoordsCache(CONFIG.GRID_H, CONFIG.GRID_W);
    }
    simState.optimizer = createOptimizer();

    // Handle grid resizing/creation
    let newGrid;
    let newWallGrid;

    if (keepState && oldGrid) {
        const oldCellDim = oldGrid.shape[3];
        const newCellDim = CONFIG.CELL_DIM;

        if (oldCellDim !== newCellDim) {
            oldGrid.dispose();
            newGrid = tf.keep(createGrid(CONFIG));
        } else {
            newGrid = resizeGrid(oldGrid, CONFIG.GRID_H, CONFIG.GRID_W, CONFIG.CELL_DIM);
        }
    } else {
        if (oldGrid) oldGrid.dispose();
        newGrid = tf.keep(createGrid(CONFIG));
    }

    // Always preserve walls (resize if grid dimensions changed)
    if (oldWallGrid) {
        newWallGrid = resizeWallGrid(oldWallGrid, CONFIG.GRID_H, CONFIG.GRID_W);
    } else {
        newWallGrid = tf.keep(tf.zeros([1, CONFIG.GRID_H, CONFIG.GRID_W, 1], 'float32'));
    }

    simState.grid = newGrid;
    simState.wallGrid = newWallGrid;

    // Initialize local win rate tensor (for update probability modulation)
    if (simState.localWinRate) simState.localWinRate.dispose();
    simState.localWinRate = tf.keep(tf.fill([1, CONFIG.GRID_H, CONFIG.GRID_W, 1], 0.5));

    // Colors
    simState.nca_colors = generateNCAColors(CONFIG.N_NCAS);
    simState.color_uniform = simState.nca_colors.flat();

    // Only setup GUI once, just update NCA buttons when species count changes
    if (!simState.guiInitialized) {
        setupGUI();
        simState.guiInitialized = true;
    }
    // Always update NCA buttons (they depend on N_NCAS)
    setupNCAButtons();

    // Create textures
    simState.TEX_W = CONFIG.GRID_W * CONFIG.PACKED_PIXELS;
    simState.TEX_H = CONFIG.GRID_H;

    simState.gridTex = simState.glsl({}, {
        size: [simState.TEX_W, simState.TEX_H],
        format: 'rgba32f',
        tag: 'gridData'
    });

    simState.wallTex = simState.glsl({}, {
        size: [CONFIG.GRID_W, CONFIG.GRID_H],
        format: 'rgba32f',
        tag: 'wallData'
    });

    // Create render shader
    simState.renderShader = createRenderShader();

    // Update border
    updateGridBorder();

    // Sync sun heatmap visibility
    updateSunHeatmapVisibility(GUI_STATE.SUN_MODE !== 'global');
    updateSeasonalPanelVisibility(GUI_STATE.SUN_MODE === 'seasonal');

    // The first simulation step after a config change will be slow (~5-10s)
    // due to WebGL shader compilation. trainStep() yields between its inference
    // and gradient phases, and simulationLoop() shows the loading spinner.
}

function resizeGrid(oldGrid, newH, newW, cellDim) {
    const tempGrid = tf.tidy(() => {
        let g = oldGrid;
        const oldH = oldGrid.shape[1];
        const oldW = oldGrid.shape[2];
        const diffH = newH - oldH;
        const diffW = newW - oldW;

        if (diffH > 0) {
            const padTop = Math.floor(diffH / 2);
            const padBottom = diffH - padTop;
            g = g.pad([[0, 0], [padTop, padBottom], [0, 0], [0, 0]]);
        } else if (diffH < 0) {
            const sliceTop = Math.floor(Math.abs(diffH) / 2);
            g = g.slice([0, sliceTop, 0, 0], [1, newH, oldW, cellDim]);
        }

        if (diffW > 0) {
            const padLeft = Math.floor(diffW / 2);
            const padRight = diffW - padLeft;
            g = g.pad([[0, 0], [0, 0], [padLeft, padRight], [0, 0]]);
        } else if (diffW < 0) {
            const sliceLeft = Math.floor(Math.abs(diffW) / 2);
            g = g.slice([0, 0, sliceLeft, 0], [1, g.shape[1], newW, cellDim]);
        }

        if (g !== oldGrid) {
            oldGrid.dispose();
        }

        return tf.keep(g);
    });

    return tempGrid;
}

function resizeWallGrid(oldWallGrid, newH, newW) {
    const tempWallGrid = tf.tidy(() => {
        let w = oldWallGrid;
        const oldH = oldWallGrid.shape[1];
        const oldW = oldWallGrid.shape[2];
        const diffH = newH - oldH;
        const diffW = newW - oldW;

        if (diffH > 0) {
            const padTop = Math.floor(diffH / 2);
            const padBottom = diffH - padTop;
            w = w.pad([[0, 0], [padTop, padBottom], [0, 0], [0, 0]]);
        } else if (diffH < 0) {
            const sliceTop = Math.floor(Math.abs(diffH) / 2);
            w = w.slice([0, sliceTop, 0, 0], [1, newH, oldW, 1]);
        }

        if (diffW > 0) {
            const padLeft = Math.floor(diffW / 2);
            const padRight = diffW - padLeft;
            w = w.pad([[0, 0], [0, 0], [padLeft, padRight], [0, 0]]);
        } else if (diffW < 0) {
            const sliceLeft = Math.floor(Math.abs(diffW) / 2);
            w = w.slice([0, 0, sliceLeft, 0], [1, w.shape[1], newW, 1]);
        }

        if (w !== oldWallGrid) {
            oldWallGrid.dispose();
        }

        return tf.keep(w);
    });

    return tempWallGrid;
}

/**
 * Resizes a 4D sun tensor [1,H,W,C] by center-padding or center-cropping dims 1 and 2.
 */
function resizeSunTensor(tensor, newH, newW) {
    return tf.tidy(() => {
        let t = tensor;
        const oldH = t.shape[1];
        const oldW = t.shape[2];

        if (oldH === newH && oldW === newW) return t.clone();

        // Height: pad or crop
        if (newH > oldH) {
            const padTop = Math.floor((newH - oldH) / 2);
            const padBot = newH - oldH - padTop;
            t = t.pad([[0, 0], [padTop, padBot], [0, 0], [0, 0]]);
        } else if (newH < oldH) {
            const cropY = Math.floor((oldH - newH) / 2);
            t = t.slice([0, cropY, 0, 0], [1, newH, -1, -1]);
        }

        // Width: pad or crop
        if (newW > oldW) {
            const padLeft = Math.floor((newW - oldW) / 2);
            const padRight = newW - oldW - padLeft;
            t = t.pad([[0, 0], [0, 0], [padLeft, padRight], [0, 0]]);
        } else if (newW < oldW) {
            const cropX = Math.floor((oldW - newW) / 2);
            t = t.slice([0, 0, cropX, 0], [1, -1, newW, -1]);
        }

        return t;
    });
}

function createRenderShader() {
    return {
        grid: simState.gridTex,
        walls: simState.wallTex,
        colors: simState.color_uniform,
        canvasAspect: 1.0,

        FP: `
            uniform sampler2D grid;
            uniform sampler2D walls;
            uniform vec3 colors[${CONFIG.ALIVE_DIM}];
            uniform float canvasAspect;

            const int ALIVE_DIM = ${CONFIG.ALIVE_DIM};
            const int GRID_W = ${CONFIG.GRID_W};
            const int GRID_H = ${CONFIG.GRID_H};
            const int PACKED_PIXELS = ${CONFIG.PACKED_PIXELS};
            const float gridAspect = float(GRID_W) / float(GRID_H);

            float get_channel(int channel, ivec2 base_pixel_coord) {
                int pixel_x_offset = channel / 4;
                int component = channel % 4;
                vec4 tex_val = texelFetch(grid, base_pixel_coord + ivec2(pixel_x_offset, 0), 0);
                return component == 0 ? tex_val.r : component == 1 ? tex_val.g : component == 2 ? tex_val.b : tex_val.a;
            }

            void fragment() {
                vec2 uv = UV;

                // Aspect ratio correction
                float aspect = gridAspect / canvasAspect;
                vec2 adjUV = uv;
                if (aspect > 1.0) {
                    adjUV.y = (uv.y - 0.5) * aspect + 0.5;
                } else {
                    adjUV.x = (uv.x - 0.5) / aspect + 0.5;
                }

                // Outside grid boundary - dark grey to distinguish from black cells
                if (adjUV.x < 0.0 || adjUV.x > 1.0 || adjUV.y < 0.0 || adjUV.y > 1.0) {
                    FOut = vec4(0.12, 0.12, 0.14, 1.0);
                    return;
                }

                // Grid coordinates
                ivec2 grid_coord = ivec2(floor(adjUV * vec2(float(GRID_W), float(GRID_H))));
                ivec2 base_pix = ivec2(grid_coord.x * PACKED_PIXELS, grid_coord.y);

                // Wall rendering
                float wall = texelFetch(walls, grid_coord, 0).r;
                if (wall > 0.5) {
                    FOut = vec4(0.85, 0.88, 0.92, 1.0);
                    return;
                }

                // Find dominant species (highest aliveness value)
                float max_strength = 0.0;
                vec3 dom_color = colors[0]; // Default to sun color
                float sun_val = get_channel(0, base_pix);

                for (int i = 1; i < ALIVE_DIM; i++) {
                    float val = get_channel(i, base_pix);
                    if (val > max_strength) {
                        max_strength = val;
                        dom_color = colors[i];
                    }
                }

                // If sun dominates, show dark background
                if (sun_val > max_strength) {
                    FOut = vec4(0.02, 0.02, 0.03, 1.0);
                    return;
                }

                // Simple, vivid coloring - full brightness for living cells
                vec3 base = dom_color;

                // Slight brightness variation based on strength
                float brightness = 0.7 + 0.3 * min(max_strength * 2.0, 1.0);
                base *= brightness;

                FOut = vec4(base, 1.0);
            }
        `
    };
}

/**
 * Toggles between global and per-cell (local) sun parameters.
 * Switches sun mode across three tiers: 'global' | 'spatial' | 'seasonal'.
 *
 * global ↔ spatial : tile / spatial mean (lossless up, lossless-to-mean down).
 * spatial ↔ seasonal: INR replaces the sun entirely when seasonal. Promotion
 *   seeds INR output bias with the current sun's spatial mean (spatial
 *   variation is lost; INR re-learns it). Demotion freezes the INR's current
 *   spatial output as the new sun_base; sun_params starts at zero.
 * global ↔ seasonal : composed via spatial.
 */
function setSunMode(newMode) {
    const oldMode = (simState.inr) ? 'seasonal'
        : (simState.sun_base.shape[1] > 1 ? 'spatial' : 'global');
    if (newMode === oldMode) return;

    const H = CONFIG.GRID_H, W = CONFIG.GRID_W;
    const CWA = simState.sun_base.shape[simState.sun_base.shape.length - 1];

    const promoteGlobalToSpatial = () => {
        const oldBase = simState.sun_base, oldParams = simState.sun_params;
        simState.sun_base = tf.keep(oldBase.tile([1, H, W, 1]));
        simState.sun_params = tf.keep(tf.variable(tf.zeros([1, H, W, CWA]), true));
        oldBase.dispose(); oldParams.dispose();
    };

    const demoteSpatialToGlobal = () => {
        const oldBase = simState.sun_base, oldParams = simState.sun_params;
        simState.sun_base = tf.keep(oldBase.mean([1, 2], true));
        const paramsMean = oldParams.mean([1, 2], true);
        simState.sun_params = tf.keep(tf.variable(paramsMean, true));
        paramsMean.dispose();
        oldBase.dispose(); oldParams.dispose();
    };

    // Seeds the INR's output bias with the spatial mean of (sun_base + sun_params)
    // so at t=0 the sun is a spatially-uniform approximation of the previous state.
    const promoteSpatialToSeasonal = () => {
        simState.inr = createInrParams(CONFIG);
        if (simState.inr_coords) simState.inr_coords.dispose();
        simState.inr_coords = buildCoordsCache(H, W);
        tf.tidy(() => {
            const sum = simState.sun_base.add(simState.sun_params);
            const mu = sum.mean([1, 2]).reshape([CWA]);
            simState.inr.inr_bout.assign(mu);
        });
    };

    // Freezes the INR's current output as the new sun_base; sun_params starts at zero.
    const demoteSeasonalToSpatial = () => {
        if (simState.inr && simState.inr_coords) {
            const inr_out = evaluateInr(simState.inr, simState.inr_coords, CONFIG);
            const oldBase = simState.sun_base, oldParams = simState.sun_params;
            simState.sun_base = tf.keep(inr_out.clone());
            simState.sun_params = tf.keep(tf.variable(tf.zeros([1, H, W, CWA]), true));
            oldBase.dispose();
            oldParams.dispose();
            inr_out.dispose();
        }
        disposeInr(simState.inr);
        simState.inr = null;
        if (simState.inr_coords) { simState.inr_coords.dispose(); simState.inr_coords = null; }
    };

    if (oldMode === 'global' && newMode === 'spatial') {
        promoteGlobalToSpatial();
    } else if (oldMode === 'spatial' && newMode === 'global') {
        demoteSpatialToGlobal();
    } else if (oldMode === 'spatial' && newMode === 'seasonal') {
        promoteSpatialToSeasonal();
    } else if (oldMode === 'seasonal' && newMode === 'spatial') {
        demoteSeasonalToSpatial();
    } else if (oldMode === 'global' && newMode === 'seasonal') {
        promoteGlobalToSpatial();
        promoteSpatialToSeasonal();
    } else if (oldMode === 'seasonal' && newMode === 'global') {
        demoteSeasonalToSpatial();
        demoteSpatialToGlobal();
    }
    console.log(`Sun mode: ${oldMode} → ${newMode}, sun_params shape: ${simState.sun_params.shape}`);
}

function updateSunHeatmapVisibility(visible) {
    const li = document.getElementById('sun-heatmap-li');
    if (li) li.style.display = visible ? 'block' : 'none';
    const rateLi = document.getElementById('sun-heatmap-rate-li');
    if (rateLi) rateLi.style.display = visible ? 'block' : 'none';
    const blendLi = document.getElementById('sun-overlay-blend-li');
    if (blendLi) blendLi.style.display = visible ? 'block' : 'none';
    if (!visible) hideSunOverlay();
}

function updateSeasonalPanelVisibility(visible) {
    const li = document.getElementById('sun-seasonal-li');
    if (li) li.style.display = visible ? 'block' : 'none';
}

let _seasonalPanelPending = false;
function updateSeasonalPanel() {
    if (_seasonalPanelPending) return;
    if (GUI_STATE.SUN_MODE !== 'seasonal' || !simState.inr) return;
    _seasonalPanelPending = true;
    getInrPeriods(simState.inr).then(periods => {
        _seasonalPanelPending = false;
        const el = document.getElementById('sun-seasonal-periods');
        if (!el) return;
        const fmt = v => v < 100 ? v.toFixed(1) : v < 10000 ? Math.round(v).toString() : v.toExponential(1);
        const lines = periods.map((p, i) => `S${i + 1}: ${fmt(p).padStart(7)}`);
        el.textContent = lines.join('\n');
    }).catch(() => { _seasonalPanelPending = false; });
}

function positionSunOverlay(centerX, centerY, w, h) {
    const overlay = document.getElementById('sun-overlay-canvas');
    if (!overlay) return;
    if (centerX === undefined) {
        const canvas = document.getElementById('c');
        const r = canvas.getBoundingClientRect();
        const canvasAspect = r.width / r.height;
        const gridAspect = CONFIG.GRID_W / CONFIG.GRID_H;
        if (canvasAspect > gridAspect) { h = r.height; w = h * gridAspect; }
        else { w = r.width; h = w / gridAspect; }
        centerX = r.left + r.width / 2;
        centerY = r.top + r.height / 2;
    }
    overlay.style.left = `${centerX}px`;
    overlay.style.top = `${centerY}px`;
    overlay.style.width = `${w}px`;
    overlay.style.height = `${h}px`;
}

function renderSunOverlay() {
    const src = document.getElementById('sun-heatmap-canvas');
    const overlay = document.getElementById('sun-overlay-canvas');
    if (!src || !overlay || !src.width || !src.height) return;
    // Source canvas is W x (H + KEY_H + 2). The heatmap proper occupies the top H rows.
    const W = src.width;
    const H = CONFIG.GRID_H;
    if (overlay.width !== W || overlay.height !== H) {
        overlay.width = W;
        overlay.height = H;
    }
    const ctx = overlay.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(src, 0, 0, W, H, 0, 0, W, H);
}

function showSunOverlay() {
    const overlay = document.getElementById('sun-overlay-canvas');
    if (!overlay) return;
    positionSunOverlay();
    renderSunOverlay();
    overlay.style.display = 'block';
}

function hideSunOverlay() {
    const overlay = document.getElementById('sun-overlay-canvas');
    if (overlay) overlay.style.display = 'none';
}

// Guard to prevent overlapping heatmap updates (async)
let _sunHeatmapPending = false;

/**
 * Computes and renders the sun divergence heatmap.
 * Divergence = ||sun_i - mean(sun)|| per cell, computed on the GPU.
 * Only the small [H,W] divergence map is transferred to CPU (~64x smaller than full params).
 */
function updateSunHeatmap() {
    if (_sunHeatmapPending) return;
    if (GUI_STATE.SUN_MODE === 'global' || !simState.sun_params || simState.sun_params.shape[1] <= 1) return;

    const canvas = document.getElementById('sun-heatmap-canvas');
    const statsEl = document.getElementById('sun-heatmap-stats');
    if (!canvas) return;

    // Compute divergence on GPU, excluding wall cells from the mean.
    // In seasonal mode, read the INR's current spatial output. Otherwise,
    // read sun_params directly.
    const { divTensor, wallTensor } = tf.tidy(() => {
        const isSeasonal = (GUI_STATE.SUN_MODE === 'seasonal') && simState.inr && simState.inr_coords;
        const field = isSeasonal
            ? evaluateInr(simState.inr, simState.inr_coords, CONFIG)   // [1,H,W,CWA]
            : simState.sun_params;                                      // [1,H,W,CWA]
        const walls = simState.wallGrid;                        // [1, H, W, 1]
        const nonWall = walls.less(0.5).cast('float32');        // 1 = open, 0 = wall
        const nonWallCount = nonWall.sum().maximum(1);
        const maskedField = field.mul(nonWall);
        const mean = maskedField.sum([1, 2], true).div(nonWallCount);
        const diff = field.sub(mean).mul(nonWall);
        const div = diff.square().sum(-1).sqrt().squeeze([0]);  // [H, W]
        const wm = walls.squeeze([0, 3]);                       // [H, W]
        return { divTensor: div, wallTensor: wm };
    });

    _sunHeatmapPending = true;
    Promise.all([divTensor.data(), wallTensor.data()]).then(([dist, wallData]) => {
        _sunHeatmapPending = false;
        const H = divTensor.shape[0];
        const W = divTensor.shape[1];
        divTensor.dispose();
        wallTensor.dispose();

        // Extra row for the color key bar
        const KEY_H = Math.max(4, Math.round(H * 0.04));
        const canvasH = H + KEY_H + 2; // 2px gap
        if (canvas.width !== W || canvas.height !== canvasH) {
            canvas.width = W;
            canvas.height = canvasH;
        }
        // Force square-ish display (slightly taller due to key)
        const displayWidth = canvas.getBoundingClientRect().width;
        if (displayWidth > 0) canvas.style.height = (displayWidth * canvasH / W) + 'px';

        // Log-scale normalization excluding walls
        const N = H * W;
        let maxLog = 0, sumDist = 0, maxDist = 0, nonWallCount = 0;
        const logDist = new Float32Array(N);
        for (let i = 0; i < N; i++) {
            const isWall = wallData[i] >= 0.5;
            if (isWall) { logDist[i] = -1; continue; } // sentinel for walls
            nonWallCount++;
            sumDist += dist[i];
            if (dist[i] > maxDist) maxDist = dist[i];
            logDist[i] = Math.log1p(dist[i]);
            if (logDist[i] > maxLog) maxLog = logDist[i];
        }

        // Turbo colormap (Mikhailov 2019): blue → cyan → green → yellow → red
        function turbo(t) {
            // Polynomial approximation of the Turbo colormap
            const r = Math.max(0, Math.min(255, (34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))))|0));
            const g = Math.max(0, Math.min(255, (23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1073.77 + t * 707.56)))))|0));
            const b = Math.max(0, Math.min(255, (27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66)))))|0));
            return [r, g, b];
        }

        // Render heatmap with Y-flip (grid row 0 = bottom, canvas row 0 = top)
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(W, canvasH);
        const px = imgData.data;
        const inv = maxLog > 1e-8 ? 1 / maxLog : 0;
        for (let y = 0; y < H; y++) {
            const srcRow = H - 1 - y; // flip Y
            for (let x = 0; x < W; x++) {
                const ld = logDist[srcRow * W + x];
                const idx = (y * W + x) * 4;
                if (ld < 0) {
                    // Wall cell — render as dark grey
                    px[idx] = 30; px[idx+1] = 30; px[idx+2] = 35; px[idx+3] = 255;
                } else {
                    const [r, g, b] = turbo(ld * inv);
                    px[idx] = r; px[idx+1] = g; px[idx+2] = b; px[idx+3] = 255;
                }
            }
        }

        // Draw color key bar at the bottom (2px gap, then KEY_H rows)
        const keyStartY = H + 2;
        for (let ky = 0; ky < KEY_H; ky++) {
            for (let x = 0; x < W; x++) {
                const t = x / (W - 1);
                const [r, g, b] = turbo(t);
                const idx = ((keyStartY + ky) * W + x) * 4;
                px[idx] = r; px[idx+1] = g; px[idx+2] = b; px[idx+3] = 255;
            }
        }
        ctx.putImageData(imgData, 0, 0);

        // Labels on the key
        ctx.fillStyle = '#ccc';
        ctx.font = `${Math.max(8, Math.round(KEY_H * 1.8))}px "JetBrains Mono", monospace`;
        ctx.textBaseline = 'top';
        ctx.fillText('low', 2, keyStartY + KEY_H + 1);
        ctx.textAlign = 'right';
        ctx.fillText('high', W - 2, keyStartY + KEY_H + 1);

        if (statsEl) {
            const meanD = nonWallCount > 0 ? sumDist / nonWallCount : 0;
            const fmt = v => v === 0 ? '0' : v < 0.001 ? v.toExponential(1) : v.toFixed(4);
            statsEl.textContent = `mean: ${fmt(meanD)}  max: ${fmt(maxDist)}`;
        }

        // If the overlay is currently visible, refresh it with the new heatmap frame.
        const overlay = document.getElementById('sun-overlay-canvas');
        if (overlay && overlay.style.display === 'block') renderSunOverlay();
    }).catch(() => { _sunHeatmapPending = false; });
}

async function resetSimulation(keepState = false) {
    simState.isTraining = false;
    GUI_STATE.DRAW_TOOL = 'none';

    const oldGrid = simState.grid;
    const oldWallGrid = simState.wallGrid;
    const oldSunBase = simState.sun_base;
    const oldSunParams = simState.sun_params;
    simState.grid = null;
    simState.wallGrid = null;
    simState.sun_base = null;
    simState.sun_params = null;
    // localWinRate will be re-initialized in initializeSimulation

    if (simState.startPauseController) {
        simState.startPauseController.name('Start');
        simState.startPauseController.domElement.parentElement.parentElement.style.backgroundColor = '#4CAF50';
    }

    await initializeSimulation(keepState, oldGrid, oldWallGrid, oldSunBase, oldSunParams);
    renderFrame();
}

function updateGridBorder() {
    const border = document.getElementById('grid-border');
    const canvas = document.getElementById('c');
    const canvasRect = canvas.getBoundingClientRect();
    const canvasAspect = canvasRect.width / canvasRect.height;
    const gridAspect = CONFIG.GRID_W / CONFIG.GRID_H;

    let w, h;
    if (canvasAspect > gridAspect) {
        h = canvasRect.height;
        w = h * gridAspect;
    } else {
        w = canvasRect.width;
        h = w / gridAspect;
    }

    // Position centered on the canvas area, not the full viewport
    const centerX = canvasRect.left + canvasRect.width / 2;
    const centerY = canvasRect.top + canvasRect.height / 2;

    border.style.left = `${centerX}px`;
    border.style.top = `${centerY}px`;
    border.style.width = `${w}px`;
    border.style.height = `${h}px`;

    positionSunOverlay(centerX, centerY, w, h);
}

// =============================================================================
// SIMULATION LOOP
// =============================================================================

async function simulationLoop() {
    if (!simState.isTraining) {
        hideLoadingSpinner();
        return;
    }

    const balance = GUI_STATE.LEARNING_BALANCE;
    let t0, t1;

    // Inference-only mode
    if (balance > 100) {
        if (simState.globalStep === 0) {
            showLoadingSpinner('Simulation commencing — first step compiles GPU shaders and is slow');
            const detail = document.getElementById('loading-spinner-detail');
            if (detail) detail.textContent = 'Forward pass → model predict + competition';
            await new Promise(r => requestAnimationFrame(r));
        }
        if (LOG_PERFORMANCE) t0 = performance.now();
        const result = tf.tidy(() =>
            simulationStep(simState.grid, simState.model, simState.sun_base, simState.sun_params, simState.wallGrid, CONFIG, simState.localWinRate, simState.inr, simState.inr_coords)
        );
        stepInrPhase(simState.inr);
        if (LOG_PERFORMANCE) {
            t1 = performance.now();
            console.log(`[PERF] Inference: ${(t1 - t0).toFixed(2)} ms`);
        }
        simState.grid.dispose();
        // Handle both return signatures (grid only or { grid, localWinRate })
        if (result.grid) {
            simState.grid = tf.keep(result.grid);
            if (result.localWinRate) {
                simState.localWinRate.dispose();
                simState.localWinRate = tf.keep(result.localWinRate);
            }
        } else {
            simState.grid = result;
        }
    }
    // Balanced mode
    else if (balance >= 0) {
        simState.stepCount++;
        const stepsPerGrad = balance === 0 ? 1 : balance;

        if (simState.stepCount >= stepsPerGrad) {
            const isFirstStep = simState.globalStep === 0;
            let progressCb = null;
            if (isFirstStep) {
                showLoadingSpinner('Simulation commencing — first step compiles GPU shaders and is slow');
                await new Promise(resolve => requestAnimationFrame(resolve));
                progressCb = (msg) => {
                    const detail = document.getElementById('loading-spinner-detail');
                    if (detail) detail.textContent = msg;
                };
            }

            if (LOG_PERFORMANCE) t0 = performance.now();
            const { newGrid, loss } = await trainStep(
                simState.grid, simState.model, simState.sun_base, simState.sun_params,
                simState.wallGrid, simState.optimizer, CONFIG, progressCb,
                simState.inr, simState.inr_coords
            );
            // Advance INR phase accumulator outside the gradient tape.
            stepInrPhase(simState.inr);
            if (LOG_PERFORMANCE) {
                t1 = performance.now();
                console.log(`[PERF] Training: ${(t1 - t0).toFixed(2)} ms, Loss: ${loss.toFixed(4)}`);
            }

            simState.grid.dispose();
            simState.grid = newGrid;
            simState.stepCount = 0;
            hideLoadingSpinner();
        } else {
            if (LOG_PERFORMANCE) t0 = performance.now();
            const result = tf.tidy(() =>
                simulationStep(simState.grid, simState.model, simState.sun_base, simState.sun_params, simState.wallGrid, CONFIG, simState.localWinRate, simState.inr, simState.inr_coords)
            );
            stepInrPhase(simState.inr);
            if (LOG_PERFORMANCE) {
                t1 = performance.now();
                console.log(`[PERF] Sim step: ${(t1 - t0).toFixed(2)} ms`);
            }
            simState.grid.dispose();
            // Handle both return signatures
            if (result.grid) {
                simState.grid = tf.keep(result.grid);
                if (result.localWinRate) {
                    simState.localWinRate.dispose();
                    simState.localWinRate = tf.keep(result.localWinRate);
                }
            } else {
                simState.grid = result;
            }
        }
    }

    // Emergency respawn for species that have died out
    if (GUI_STATE.MIN_OCCUPANCY > 0) {
        const respawnedGrid = emergencyRespawn(simState.grid, simState.wallGrid, CONFIG);
        if (respawnedGrid !== simState.grid) {
            simState.grid.dispose();
            simState.grid = respawnedGrid;
        }
    }

    // --- Per-step recording frame capture ---
    if (typeof recordingCaptureFrame === 'function') recordingCaptureFrame();

    // --- Timeline metrics collection ---
    simState.globalStep++;
    runManager.globalStep = simState.globalStep; // Keep in sync

    // Apply replay events if in replay mode
    if (runManager.mode === 'REPLAY' && runManager.replayEventLog) {
        applyReplayEvents();
    }

    // Collect metrics every step (async, non-blocking)
    if (simState.globalStep % 2 === 0) { // Every 2 steps to reduce overhead
        collectMetrics();
    }
    // Update sun divergence heatmap at user-configured cadence when sun is non-global
    const heatmapEvery = Math.max(1, GUI_STATE.HEATMAP_UPDATE_EVERY | 0);
    if (GUI_STATE.USE_LOCAL_SUN && simState.globalStep % heatmapEvery === 0) {
        updateSunHeatmap();
    }
    // Update seasonal periods read-out every 30 steps when seasonal mode is active
    if (GUI_STATE.SUN_MODE === 'seasonal' && simState.globalStep % 30 === 0) {
        updateSeasonalPanel();
    }
    // Auto-checkpoint every 500 steps
    if (runManager.shouldAutoSave()) {
        autoSaveCheckpoint();
    }
    // NaN detection every 100 steps
    if (simState.globalStep % 100 === 0) {
        checkForNaN();
    }

    // Tensor leak monitoring
    if (simState.globalStep % 100 === 0) {
        const mem = tf.memory();
        if (mem.numTensors > 500) {
            console.warn(`[MEMORY] ${mem.numTensors} tensors alive, ${(mem.numBytes / 1024 / 1024).toFixed(1)}MB`);
        }
    }

    setTimeout(simulationLoop, GUI_STATE.UPDATE_DELAY_MS);
}

/**
 * Collect population and diversity metrics for the timeline.
 * Non-blocking: uses async tensor readback.
 */
function collectMetrics() {
    computePopulations(simState.grid, CONFIG.ALIVE_DIM).then(pops => {
        const ncaPops = Array.from(pops).slice(1); // Exclude sun for entropy
        const entropy = computeShannonEntropy(ncaPops);
        const loss = getLastTrainingLoss();
        runManager.recordStep(loss, Array.from(pops), entropy);
        // Update timeline if it exists
        if (typeof timelineDashboard !== 'undefined' && timelineDashboard) {
            timelineDashboard.needsRedraw = true;
        }
    });
}

/**
 * Auto-save checkpoint (single rolling slot).
 */
async function autoSaveCheckpoint() {
    try {
        await saveCheckpointToSlot('auto');
    } catch (e) {
        console.warn('Auto-save failed:', e);
    }
}

/**
 * Check for NaN in grid and auto-recover.
 */
function checkForNaN() {
    const hasNaN = tf.tidy(() => tf.any(tf.isNaN(simState.grid)));
    hasNaN.data().then(data => {
        if (data[0]) {
            console.error('[NaN detected] Attempting auto-recovery...');
            recoverFromNaN();
        }
        hasNaN.dispose();
    });
}

/**
 * Recover from NaN by restoring the most recent checkpoint.
 */
async function recoverFromNaN() {
    // Try auto-save first, then any user checkpoint
    const cpId = runManager.autoSaveId;
    if (cpId) {
        try {
            const cp = await storageManager.getCheckpoint(cpId);
            if (cp) {
                await loadCheckpointData(cp);
                showToast('Numerical instability detected — restored from checkpoint');
                return;
            }
        } catch (e) {
            console.error('Recovery failed:', e);
        }
    }
    // No checkpoint available — reset
    showToast('Numerical instability detected — resetting simulation');
    resetSimulation(false);
}


// =============================================================================
// RENDERING
// =============================================================================

function renderFrame() {
    if (!simState.grid || simState.grid.isDisposed) {
        return;
    }

    const t0 = LOG_PERFORMANCE ? performance.now() : 0;
    let t_dataSync = 0, t_copy = 0, t_texture = 0, t_render = 0;

    tf.tidy(() => {
        // Update grid texture
        let t1 = LOG_PERFORMANCE ? performance.now() : 0;
        const gridDataRaw = simState.grid.dataSync();
        const wallDataRaw = simState.wallGrid.dataSync();
        if (LOG_PERFORMANCE) t_dataSync = performance.now() - t1;

        t1 = LOG_PERFORMANCE ? performance.now() : 0;
        const gridTextureData = new Float32Array(simState.TEX_W * simState.TEX_H * 4);

        let raw_idx = 0;
        let tex_idx = 0;
        for (let y = 0; y < CONFIG.GRID_H; y++) {
            for (let x = 0; x < CONFIG.GRID_W; x++) {
                tex_idx = (y * simState.TEX_W + x * CONFIG.PACKED_PIXELS) * 4;
                for (let c = 0; c < CONFIG.CELL_DIM; c++) {
                    gridTextureData[tex_idx + c] = gridDataRaw[raw_idx + c];
                }
                raw_idx += CONFIG.CELL_DIM;
            }
        }

        const wallTextureData = new Float32Array(CONFIG.GRID_W * CONFIG.GRID_H * 4);
        let wall_idx = 0;
        for (let i = 0; i < wallDataRaw.length; i++) {
            wallTextureData[wall_idx] = wallDataRaw[i];
            wall_idx += 4;
        }
        if (LOG_PERFORMANCE) t_copy = performance.now() - t1;

        t1 = LOG_PERFORMANCE ? performance.now() : 0;
        simState.gridTex = simState.glsl({}, {
            data: gridTextureData,
            size: [simState.TEX_W, simState.TEX_H],
            format: 'rgba32f',
            tag: 'gridData'
        });

        simState.wallTex = simState.glsl({}, {
            data: wallTextureData,
            size: [CONFIG.GRID_W, CONFIG.GRID_H],
            format: 'rgba32f',
            tag: 'wallData'
        });
        if (LOG_PERFORMANCE) t_texture = performance.now() - t1;

        // Render
        t1 = LOG_PERFORMANCE ? performance.now() : 0;
        simState.renderShader.grid = simState.gridTex;
        simState.renderShader.walls = simState.wallTex;

        const canvas = document.getElementById('c');
        simState.renderShader.canvasAspect = canvas.clientWidth / canvas.clientHeight;

        simState.glsl.adjustCanvas();
        simState.glsl(simState.renderShader);
        if (LOG_PERFORMANCE) t_render = performance.now() - t1;
    });

    if (LOG_PERFORMANCE) {
        const total = performance.now() - t0;
        console.log(`[PERF] render: ${total.toFixed(1)}ms (dataSync: ${t_dataSync.toFixed(1)}, copy: ${t_copy.toFixed(1)}, texture: ${t_texture.toFixed(1)}, render: ${t_render.toFixed(1)})`);
    }
}

// --- FPS Counter ---
{
    const el = document.createElement('div');
    el.id = 'fps-counter';
    el.style.cssText = 'position:fixed; bottom:8px; right:8px; background:rgba(0,0,0,0.7); color:#0f0; font:12px "JetBrains Mono",monospace; padding:3px 8px; border-radius:4px; z-index:9999; pointer-events:none; display:none;';
    el.textContent = '— FPS';
    document.body.appendChild(el);
}
let _fpsFrames = 0, _fpsLastTime = performance.now(), _fpsDisplay = 0;
function updateFPS() {
    _fpsFrames++;
    const now = performance.now();
    if (now - _fpsLastTime >= 1000) {
        _fpsDisplay = _fpsFrames;
        _fpsFrames = 0;
        _fpsLastTime = now;
        const el = document.getElementById('fps-counter');
        if (el) el.textContent = `${_fpsDisplay} FPS`;
    }
}

function renderLoop() {
    renderFrame();
    if (GUI_STATE.SHOW_FPS) updateFPS();
    requestAnimationFrame(renderLoop);
}

// =============================================================================
// UI HELPERS
// =============================================================================

function showLoadingSpinner(message) {
    let spinner = document.getElementById('loading-spinner-overlay');
    if (!spinner) {
        spinner = document.createElement('div');
        spinner.id = 'loading-spinner-overlay';
        spinner.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(5, 5, 5, 0.85);
            gap: 20px;
        `;
        spinner.innerHTML = `
            <div class="loading-petri-ring">
                <div class="loading-petri-ring-inner"></div>
            </div>
            <p id="loading-spinner-status" style="color:#ccc;font-family:'Inter',sans-serif;font-size:14px;margin:0;letter-spacing:0.3px;text-align:center;max-width:400px;line-height:1.4;"></p>
            <p id="loading-spinner-detail" style="color:#888;font-family:'JetBrains Mono','Menlo',monospace;font-size:12px;margin:0;letter-spacing:0.2px;min-height:1.2em;"></p>
            <div class="loading-progress-track">
                <div class="loading-progress-fill"></div>
            </div>
        `;
        // Inject styles if not present
        if (!document.getElementById('loading-spinner-styles')) {
            const style = document.createElement('style');
            style.id = 'loading-spinner-styles';
            style.textContent = `
                .loading-petri-ring {
                    width: 64px; height: 64px;
                    border-radius: 50%;
                    border: 3px solid rgba(255,255,255,0.08);
                    position: relative;
                    animation: petri-pulse 2s ease-in-out infinite;
                }
                .loading-petri-ring-inner {
                    position: absolute; inset: 4px;
                    border-radius: 50%;
                    border: 3px solid transparent;
                    border-top-color: #7af;
                    border-right-color: #f4a;
                    animation: spin 1s linear infinite;
                }
                @keyframes petri-pulse {
                    0%, 100% { transform: scale(1); border-color: rgba(255,255,255,0.08); }
                    50% { transform: scale(1.08); border-color: rgba(120,170,255,0.2); }
                }
                .loading-progress-track {
                    width: 200px; height: 3px;
                    background: rgba(255,255,255,0.08);
                    border-radius: 2px;
                    overflow: hidden;
                }
                .loading-progress-fill {
                    height: 100%;
                    border-radius: 2px;
                    background: linear-gradient(90deg, #7af, #f4a, #7af);
                    background-size: 200% 100%;
                    animation: loading-shimmer 1.5s ease-in-out infinite;
                    width: 100%;
                }
                @keyframes loading-shimmer {
                    0% { background-position: 200% 0; }
                    100% { background-position: -200% 0; }
                }
            `;
            document.head.appendChild(style);
        }
        document.body.appendChild(spinner);
    }
    const statusEl = spinner.querySelector('#loading-spinner-status');
    if (statusEl) statusEl.textContent = message || '';
    spinner.style.display = 'flex';
}

function hideLoadingSpinner() {
    const spinner = document.getElementById('loading-spinner-overlay');
    if (spinner) {
        spinner.style.display = 'none';
        const detail = spinner.querySelector('#loading-spinner-detail');
        if (detail) detail.textContent = '';
    }
}

// =============================================================================
// TOAST NOTIFICATIONS
// =============================================================================

function showToast(message, duration = 3000) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = `
            position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
            z-index: 10000; display: flex; flex-direction: column; align-items: center; gap: 8px;
        `;
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.style.cssText = `
        background: rgba(20, 20, 20, 0.95); color: #e0e0e0; padding: 10px 20px;
        border-radius: 6px; font-family: 'Inter', sans-serif; font-size: 13px;
        border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(8px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4); opacity: 0; transition: opacity 0.3s;
    `;
    toast.textContent = message;
    container.appendChild(toast);
    requestAnimationFrame(() => { toast.style.opacity = '1'; });
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// =============================================================================
// AUTOMATED SCREENSHOT SYSTEM
// =============================================================================

let screenshotCounter = 0;

/**
 * Export a comprehensive data bundle as JSON.
 * Contains everything needed to rebuild figures programmatically:
 * (1) Grid species map (recolorable), (2) All hyperparams + history, (3) Timeline data.
 *
 * Press P to trigger.
 */
async function exportDataBundle(label) {
    renderFrame();

    const step = simState.globalStep;
    const tau = GUI_STATE.SOFTMAX_TEMP;
    const threshold = GUI_STATE.ALIVE_THRESHOLD;
    const lr = GUI_STATE.LEARNING_RATE;

    // (1) Extract grid species map: per-cell dominant species index (0=sun, 1-N=species)
    // Also extract raw alive channels for recoloring
    const gridData = await simState.grid.data();
    const H = CONFIG.GRID_H;
    const W = CONFIG.GRID_W;
    const ALIVE_DIM = CONFIG.ALIVE_DIM;
    const CELL_DIM = CONFIG.CELL_DIM;

    // Species map: argmax of alive channels per cell
    const speciesMap = new Uint8Array(H * W);
    // Raw alive values: [H*W, ALIVE_DIM] as flat Float32
    const aliveRaw = new Float32Array(H * W * ALIVE_DIM);

    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const cellIdx = (y * W + x) * CELL_DIM;
            let maxAlive = -1;
            let maxSpecies = 0;
            for (let s = 0; s < ALIVE_DIM; s++) {
                const val = gridData[cellIdx + s];
                aliveRaw[(y * W + x) * ALIVE_DIM + s] = val;
                if (val > maxAlive) {
                    maxAlive = val;
                    maxSpecies = s;
                }
            }
            speciesMap[y * W + x] = maxSpecies;
        }
    }

    // (2) Full hyperparameter snapshot + event history
    const config = { ...CONFIG };
    const guiState = { ...GUI_STATE };
    const events = runManager.eventLog ? runManager.eventLog.serialize() : [];

    // (3) Timeline data: full metrics history
    const metrics = runManager.metrics;
    const timelineData = {
        steps: [],
        populations: [],  // array of arrays: [[sun, s1, s2, ...], ...]
        entropy: [],
        loss: [],
    };

    for (let i = 0; i < metrics.length; i++) {
        timelineData.steps.push(metrics.getStep(i));
        timelineData.populations.push(Array.from(metrics.getPopulations(i)));
        timelineData.entropy.push(metrics.getEntropy(i));
        timelineData.loss.push(metrics.getLoss(i));
    }

    // (4) Species colors (RGB, 0-1) for reference
    const speciesColors = simState.nca_colors ? simState.nca_colors.map(c => [...c]) : [];

    // Build the bundle
    const bundle = {
        metadata: {
            step,
            seed: getCurrentSeed(),
            timestamp: new Date().toISOString(),
            gridW: W,
            gridH: H,
            nSpecies: CONFIG.N_NCAS,
            aliveDim: ALIVE_DIM,
        },
        config,
        guiState,
        events,
        speciesColors,
        // Grid data as base64 for compactness
        speciesMap: arrayToBase64(speciesMap),
        aliveChannels: float32ToBase64(aliveRaw),
        timeline: timelineData,
    };

    // Generate filename
    screenshotCounter++;
    const prefix = label || 'data';
    const optShort = (GUI_STATE.OPTIMIZER_TYPE || 'sgd').replace(/\+/g, '').toLowerCase();
    const params = `step${step}_${optShort}_tau${tau}_thr${threshold}_lr${lr}`;
    const filename = `${prefix}_${params}_${screenshotCounter}.json`;

    // Download JSON
    const blob = new Blob([JSON.stringify(bundle)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);

    showToast(`Data bundle saved: ${filename}`);
    return bundle;
}

/** Encode Uint8Array as base64 string. */
function arrayToBase64(uint8Array) {
    let binary = '';
    for (let i = 0; i < uint8Array.length; i++) {
        binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary);
}

/** Encode Float32Array as base64 string. */
function float32ToBase64(float32Array) {
    const uint8 = new Uint8Array(float32Array.buffer, float32Array.byteOffset, float32Array.byteLength);
    let binary = '';
    for (let i = 0; i < uint8.length; i++) {
        binary += String.fromCharCode(uint8[i]);
    }
    return btoa(binary);
}

/**
 * Capture just the simulation canvas (no timeline) for clean figure panels.
 * Press O to trigger.
 */
async function captureSimOnly(label) {
    renderFrame();
    const canvas = document.getElementById('c');
    const step = simState.globalStep;
    const tau = GUI_STATE.SOFTMAX_TEMP;
    const threshold = GUI_STATE.ALIVE_THRESHOLD;
    const optShort = (GUI_STATE.OPTIMIZER_TYPE || 'sgd').replace(/\+/g, '').toLowerCase();
    const filename = `${label || 'sim'}_step${step}_${optShort}_tau${tau}_thr${threshold}.png`;

    canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        showToast(`Sim screenshot: ${filename}`);
    }, 'image/png');
}

// =============================================================================
// CHECKPOINT MANAGEMENT
// =============================================================================

const MAX_USER_CHECKPOINTS = 5;

/**
 * Save a checkpoint to a slot.
 * @param {'manual'|'auto'} trigger
 */
async function saveCheckpointToSlot(trigger = 'manual') {
    if (trigger === 'manual' && runManager.getUserCheckpointCount() >= MAX_USER_CHECKPOINTS) {
        showToast(`Maximum ${MAX_USER_CHECKPOINTS} checkpoints reached. Delete one first.`);
        return null;
    }

    // Render a fresh frame so the canvas isn't blank for thumbnail
    renderFrame();

    const canvas = document.getElementById('c');
    const [gridState, modelWeights, thumbnail] = await Promise.all([
        serializeGridState(simState),
        serializeModelWeights(simState.model),
        captureThumbnail(canvas),
    ]);

    // Serialize optimizer state (getWeights returns NamedTensor[])
    let optimizerState = null;
    try {
        const optWeights = await simState.optimizer.getWeights();
        optimizerState = await Promise.all(optWeights.map(async (nt) => ({
            name: nt.name,
            shape: nt.tensor.shape.slice(),
            data: (await nt.tensor.data()).buffer.slice(0),
        })));
    } catch (e) {
        // SGD has no weights to save — that's fine
    }

    // Serialize metrics
    const metricsSnapshot = runManager.metrics.serialize();

    const checkpoint = {
        id: generateUUID(),
        runId: runManager.currentRun ? runManager.currentRun.id : 'unknown',
        step: simState.globalStep,
        label: trigger === 'auto' ? 'Auto-save' : `Step ${simState.globalStep}`,
        trigger: trigger,
        config: {
            CONFIG: { ...CONFIG },
            GUI_STATE: { ...GUI_STATE },
        },
        ...gridState,
        modelWeights: modelWeights,
        optimizerState: optimizerState,
        metricsSnapshot: metricsSnapshot,
        thumbnail: thumbnail,
        formatVersion: 1,
    };

    // For auto-save, delete the previous auto-save
    if (trigger === 'auto' && runManager.autoSaveId) {
        try {
            await storageManager.deleteCheckpoint(runManager.autoSaveId);
            runManager.checkpoints.delete(runManager.autoSaveId);
        } catch (e) { /* ignore */ }
    }

    await storageManager.saveCheckpoint(checkpoint);
    runManager.registerCheckpoint(checkpoint);

    if (trigger === 'manual') {
        showToast(`Checkpoint saved (${runManager.getUserCheckpointCount()}/${MAX_USER_CHECKPOINTS})`);
        updateCheckpointTray();
    }

    // Also save the run record
    if (runManager.currentRun) {
        await storageManager.saveRun(runManager.currentRun);
    }

    return checkpoint;
}

/**
 * Load a checkpoint and restore full simulation state.
 */
async function loadCheckpointData(checkpoint) {
    // Pause simulation
    const wasTraining = simState.isTraining;
    simState.isTraining = false;

    // Restore CONFIG and GUI_STATE
    if (checkpoint.config) {
        Object.assign(CONFIG, checkpoint.config.CONFIG);
        Object.assign(GUI_STATE, checkpoint.config.GUI_STATE);
        updateDerivedConfig();
    }

    // Restore grid state
    restoreGridState(simState, checkpoint);

    // Rebuild model if architecture params changed
    if (simState.model) simState.model.dispose();
    simState.model = createModel(CONFIG);

    // Restore model weights
    if (checkpoint.modelWeights) {
        restoreModelWeights(simState.model, checkpoint.modelWeights);
    }

    // Recreate optimizer and restore its state if available
    simState.optimizer = createOptimizer();
    if (checkpoint.optimizerState && checkpoint.optimizerState.length > 0) {
        try {
            const namedTensors = checkpoint.optimizerState.map(s => ({
                name: s.name,
                tensor: tf.tensor(new Float32Array(s.data), s.shape),
            }));
            await simState.optimizer.setWeights(namedTensors);
            // Dispose the temp tensors (setWeights copies them internally)
            namedTensors.forEach(nt => nt.tensor.dispose());
        } catch (e) {
            console.warn('Could not restore optimizer state:', e);
        }
    }

    // Restore step counter
    simState.globalStep = checkpoint.step;
    runManager.globalStep = checkpoint.step;
    simState.stepCount = 0;

    // Restore metrics (truncate to checkpoint step, or replace entirely)
    if (checkpoint.metricsSnapshot) {
        runManager.metrics = MetricsBuffer.deserialize(checkpoint.metricsSnapshot);
    } else {
        runManager.metrics.truncateToStep(checkpoint.step);
    }

    // Update colors and GUI
    simState.nca_colors = generateNCAColors(CONFIG.N_NCAS);
    simState.color_uniform = simState.nca_colors.flat();

    // Recreate textures
    simState.TEX_W = CONFIG.GRID_W * CONFIG.PACKED_PIXELS;
    simState.TEX_H = CONFIG.GRID_H;
    simState.gridTex = simState.glsl({}, {
        size: [simState.TEX_W, simState.TEX_H],
        format: 'rgba32f',
        tag: 'gridData'
    });
    simState.wallTex = simState.glsl({}, {
        size: [CONFIG.GRID_W, CONFIG.GRID_H],
        format: 'rgba32f',
        tag: 'wallData'
    });
    simState.renderShader = createRenderShader();

    // Rebuild GUI controllers to reflect restored state
    if (simState.gui) {
        rebuildGUI();
    }

    updateGridBorder();

    // Sync sun heatmap visibility with restored state
    updateSunHeatmapVisibility(GUI_STATE.SUN_MODE !== 'global');
    updateSeasonalPanelVisibility(GUI_STATE.SUN_MODE === 'seasonal');

    // Force timeline redraw immediately
    if (timelineDashboard) {
        timelineDashboard.needsRedraw = true;
        timelineDashboard.lastRenderedLength = 0;
    }

    // Reset Start/Pause button to "Start" state since we paused
    if (simState.startPauseController) {
        simState.startPauseController.name('Start');
        const btnEl = simState.startPauseController.domElement.parentElement.parentElement;
        btnEl.classList.remove('btn-pause');
        btnEl.classList.add('btn-start');
    }

    showToast(`Restored checkpoint at step ${checkpoint.step}`);
}

/**
 * Delete a user checkpoint.
 */
async function deleteCheckpoint(checkpointId) {
    await storageManager.deleteCheckpoint(checkpointId);
    runManager.checkpoints.delete(checkpointId);
    if (runManager.currentRun) {
        runManager.currentRun.checkpointIds =
            runManager.currentRun.checkpointIds.filter(id => id !== checkpointId);
        await storageManager.saveRun(runManager.currentRun);
    }
    updateCheckpointTray();
    showToast('Checkpoint deleted');
}

/**
 * Update checkpoint tray UI (stub — implemented in Step 4).
 */
function updateCheckpointTray() {
    // Will be implemented with the checkpoint tray UI
    if (typeof renderCheckpointTray === 'function') {
        renderCheckpointTray();
    }
}

/**
 * Rebuild the dat.GUI to reflect restored state.
 */
function rebuildGUI() {
    // Update all controllers to match current GUI_STATE
    if (simState.gui) {
        for (const folder of Object.values(simState.gui.__folders)) {
            for (const controller of folder.__controllers) {
                controller.updateDisplay();
            }
        }
        // Rebuild NCA draw buttons
        setupNCAButtons();
    }
}

// =============================================================================
// EXPORT / IMPORT
// =============================================================================

/**
 * Export current run as a .petri file download.
 */
async function exportRunAsPetri() {
    if (!runManager.currentRun) {
        showToast('No active run to export');
        return;
    }

    // Save current run state first
    await storageManager.saveRun(runManager.currentRun);
    await storageManager.saveEvents(runManager.currentRun.id, runManager.eventLog.serialize());

    try {
        const bundle = await exportPetriBundleForRun(runManager.currentRun.id);
        const blob = new Blob([bundle], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `petri-run-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.petri`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Run exported');
    } catch (e) {
        console.error('Export failed:', e);
        showToast('Export failed: ' + e.message);
    }
}

/**
 * Import a .petri file.
 */
async function importPetriFile(file) {
    try {
        const buffer = await file.arrayBuffer();
        const { header, binaryData } = parsePetriBundle(buffer);

        // Save the run
        if (header.run) {
            await storageManager.saveRun(header.run);
        }

        // Save events
        if (header.events && header.run) {
            await storageManager.saveEvents(header.run.id, header.events);
        }

        // Save and load first checkpoint
        if (header.checkpoints && header.checkpoints.length > 0) {
            const firstCp = extractCheckpointFromBundle(header.checkpoints[0], binaryData);
            await storageManager.saveCheckpoint(firstCp);

            // Save remaining checkpoints
            for (let i = 1; i < header.checkpoints.length; i++) {
                const cp = extractCheckpointFromBundle(header.checkpoints[i], binaryData);
                await storageManager.saveCheckpoint(cp);
            }

            // Load the last checkpoint
            const lastCpMeta = header.checkpoints[header.checkpoints.length - 1];
            const lastCp = extractCheckpointFromBundle(lastCpMeta, binaryData);
            await loadCheckpointData(lastCp);

            // Register all checkpoints with run manager
            for (const cpMeta of header.checkpoints) {
                runManager.registerCheckpoint({
                    id: cpMeta.id,
                    step: cpMeta.step,
                    label: cpMeta.label,
                    trigger: cpMeta.trigger,
                });
            }

            renderCheckpointTray();
        }

        // Load event log for replay
        if (header.events) {
            runManager.eventLog = EventLog.deserialize(header.events);
        }

        showToast(`Imported run with ${header.checkpoints?.length || 0} checkpoints`);
    } catch (e) {
        console.error('Import failed:', e);
        showToast('Import failed: ' + e.message);
    }
}

/**
 * Copy recipe URL to clipboard.
 */
// Default GUI_STATE values for recipe delta encoding
const GUI_STATE_DEFAULTS = { ...GUI_STATE };

function copyRecipeURL() {
    if (!runManager.currentRun) {
        showToast('No active run');
        return;
    }

    const recipe = createRecipe(runManager.currentRun, runManager.eventLog, GUI_STATE_DEFAULTS);
    const hash = encodeRecipeToURL(recipe);
    const url = window.location.origin + window.location.pathname + hash;

    navigator.clipboard.writeText(url).then(() => {
        showToast('Recipe URL copied to clipboard');
    }).catch(() => {
        // Fallback
        prompt('Copy this URL:', url);
    });
}

/**
 * Check URL for a recipe on load.
 */
function checkURLForRecipe() {
    const recipe = decodeRecipeFromURL(window.location.hash);
    if (!recipe) return;

    // Apply recipe config
    if (recipe.cfg) {
        Object.assign(GUI_STATE, recipe.cfg);
    }
    if (recipe.seed !== undefined) {
        initSeededRNG(recipe.seed);
    }

    // Store events for replay during simulation
    if (recipe.events && recipe.events.length > 0) {
        // Convert compact events to full event log
        for (const ev of recipe.events) {
            runManager.eventLog.logParamChange(ev.t, ev.k, undefined, ev.v);
        }
        showToast('Loaded recipe from URL — events will apply during simulation');
    }

    // Clear the hash to prevent re-loading
    history.replaceState(null, '', window.location.pathname);
}

// =============================================================================
// DRAG AND DROP IMPORT
// =============================================================================

function setupDragDropImport() {
    const body = document.body;

    body.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    body.addEventListener('drop', async (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        for (const file of files) {
            if (file.name.endsWith('.petri')) {
                await importPetriFile(file);
                return;
            }
        }
        showToast('Drop a .petri file to import');
    });
}

// =============================================================================
// CHECKPOINT TRAY UI
// =============================================================================

function renderCheckpointTray() {
    const slotsContainer = document.getElementById('checkpoint-slots');
    if (!slotsContainer) return;

    slotsContainer.innerHTML = '';

    // Get user checkpoints sorted by step
    const userCheckpoints = [];
    for (const cp of runManager.checkpoints.values()) {
        if (cp.trigger !== 'auto') userCheckpoints.push(cp);
    }
    userCheckpoints.sort((a, b) => a.step - b.step);

    // Render filled slots
    for (let i = 0; i < MAX_USER_CHECKPOINTS; i++) {
        const slot = document.createElement('div');
        slot.className = 'checkpoint-slot';

        if (i < userCheckpoints.length) {
            const cp = userCheckpoints[i];
            slot.classList.add('filled');
            slot.title = `${cp.label || 'Checkpoint'} - Step ${cp.step}`;
            slot.dataset.checkpointId = cp.id;

            // Try to load thumbnail async
            storageManager.getCheckpoint(cp.id).then(fullCp => {
                if (fullCp && fullCp.thumbnail) {
                    const url = URL.createObjectURL(fullCp.thumbnail);
                    slot.innerHTML = `<img src="${url}"><span class="cp-step">${cp.step}</span>`;
                } else {
                    slot.innerHTML = `<span style="font-size:14px;color:#b8a56a;">&#9670;</span><span class="cp-step">${cp.step}</span>`;
                }
            });

            slot.addEventListener('click', (e) => {
                showCheckpointPopover(cp, e);
            });
        } else {
            // Empty slot
            slot.innerHTML = `<span>${i + 1}</span>`;
        }

        slotsContainer.appendChild(slot);
    }
}

function showCheckpointPopover(cpMeta, event) {
    // Remove existing popovers
    document.querySelectorAll('.checkpoint-popover').forEach(p => p.remove());

    const popover = document.createElement('div');
    popover.className = 'checkpoint-popover';

    popover.innerHTML = `
        <div class="cp-name">${cpMeta.label || `Step ${cpMeta.step}`}</div>
        <div class="cp-info">Step ${cpMeta.step}</div>
        <div class="cp-actions">
            <button class="resume-btn">Resume</button>
            <button class="export-btn">Export</button>
            <button class="delete-btn">Delete</button>
        </div>
    `;

    // Load and show thumbnail
    storageManager.getCheckpoint(cpMeta.id).then(fullCp => {
        if (fullCp && fullCp.thumbnail) {
            const url = URL.createObjectURL(fullCp.thumbnail);
            const img = document.createElement('img');
            img.src = url;
            popover.insertBefore(img, popover.firstChild);
        }
    });

    popover.querySelector('.resume-btn').addEventListener('click', async () => {
        popover.remove();
        const fullCp = await storageManager.getCheckpoint(cpMeta.id);
        if (fullCp) await loadCheckpointData(fullCp);
    });

    popover.querySelector('.export-btn').addEventListener('click', async () => {
        popover.remove();
        await exportRunAsPetri();
    });

    popover.querySelector('.delete-btn').addEventListener('click', async () => {
        popover.remove();
        await deleteCheckpoint(cpMeta.id);
    });

    // Position near the clicked slot (use fixed positioning like the tray)
    popover.style.position = 'fixed';
    popover.style.left = `${event.clientX - 60}px`;
    popover.style.top = `${document.getElementById('checkpoint-tray').getBoundingClientRect().bottom + 4}px`;
    popover.style.zIndex = '300';

    document.body.appendChild(popover);

    // Close on click outside
    const closeHandler = (e) => {
        if (!popover.contains(e.target)) {
            popover.remove();
            document.removeEventListener('click', closeHandler);
        }
    };
    setTimeout(() => document.addEventListener('click', closeHandler), 0);
}

// =============================================================================
// KEYBOARD SHORTCUTS
// =============================================================================

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ignore if typing in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' ||
            e.target.isContentEditable) return;

        switch (e.key) {
            case ' ':
                // Play/pause simulation
                e.preventDefault();
                GUI_BUTTONS.startPause();
                break;
            case 'p':
            case 'P':
                // Export full data bundle (grid + params + timeline as JSON)
                e.preventDefault();
                exportDataBundle();
                break;
            case 'o':
            case 'O':
                // Screenshot sim canvas only (clean, no timeline)
                e.preventDefault();
                captureSimOnly();
                break;
            case 's':
            case 'S':
                // Save checkpoint
                e.preventDefault();
                saveCheckpointToSlot('manual');
                break;
            case '1': case '2': case '3': case '4': case '5': {
                // Select checkpoint by number
                const idx = parseInt(e.key) - 1;
                const userCps = [];
                for (const cp of runManager.checkpoints.values()) {
                    if (cp.trigger !== 'auto') userCps.push(cp);
                }
                userCps.sort((a, b) => a.step - b.step);
                if (idx < userCps.length) {
                    const slots = document.querySelectorAll('.checkpoint-slot.filled');
                    if (slots[idx]) slots[idx].click();
                }
                break;
            }
        }
    });
}

// =============================================================================
// REPLAY MODE
// =============================================================================

function setupReplayControls() {
    const controls = document.getElementById('replay-controls');
    if (!controls) return;

    document.getElementById('replay-playpause')?.addEventListener('click', () => {
        if (runManager.mode === 'REPLAY') {
            simState.isTraining = !simState.isTraining;
            if (simState.isTraining) simulationLoop();
            updateReplayDisplay();
        }
    });

    document.getElementById('replay-prev')?.addEventListener('click', () => {
        // Jump to previous checkpoint
        const cps = runManager.getCheckpointsSorted();
        const current = simState.globalStep;
        for (let i = cps.length - 1; i >= 0; i--) {
            if (cps[i].step < current) {
                jumpToCheckpoint(cps[i].id);
                return;
            }
        }
    });

    document.getElementById('replay-next')?.addEventListener('click', () => {
        // Jump to next checkpoint
        const cps = runManager.getCheckpointsSorted();
        const current = simState.globalStep;
        for (const cp of cps) {
            if (cp.step > current) {
                jumpToCheckpoint(cp.id);
                return;
            }
        }
    });

    document.getElementById('replay-back')?.addEventListener('click', () => {
        // Step back = jump to previous checkpoint (can't truly step back)
        const cps = runManager.getCheckpointsSorted();
        const current = simState.globalStep;
        for (let i = cps.length - 1; i >= 0; i--) {
            if (cps[i].step < current) {
                jumpToCheckpoint(cps[i].id);
                return;
            }
        }
    });

    document.getElementById('replay-forward')?.addEventListener('click', () => {
        // Step forward a few steps
        if (!simState.isTraining) {
            simState.isTraining = true;
            // Run a burst of steps then pause
            let stepsToRun = 10;
            const burst = () => {
                if (stepsToRun-- > 0 && simState.isTraining) {
                    simulationLoop().then(() => {
                        if (stepsToRun > 0) setTimeout(burst, 0);
                        else { simState.isTraining = false; updateReplayDisplay(); }
                    });
                }
            };
            burst();
        }
    });

    document.getElementById('replay-takeover')?.addEventListener('click', () => {
        takeoverFromReplay();
    });

    document.getElementById('replay-speed')?.addEventListener('change', (e) => {
        runManager.replaySpeed = parseInt(e.target.value);
    });
}

async function jumpToCheckpoint(checkpointId) {
    const cp = await storageManager.getCheckpoint(checkpointId);
    if (cp) {
        await loadCheckpointData(cp);
        updateReplayDisplay();
    }
}

function enterReplayMode(eventLog, maxStep) {
    runManager.enterReplayMode(eventLog, maxStep);
    simState.isTraining = false;

    // Show replay controls
    const controls = document.getElementById('replay-controls');
    if (controls) controls.style.display = 'flex';

    // Update visual indicators
    const gridBorder = document.getElementById('grid-border');
    if (gridBorder) gridBorder.style.borderColor = '#b8a56a';

    updateReplayDisplay();
    showToast('Entered replay mode');
}

function takeoverFromReplay() {
    const parentRunId = runManager.currentRun ? runManager.currentRun.id : null;
    const parentStep = simState.globalStep;

    runManager.takeover(parentRunId, parentStep, getCurrentSeed(), CONFIG, GUI_STATE);

    // Hide replay controls
    const controls = document.getElementById('replay-controls');
    if (controls) controls.style.display = 'none';

    // Restore normal visual indicators
    const gridBorder = document.getElementById('grid-border');
    if (gridBorder) gridBorder.style.borderColor = '';

    showToast(`Branched from step ${parentStep} — now in live mode`);
}

/**
 * Apply any replay events that should fire at the current step.
 */
function applyReplayEvents() {
    const events = runManager.replayEventLog.events;
    while (runManager.replayEventIdx < events.length) {
        const ev = events[runManager.replayEventIdx];
        if (ev.step > simState.globalStep) break;
        if (ev.step === simState.globalStep && ev.type === 'param') {
            if (ev.param in GUI_STATE) {
                GUI_STATE[ev.param] = ev.newValue;
                rebuildGUI();
                showToast(`Replay: ${ev.param} → ${ev.newValue}`);
            }
        }
        runManager.replayEventIdx++;
    }

    // Check if replay is complete
    if (simState.globalStep >= runManager.replayMaxStep) {
        simState.isTraining = false;
        showToast('Replay complete');
        updateReplayDisplay();
    }
}

function updateReplayDisplay() {
    const display = document.getElementById('replay-step-display');
    if (display) {
        display.textContent = `Step ${simState.globalStep} / ${runManager.replayMaxStep}`;
    }

    const playPause = document.getElementById('replay-playpause');
    if (playPause) {
        playPause.textContent = simState.isTraining ? '\u275A\u275A' : '\u25B6';
    }
}

// =============================================================================
// MAIN ENTRY
// =============================================================================

// --- Boot overlay helpers ---
function bootStatus(msg) {
    const el = document.getElementById('boot-status');
    if (el) el.textContent = msg;
}
function dismissBootOverlay() {
    const overlay = document.getElementById('boot-overlay');
    if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => overlay.remove(), 500);
    }
}
const yieldToMain = () => new Promise(r => setTimeout(r, 0));

async function main() {
    // Set TF.js WebGL optimization flags before backend initialization
    tf.env().set('WEBGL_CPU_FORWARD', false);    // Prevent CPU fallback
    tf.env().set('WEBGL_PACK', true);             // Enable texture packing
    tf.env().set('WEBGL_LAZILY_UNPACK', true);    // Defer unpacking
    tf.env().set('WEBGL_FLUSH_THRESHOLD', -1);    // Disable auto-flush
    // GPU memory management
    tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 256 * 1024 * 1024); // 256MB cap

    bootStatus('Initializing GPU backend...');
    await tf.setBackend('webgl');
    await yieldToMain();

    // Initialize IndexedDB storage
    await storageManager.init();

    const canvas = document.getElementById('c');
    const gl = canvas.getContext('webgl2', {
        alpha: false,
        antialias: true,
        preserveDrawingBuffer: true,
    });
    simState.glsl = SwissGL(gl);

    bootStatus('Building simulation...');
    await yieldToMain();
    // GUI is set up inside initializeSimulation() (guarded by guiInitialized flag)
    await initializeSimulation();
    await yieldToMain();

    setupMouseListeners();
    setupKeyboardShortcuts();

    // Initialize checkpoint tray UI
    renderCheckpointTray();

    // Set up save checkpoint button
    const saveBtn = document.getElementById('save-checkpoint-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => saveCheckpointToSlot('manual'));
    }

    // Set up replay transport controls
    setupReplayControls();

    // Update grid border on resize
    window.addEventListener('resize', () => updateGridBorder());

    // Set up drag-and-drop import
    setupDragDropImport();

    // Check URL for recipe
    checkURLForRecipe();

    console.log("Petri Dish NCA initialized. Press Start to begin.");
    dismissBootOverlay();
    renderLoop();
}

// Start application
main();
