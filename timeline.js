/**
 * Timeline Dashboard - Canvas2D multi-row visualization
 *
 * Renders: population strata, diversity line, loss line,
 * checkpoint markers, and param-change markers.
 */

// =============================================================================
// TIMELINE DASHBOARD
// =============================================================================

let timelineDashboard = null;

class TimelineDashboard {
    constructor(container, canvas) {
        this.container = container;
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.needsRedraw = true;
        this.collapsed = false;

        // Layout constants (in CSS pixels; canvas pixels = * devicePixelRatio)
        this.ROW_HEIGHTS = {
            markers: 12,
            population: 42,
            diversity: 20,
            loss: 20,
            axis: 14,
        };
        this.TOTAL_HEIGHT = 108; // Sum of rows
        this.PADDING_LEFT = 30;
        this.PADDING_RIGHT = 16;

        // Viewport: which range of steps are visible
        this.viewStart = 0;   // Leftmost step
        this.viewEnd = 1000;  // Rightmost step (auto-expands)
        this.autoScroll = true; // Pin to latest data

        // Hover state
        this.hoverX = -1;
        this.hoverStep = -1;
        this.tooltipVisible = false;

        // Cached offscreen canvas for incremental rendering
        this.offscreen = null;
        this.offscreenCtx = null;
        this.lastRenderedLength = 0;

        // Popover state
        this.activePopover = null;

        this._setupListeners();
        this._resize();
    }

    _setupListeners() {
        // Resize
        window.addEventListener('resize', () => this._resize());

        // Mouse hover for tooltip
        this.canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => this._onMouseLeave());

        // Click for checkpoint popovers
        this.canvas.addEventListener('click', (e) => this._onClick(e));

        // Scroll to zoom
        this.canvas.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });

        // Collapse toggle
        this.container.addEventListener('dblclick', () => this._toggleCollapse());
    }

    _resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.container.getBoundingClientRect();
        this.width = rect.width;
        this.height = this.collapsed ? 4 : this.TOTAL_HEIGHT;
        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.canvas.style.width = `${this.width}px`;
        this.canvas.style.height = `${this.height}px`;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.needsRedraw = true;
        this.lastRenderedLength = 0; // Force full redraw
    }

    _toggleCollapse() {
        this.collapsed = !this.collapsed;
        this.container.classList.toggle('collapsed', this.collapsed);
        this._resize();
    }

    // =========================================================================
    // RENDERING
    // =========================================================================

    render() {
        if (!this.needsRedraw) return;
        this.needsRedraw = false;

        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;

        // Clear
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, w, h);

        if (this.collapsed || !runManager || !runManager.metrics) return;

        const metrics = runManager.metrics;
        if (metrics.length === 0) return;

        // Auto-scroll: viewport grows with data, keeps latest visible
        if (this.autoScroll) {
            const latestStep = metrics.getStep(metrics.length - 1);
            const firstStep = metrics.getStep(0);
            const dataRange = latestStep - firstStep;
            // Viewport = data range + 15% padding on the right
            const padding = Math.max(20, dataRange * 0.15);
            this.viewStart = firstStep;
            this.viewEnd = latestStep + padding;
        }

        const drawW = w - this.PADDING_LEFT - this.PADDING_RIGHT;
        let y = 0;

        // Row 1: Markers (checkpoints + param changes)
        this._renderMarkers(ctx, this.PADDING_LEFT, y, drawW, this.ROW_HEIGHTS.markers);
        y += this.ROW_HEIGHTS.markers;

        // Row 2: Population strata (stacked area)
        this._renderRowLabel(ctx, 'POP', this.PADDING_LEFT, y, this.ROW_HEIGHTS.population);
        this._renderPopulationStrata(ctx, this.PADDING_LEFT, y, drawW, this.ROW_HEIGHTS.population);
        y += this.ROW_HEIGHTS.population;

        this._renderSeparator(ctx, this.PADDING_LEFT, y, drawW);

        // Row 3: Diversity line
        this._renderRowLabel(ctx, 'DIV', this.PADDING_LEFT, y, this.ROW_HEIGHTS.diversity);
        this._renderSparkline(ctx, this.PADDING_LEFT, y, drawW, this.ROW_HEIGHTS.diversity,
            (i) => metrics.getEntropy(i), 'rgba(255,255,255,0.7)', 0, 1);
        y += this.ROW_HEIGHTS.diversity;

        this._renderSeparator(ctx, this.PADDING_LEFT, y, drawW);

        // Row 4: Loss line
        this._renderRowLabel(ctx, 'LOSS', this.PADDING_LEFT, y, this.ROW_HEIGHTS.loss);
        this._renderLossLine(ctx, this.PADDING_LEFT, y, drawW, this.ROW_HEIGHTS.loss);
        y += this.ROW_HEIGHTS.loss;

        this._renderSeparator(ctx, this.PADDING_LEFT, y, drawW);

        // Row 5: Step axis
        this._renderAxis(ctx, this.PADDING_LEFT, y, drawW, this.ROW_HEIGHTS.axis);

        // Hover crosshair
        if (this.hoverX >= 0) {
            ctx.strokeStyle = 'rgba(255,255,255,0.2)';
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(this.hoverX, 0);
            ctx.lineTo(this.hoverX, h);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Recording indicator
        this._renderRecIndicator(ctx, w - 50, 3);
    }

    // =========================================================================
    // POPULATION STRATA (Stacked Area Chart)
    // =========================================================================

    _renderPopulationStrata(ctx, x, y, w, h) {
        const metrics = runManager.metrics;
        const n = metrics.length;
        if (n === 0) return;

        const nSpecies = metrics.speciesCount + 1; // +1 for sun
        const colors = this._getSpeciesColors();
        const viewRange = this.viewEnd - this.viewStart;
        if (viewRange <= 0) return;

        // Pre-compute a mapping from data index to screen X range
        // so we can draw each data point as a filled column with no gaps
        const firstStep = metrics.getStep(0);
        const lastStep = metrics.getStep(n - 1);
        const dataRange = lastStep - firstStep;

        // For each pixel column, find the data point and draw it
        for (let px = 0; px < Math.ceil(w); px++) {
            const step = this.viewStart + (px / w) * viewRange;

            // Map step to data index (linear interpolation since steps are ~evenly spaced)
            let dataIdx;
            if (dataRange <= 0) {
                dataIdx = 0;
            } else {
                dataIdx = Math.round(((step - firstStep) / dataRange) * (n - 1));
                dataIdx = Math.max(0, Math.min(n - 1, dataIdx));
            }

            const pops = metrics.getPopulations(dataIdx);

            let total = 0;
            for (let i = 0; i < pops.length; i++) total += pops[i];
            if (total < 1e-8) continue;

            // Draw stacked bands bottom-to-top, 1px wide
            let cumY = y + h;
            for (let i = 0; i < Math.min(pops.length, nSpecies); i++) {
                const frac = pops[i] / total;
                const bandH = frac * h;
                if (bandH < 0.3) continue;

                ctx.fillStyle = colors[i] || 'rgba(40,40,40,0.6)';
                ctx.fillRect(x + px, cumY - bandH, 1, bandH);
                cumY -= bandH;
            }
        }
    }

    // =========================================================================
    // SPARKLINE RENDERER
    // =========================================================================

    _renderSparkline(ctx, x, y, w, h, valueFn, color, minVal, maxVal) {
        const metrics = runManager.metrics;
        const n = metrics.length;
        if (n < 2) return;

        // Auto-range if not specified
        if (minVal === undefined || maxVal === undefined) {
            minVal = Infinity;
            maxVal = -Infinity;
            for (let i = 0; i < n; i++) {
                const v = valueFn(i);
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
            if (maxVal - minVal < 1e-8) { maxVal = minVal + 1; }
        }

        const range = maxVal - minVal || 1;
        const padding = 2;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();

        const numPoints = Math.min(w, n);
        const stepSize = n / numPoints;
        let started = false;

        for (let px = 0; px < numPoints; px++) {
            const dataIdx = Math.min(Math.floor(px * stepSize), n - 1);
            const step = metrics.getStep(dataIdx);
            if (step < this.viewStart || step > this.viewEnd) continue;

            const screenX = x + this._stepToX(step, w);
            const val = valueFn(dataIdx);
            const screenY = y + padding + (1 - (val - minVal) / range) * (h - 2 * padding);

            if (!started) {
                ctx.moveTo(screenX, screenY);
                started = true;
            } else {
                ctx.lineTo(screenX, screenY);
            }
        }

        ctx.stroke();
        ctx.lineWidth = 1;
    }

    _renderLossLine(ctx, x, y, w, h) {
        const metrics = runManager.metrics;
        const n = metrics.length;
        if (n < 2) return;

        // Auto-range loss
        let minLoss = Infinity, maxLoss = -Infinity;
        for (let i = Math.max(0, n - 1000); i < n; i++) {
            const v = metrics.getLoss(i);
            if (isFinite(v)) {
                if (v < minLoss) minLoss = v;
                if (v > maxLoss) maxLoss = v;
            }
        }
        if (!isFinite(minLoss)) minLoss = 0;
        if (!isFinite(maxLoss) || maxLoss <= minLoss) maxLoss = minLoss + 1;

        this._renderSparkline(ctx, x, y, w, h,
            (i) => metrics.getLoss(i), 'rgba(210, 153, 34, 0.7)', minLoss, maxLoss);
    }

    // =========================================================================
    // MARKERS ROW
    // =========================================================================

    _renderMarkers(ctx, x, y, w, h) {
        const midY = y + h / 2;

        // Checkpoint diamonds
        const checkpoints = runManager.getCheckpointsSorted();
        for (const cp of checkpoints) {
            if (cp.trigger === 'auto') continue; // Only show user checkpoints
            const sx = x + this._stepToX(cp.step, w);
            if (sx < x || sx > x + w) continue;

            ctx.fillStyle = '#b8a56a';
            ctx.beginPath();
            ctx.moveTo(sx, midY - 4);
            ctx.lineTo(sx + 4, midY);
            ctx.lineTo(sx, midY + 4);
            ctx.lineTo(sx - 4, midY);
            ctx.closePath();
            ctx.fill();
        }

        // Param change triangles
        if (runManager.eventLog) {
            const paramChanges = runManager.eventLog.getParamChanges();
            for (const ev of paramChanges) {
                const sx = x + this._stepToX(ev.step, w);
                if (sx < x || sx > x + w) continue;

                ctx.fillStyle = 'rgba(255,255,255,0.3)';
                ctx.beginPath();
                ctx.moveTo(sx - 3, midY + 4);
                ctx.lineTo(sx + 3, midY + 4);
                ctx.lineTo(sx, midY - 2);
                ctx.closePath();
                ctx.fill();
            }
        }
    }

    // =========================================================================
    // ROW SEPARATORS & LABELS
    // =========================================================================

    _renderSeparator(ctx, x, y, w) {
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y + 0.5);
        ctx.lineTo(x + w, y + 0.5);
        ctx.stroke();
    }

    _renderRowLabel(ctx, label, x, y, rowHeight) {
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '500 8px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x + 2, y + rowHeight / 2);
    }

    // =========================================================================
    // STEP AXIS
    // =========================================================================

    _renderAxis(ctx, x, y, w, h) {
        const range = this.viewEnd - this.viewStart;
        if (range <= 0) return;

        // Determine tick interval
        const targetTicks = Math.floor(w / 80);
        const rawInterval = range / targetTicks;
        const niceInterval = this._niceInterval(rawInterval);

        ctx.fillStyle = '#555';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        const firstTick = Math.ceil(this.viewStart / niceInterval) * niceInterval;
        for (let step = firstTick; step <= this.viewEnd; step += niceInterval) {
            const sx = x + this._stepToX(step, w);

            // Tick mark
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.beginPath();
            ctx.moveTo(sx, y);
            ctx.lineTo(sx, y + 3);
            ctx.stroke();

            // Label
            ctx.fillText(this._formatStep(step), sx, y + 4);
        }
    }

    _niceInterval(raw) {
        const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
        const residual = raw / magnitude;
        if (residual <= 1.5) return magnitude;
        if (residual <= 3.5) return 2 * magnitude;
        if (residual <= 7.5) return 5 * magnitude;
        return 10 * magnitude;
    }

    _formatStep(step) {
        if (step >= 1000000) return `${(step / 1000000).toFixed(1)}M`;
        if (step >= 1000) return `${(step / 1000).toFixed(step >= 10000 ? 0 : 1)}k`;
        return `${step}`;
    }

    // =========================================================================
    // RECORDING INDICATOR
    // =========================================================================

    _renderRecIndicator(ctx, x, y) {
        if (runManager.mode !== 'LIVE') {
            // Replay indicator
            ctx.fillStyle = '#b8a56a';
            ctx.font = '500 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText('REPLAY', x + 46, y + 9);
            return;
        }

        // Recording dot
        const alpha = 0.7 + 0.3 * Math.sin(Date.now() / 1000 * Math.PI);
        ctx.fillStyle = `rgba(255, 59, 48, ${alpha})`;
        ctx.beginPath();
        ctx.arc(x, y + 6, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#ff3b30';
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText('REC', x + 8, y + 9);
    }

    // =========================================================================
    // COORDINATE HELPERS
    // =========================================================================

    _stepToX(step, drawWidth) {
        const range = this.viewEnd - this.viewStart;
        if (range <= 0) return 0;
        return ((step - this.viewStart) / range) * drawWidth;
    }

    _xToStep(screenX) {
        const drawW = this.width - this.PADDING_LEFT - this.PADDING_RIGHT;
        const localX = screenX - this.PADDING_LEFT;
        const frac = localX / drawW;
        return Math.round(this.viewStart + frac * (this.viewEnd - this.viewStart));
    }

    _getSpeciesColors() {
        // Get colors from simState if available, with desaturation for strata
        const colors = ['rgba(40, 40, 40, 0.6)']; // Sun = dark grey
        if (typeof simState !== 'undefined' && simState.nca_colors) {
            for (const [r, g, b] of simState.nca_colors) {
                colors.push(`rgba(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(b*255)}, 0.85)`);
            }
        }
        return colors;
    }

    // =========================================================================
    // INTERACTION
    // =========================================================================

    _onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.hoverX = e.clientX - rect.left;
        this.hoverStep = this._xToStep(this.hoverX);
        this._updateTooltip(e.clientX, e.clientY);
        this.needsRedraw = true;
    }

    _onMouseLeave() {
        this.hoverX = -1;
        this.hoverStep = -1;
        this._hideTooltip();
        this.needsRedraw = true;
    }

    _onClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickStep = this._xToStep(clickX);

        // Check if clicked near a checkpoint marker
        const checkpoints = runManager.getCheckpointsSorted();
        const drawW = this.width - this.PADDING_LEFT - this.PADDING_RIGHT;
        for (const cp of checkpoints) {
            if (cp.trigger === 'auto') continue;
            const cpX = this.PADDING_LEFT + this._stepToX(cp.step, drawW);
            if (Math.abs(clickX - cpX) < 8) {
                this._showCheckpointPopover(cp, e.clientX, e.clientY);
                return;
            }
        }

        // Close any open popover
        this._closePopover();
    }

    _onWheel(e) {
        e.preventDefault();
        const zoomFactor = e.deltaY > 0 ? 1.15 : 0.87;
        const range = this.viewEnd - this.viewStart;
        const mouseStep = this._xToStep(e.clientX - this.canvas.getBoundingClientRect().left);

        // Zoom centered on mouse position
        const leftFrac = (mouseStep - this.viewStart) / range;
        const newRange = range * zoomFactor;
        this.viewStart = mouseStep - leftFrac * newRange;
        this.viewEnd = this.viewStart + newRange;

        // Clamp
        if (this.viewStart < 0) {
            this.viewStart = 0;
            this.viewEnd = newRange;
        }

        // Disable auto-scroll when user zooms
        this.autoScroll = false;
        this.needsRedraw = true;
        this.lastRenderedLength = 0;

        // Re-enable auto-scroll if we're near the end
        const metrics = runManager.metrics;
        if (metrics.length > 0) {
            const latestStep = metrics.getStep(metrics.length - 1);
            if (this.viewEnd >= latestStep * 0.95) {
                this.autoScroll = true;
            }
        }
    }

    // =========================================================================
    // TOOLTIP
    // =========================================================================

    _updateTooltip(clientX, clientY) {
        const metrics = runManager.metrics;
        if (!metrics || metrics.length === 0 || this.hoverStep < 0) {
            this._hideTooltip();
            return;
        }

        // Find closest data point
        let closestIdx = 0;
        let closestDist = Infinity;
        for (let i = 0; i < metrics.length; i++) {
            const dist = Math.abs(metrics.getStep(i) - this.hoverStep);
            if (dist < closestDist) {
                closestDist = dist;
                closestIdx = i;
            }
        }

        const step = metrics.getStep(closestIdx);
        const loss = metrics.getLoss(closestIdx);
        const entropy = metrics.getEntropy(closestIdx);
        const pops = metrics.getPopulations(closestIdx);

        let total = 0;
        for (let i = 0; i < pops.length; i++) total += pops[i];

        // Build tooltip HTML
        let html = `<div style="margin-bottom:4px;color:#e0e0e0;font-weight:500;">Step ${this._formatStep(step)}</div>`;
        html += `<div>Diversity: ${entropy.toFixed(3)}</div>`;
        html += `<div>Loss: ${loss.toFixed(4)}</div>`;
        html += `<div style="margin-top:4px;">`;

        const colors = this._getSpeciesColors();
        const labels = ['Sun'];
        for (let i = 1; i < pops.length; i++) labels.push(`S${i}`);

        for (let i = 0; i < pops.length; i++) {
            const pct = total > 0 ? (pops[i] / total * 100).toFixed(1) : '0.0';
            html += `<span style="color:${colors[i] || '#666'}">${labels[i]}: ${pct}%</span> `;
        }
        html += `</div>`;

        // Check for param changes near this step
        if (runManager.eventLog) {
            const events = runManager.eventLog.getParamChanges();
            for (const ev of events) {
                if (Math.abs(ev.step - step) < (this.viewEnd - this.viewStart) * 0.02) {
                    html += `<div style="margin-top:4px;color:#b8a56a;">` +
                        `${ev.param}: ${ev.oldValue} → ${ev.newValue}</div>`;
                }
            }
        }

        const tooltip = document.getElementById('timeline-tooltip');
        tooltip.innerHTML = html;
        tooltip.style.display = 'block';

        // Position tooltip
        const rect = this.container.getBoundingClientRect();
        let left = clientX - rect.left + 12;
        if (left + 200 > this.width) left = clientX - rect.left - 200;
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${this.TOTAL_HEIGHT + 4}px`;
    }

    _hideTooltip() {
        const tooltip = document.getElementById('timeline-tooltip');
        if (tooltip) tooltip.style.display = 'none';
    }

    // =========================================================================
    // CHECKPOINT POPOVER
    // =========================================================================

    async _showCheckpointPopover(cpMeta, clientX, clientY) {
        this._closePopover();

        // Load full checkpoint data for thumbnail
        const cp = await storageManager.getCheckpoint(cpMeta.id);
        if (!cp) return;

        const popover = document.createElement('div');
        popover.className = 'checkpoint-popover';

        // Thumbnail
        let thumbHtml = '';
        if (cp.thumbnail) {
            const url = URL.createObjectURL(cp.thumbnail);
            thumbHtml = `<img src="${url}" alt="Checkpoint">`;
        }

        popover.innerHTML = `
            ${thumbHtml}
            <div class="cp-name" contenteditable="true">${cp.label || `Step ${cp.step}`}</div>
            <div class="cp-info">Step ${cp.step}</div>
            <div class="cp-actions">
                <button class="resume-btn">Resume</button>
                <button class="delete-btn">Delete</button>
            </div>
        `;

        // Position
        const rect = this.container.getBoundingClientRect();
        popover.style.left = `${clientX - rect.left - 90}px`;
        popover.style.top = `${this.TOTAL_HEIGHT + 8}px`;

        // Event handlers
        popover.querySelector('.resume-btn').addEventListener('click', async () => {
            this._closePopover();
            const fullCp = await storageManager.getCheckpoint(cpMeta.id);
            if (fullCp) await loadCheckpointData(fullCp);
        });

        popover.querySelector('.delete-btn').addEventListener('click', async () => {
            this._closePopover();
            await deleteCheckpoint(cpMeta.id);
        });

        // Close on click outside
        popover._closeHandler = (e) => {
            if (!popover.contains(e.target)) this._closePopover();
        };
        setTimeout(() => document.addEventListener('click', popover._closeHandler), 0);

        this.container.appendChild(popover);
        this.activePopover = popover;
    }

    _closePopover() {
        if (this.activePopover) {
            if (this.activePopover._closeHandler) {
                document.removeEventListener('click', this.activePopover._closeHandler);
            }
            this.activePopover.remove();
            this.activePopover = null;
        }
    }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

function initTimeline() {
    const container = document.getElementById('timeline-container');
    const canvas = document.getElementById('timeline-canvas');
    if (!container || !canvas) return;

    timelineDashboard = new TimelineDashboard(container, canvas);

    // Render loop — piggyback on requestAnimationFrame
    function timelineRenderLoop() {
        if (timelineDashboard) {
            // Force redraw periodically when simulation is running
            if (simState && simState.isTraining) {
                timelineDashboard.needsRedraw = true;
            }
            timelineDashboard.render();
        }
        requestAnimationFrame(timelineRenderLoop);
    }
    requestAnimationFrame(timelineRenderLoop);
}

// Auto-init when DOM is ready (but after main.js loads)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Delay slightly to ensure main.js has initialized
        setTimeout(initTimeline, 100);
    });
} else {
    setTimeout(initTimeline, 100);
}
