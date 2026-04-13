# Digital Ecosystem

An interactive artificial life simulation where neural cellular automata species compete for territory on a 2D grid. Runs entirely in the browser.

**[Read the blog post](https://pub.sakana.ai/digital-ecosystem)** &middot; **[Try the live demo](https://pub.sakana.ai/digital-ecosystem/app)**

## Run locally

Open `index.html` in a browser. That's it — no server, no build step, no dependencies to install.

TensorFlow.js loads from CDN, so you need an internet connection on first load. Chrome or Edge tend to give the best WebGL2 performance.

If your browser blocks local file access you can serve it:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

Click **Load Showcase** to load a pre-trained ecosystem — it starts automatically.

## What this is

Multiple NCA species — each a small convolutional network — share a grid and fight over it. Every species proposes per-pixel updates, and a competition mechanism decides who wins each cell: each species carries attack and defence vectors, and winners are picked by cosine similarity between attackers and defenders, sharpened through a softmax. Competition is soft and differentiable, so coexistence is a valid outcome, not just winner-take-all.

The networks train *during* the simulation via gradient descent. Species adapt their strategies in real time. You can watch ecosystems form, collapse, stabilise, and re-emerge — then intervene by drawing walls, seeding new species, erasing territory, or tuning any of 20+ parameters through the control panel.

This builds on the [Petri Dish NCA](https://pub.sakana.ai/pdnca) framework (Zhang, Risi & Darlow, 2025), adding six algorithmic updates for stability and richer dynamics: presence gating, emergency respawn, a differentiable growth gate, spatial concentration, win-rate feedback, and a soft-minimum loss function. The [blog post](https://pub.sakana.ai/digital-ecosystem) walks through these in detail with five case studies exploring phenomena like edge-of-chaos transitions, emergent cooperation, and the surprisingly large effect of optimiser choice.

## How it works

Each grid cell stores a state vector with three parts:

- **Aliveness channels** — one per species plus the "sun", a learnable neutral entity that fills empty space
- **Attack and defence vectors** — half of `CELL_STATE_DIM` each (16 per side by default), used for competition
- **Hidden state** — 2 channels of recurrent memory carried between steps

Each simulation step runs:

1. Every species' network proposes an update for each cell (3&times;3 encoder convolution followed by inverted residual blocks with depthwise-separable convolutions)
2. Competition: cosine similarity between each attacker's attack vector and each defender's defence vector &rarr; softmax over similarity scores &rarr; weighted combination of proposed updates
3. Loss: maximise own population via soft-minimum over per-species aliveness, with an entropy bonus to prevent any single species from dominating
4. Backpropagation through the step updates each species' network weights

The sun acts as a baseline competitor — empty space defaults to sun control, so species must actively outcompete it to claim territory. A growth gate (sigmoid with tunable steepness) controls which cells are eligible for update, acting as a Langton-&lambda;-style knob between frozen, critical, and chaotic regimes.

## Repository structure

```
index.html              UI layout and styling
main.js                 Application controller, rendering loop, user interaction
petri-dish.js           NCA simulation engine: model creation, competition, training
swissgl.js              WebGL2 rendering library (third-party)
dat.gui.min.js          GUI control widgets (third-party)
recording.js            Video recording and .petri checkpoint serialisation
timeline.js             Timeline visualisation component
timeline-store.js       Timeline state management, PRNG, checkpoint persistence
presets/
  default-ecosystem.petri   Pre-trained ecosystem checkpoint (~4.7 MB)
```

## Citation

```bibtex
@misc{darlow2026digitalecosystems,
  title   = {Digital Ecosystems: Interactive Multi-Agent Neural Cellular Automata},
  author  = {Luke Darlow},
  year    = {2026},
  url     = {https://pub.sakana.ai/digital-ecosystem}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
