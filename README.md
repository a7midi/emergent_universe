Emergent Universe Simulation Suite
This software suite is a computational laboratory designed to explore the theories and predictions presented in the trilogy of papers by A. Alayar on deterministic causal sites. Its primary goal is to provide a practical, empirical test of the central claim: that stable, particle-like structures can spontaneously emerge from a simple, finite, and purely deterministic computational substrate.

Where the papers provide the analytical proofs, this suite provides the means to run controlled experiments, observe the emergent phenomena, and discover the specific conditions under which they occur.

Core Concepts & Architecture
The architecture of this program is a direct, one-to-one implementation of the core theoretical objects described in the papers.

The Causal Site (C) → src/causal_site.py
The foundational structure of the universe, the finite acyclic causal site, is instantiated by the CausalSite class. It procedurally generates a layered directed graph that respects the key "Standing Hypotheses" of the theory, such as finite height and unit-grade arrows.

The Update Functor (T) → src/state_manager.py
The deterministic evolution of the universe is handled by the StateManager. Its tick() method is a single application of the update functor T to the entire system. It contains different fusion_rule implementations, which are the concrete "laws of physics" that determine how a site's state evolves based on its causal predecessors.

The Particle (T-invariant, indecomposable subobject) → src/particle_detector.py
This is the discovery engine of the simulation. The ParticleDetector class is an algorithm designed to find emergent particles by directly implementing their mathematical definition:

T-invariance (Stability): It hashes the state of local regions (cells) at each tick and maintains a history to find repeating, cyclical patterns (loops).

Indecomposability (Coherence): Once looping cells are found, it uses a graph traversal algorithm (find_connected_clusters) to group adjacent, co-cycling cells into single, coherent objects.

Getting Started
1. Environment Setup
This project uses a Python virtual environment to manage dependencies.

a. Create the Virtual Environment:
Open your terminal in the project's root directory and run:

python -m venv venv

b. Activate the Environment:

On Windows (PowerShell):

.\venv\Scripts\Activate.ps1

On macOS/Linux:

source venv/bin/activate

c. Install Dependencies:
The required packages are listed in requirements.txt. Install them with:

pip install -r requirements.txt

2. Running the Simulation
All simulation parameters are controlled by the config.yaml file. The main entry point is main.py.

To run a simulation, execute the following command from the project's root directory:

python main.py

A progress bar will show the simulation running. During the run, it will print diagnostic logs based on the log_interval setting. After completion, a final report will summarize any stable particles found at the very end of the simulation.

The Experimental Workflow: Finding Particles
The search for emergent particles is an experimental process. A "null result" (no particles found) is a valid and important data point. The key is to tune the parameters in config.yaml to create a "habitable zone" where stability and complexity are balanced.

Key Parameters in config.yaml
simulation.hidden_noise: The most critical setting.

true: Creates a "hot" universe with a constant influx of new information. Good for observing transient, "virtual" particles.

false: Creates a "cold," fully deterministic universe after t=0. This is the ideal environment to search for truly stable, long-lived particles.

tags.fusion_mode: The "law of physics."

"injective": A highly complex rule that is faithful to the theory's history-preserving dynamics but makes loops very rare.

"sum_mod_q": A simpler, less chaotic rule that is more conducive to forming stable patterns.

tags.alphabet_size_q: The information capacity of each site. Our experiments suggest optimal values are often prime numbers (e.g., 3, 11, 17).

causal_site.max_out_degree_R: A hard cap on the number of causal predecessors a site can have. This is a crucial parameter for controlling chaos and promoting stability. R=2 has been found to be a consistently effective setting.

detector.max_history_length: The memory of the particle detector. If you are looking for particles with long periods, this value must be larger than the expected period.

Recommended Starter Configuration
The following configuration from our experiments has been shown to be effective at producing particles. It creates a simple, low-noise environment, which is the perfect starting point for any new particle hunt.

# "Small-alphabet, low-noise" baseline from our experiments
simulation:
  total_ticks: 20000
  seed: 42
  hidden_noise: false
  log_interval: 100

causal_site:
  layers: 30
  avg_nodes_per_layer: 20
  edge_probability: 0.05

tags:
  alphabet_size_q: 3
  fusion_mode: "sum_mod_q"
  max_out_degree_R: 2

detector:
  grid_size: 8
  max_history_length: 4000
  min_loop_period: 5
  min_particle_size: 2

By systematically adjusting these parameters, you can explore the rich and complex behavior of this
