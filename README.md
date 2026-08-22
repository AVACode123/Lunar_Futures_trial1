# Lunar Futures

**Exploring rare cooperative futures in lunar geopolitics with LLM-based
multi-agent simulation**

Lunar Futures is an experimental multi-agent simulation for exploring
how geopolitical competition around the lunar south pole might evolve
over time.

The current prototype places **the United States, China, and Japan** in
a shared lunar environment. Each country is represented by an LLM agent
with public goals, strategic interests, and behavioral tendencies.
Agents repeatedly choose actions while the simulated world changes in
response to their decisions and occasional external events.

Rather than predicting a single future, the project generates **many
possible worldlines from the same initial conditions** and searches for
unusual trajectories --- especially rare cases in which a competitive
system moves toward cooperation, shared infrastructure, scientific
openness, or sustained exploration.

## Core idea: Rare Futures

A conventional scenario exercise usually asks:

> What is likely to happen?

Lunar Futures asks a complementary question:

> Among many simulated futures, are there rare but desirable
> trajectories --- and what conditions allowed them to emerge?

Most simulations may converge toward familiar patterns such as strategic
competition, resource claims, surveillance, and declining neutral
access.

Occasionally, however, an external event or an unexpected sequence of
agent decisions may push a worldline into a less frequently explored
region of the state space.

These outlier trajectories are treated as **Rare Future candidates**.

The long-term goal is to trace them backward and identify combinations
of events, policies, infrastructure, and interactions that may increase
the probability of reaching desirable futures.

## Prototype simulation

The current prototype uses three LLM agents:

  -----------------------------------------------------------------------
  Agent                               Simplified role
  ----------------------------------- -----------------------------------
  United States                       Secure access to lunar resources
                                      and prevent strategic dominance by
                                      competitors

  China                               Maintain long-term access to
                                      resources and strategic lunar
                                      infrastructure

  Japan                               Promote sustainable development
                                      while becoming an important
                                      provider of shared infrastructure
                                      and trusted services
  -----------------------------------------------------------------------

The agents currently choose among actions such as:

-   claiming a resource zone
-   expanding surveillance
-   proposing joint mining
-   proposing a shared rescue network
-   sharing scientific data
-   investing in Mars exploration
-   forming a coalition
-   waiting and observing

The current implementation uses **Qwen3 8B through Ollama** as the local
LLM.

## World state

Each turn is represented by a multidimensional state vector. The present
prototype tracks variables including:

-   US--China tension
-   shared lunar infrastructure
-   scientific openness
-   Mars exploration progress
-   neutral access to lunar resources/infrastructure
-   US power
-   China power
-   US public willingness to cooperate
-   Chinese public willingness to cooperate
-   strength of a potential third force
-   international trust

These variables are deliberately simplified. The purpose of the
prototype is not yet to construct a validated geopolitical model, but to
test whether LLM agents can generate diverse, interpretable trajectories
that can later be analyzed systematically.

## Event Agent

Each turn also includes a stochastic external-event process.

Most turns contain no major event. Occasionally, the simulation
introduces events such as:

-   rising US--China tensions on Earth
-   discovery of new lunar ice resources
-   an accident at a lunar base
-   successful low-cost lunar transportation by India
-   a Japanese technology breakthrough
-   a major scientific discovery
-   a breakthrough relevant to Mars exploration
-   a rare, initially unspecified event

The event does **not** determine the agents' response. Instead, the
event changes the context, after which the LLM agents decide what to do.

This distinction is important: the same accident, discovery, or
geopolitical shock can potentially lead to cooperation, exploitation,
escalation, or the emergence of a third-party mediator.

## Simulation flow

``` text
Same initial state
        |
        v
 External event?
        |
        v
 United States acts
        |
        v
 China acts
        |
        v
 Japan acts
        |
        v
 World state changes
        |
        v
 Earth-side factors drift
        |
        v
 Next turn
```

Repeating this process produces one **worldline**.

Repeating the entire simulation from the same initial state produces an
ensemble of possible futures.

## Initial experiment

As a proof of concept, we generated **10 worldlines** and recorded the
world state after each turn.

The state vectors were standardized and projected into two dimensions
using **principal component analysis (PCA)**.

In this initial dataset:

-   PC1 explained approximately **34.6%** of the variance.
-   PC2 explained approximately **25.6%**.
-   Together, the first two components represented approximately **60%**
    of the observed variation.

Even from identical initial conditions, the trajectories separated
substantially.

Some worldlines moved toward high tension and restricted neutral access,
while others moved toward stronger shared infrastructure, trust, and
cooperation. External-event points could also be mapped onto the
trajectories, allowing us to inspect when a worldline began to diverge.

![PCA trajectories of simulated lunar
futures](results/lunar_futures_10_worldlines_pca.png)

This is a small proof-of-concept dataset and should **not** be
interpreted as a geopolitical prediction. The purpose is to demonstrate
a computational framework for generating and exploring alternative
futures.

## From outliers to explanations

The next step is not simply to identify statistical outliers.

We are interested in trajectories that are both:

**Novel** --- they enter regions rarely visited by other simulations.

**Desirable** --- they exhibit combinations such as lower geopolitical
tension, stronger international trust, greater neutral access, shared
infrastructure, scientific openness, and continued progress toward
deeper-space exploration.

Conceptually:

``` text
Rare Future score ≈ Novelty × Desirability
```

Once such a trajectory is found, its history can be traced backward:

``` text
Rare cooperative future
        ^
        |
Multilateral infrastructure emerges
        ^
        |
Unexpected cooperation after a crisis
        ^
        |
External event / policy decision
        ^
        |
Earlier geopolitical conditions
```

Across hundreds or thousands of simulations, recurring patterns in
successful outliers may reveal candidate **design principles for
cooperation**.

## Why GPU-scale simulation?

The present experiment uses a small local model and only a limited
number of agents and worldlines.

The intended next stage is to scale toward:

-   hundreds to thousands of worldlines
-   longer simulations
-   more diverse stochastic events
-   multiple actors within each country or organization
-   additional countries, companies, and international organizations
-   richer memory and negotiation
-   LLM-generated rare events
-   PCA / UMAP state-space mapping
-   trajectory clustering
-   density estimation
-   automated Rare Future detection
-   backward analysis of successful trajectories

Large-scale parallel inference would make it possible to explore a much
broader region of the future state space.

## Repository structure

A suggested repository layout is:

``` text
lunar-futures/
├── README.md
├── README_JAPANESE.md
├── agents.json
├── simulation.py
├── batch_simulation.py
├── plot_futures.py
├── results/
│   ├── lunar_futures_10_worldlines_pca.png
│   ├── lunar_futures_10_worldlines_pca.csv
│   └── lunar_futures_pca_loadings.csv
└── runs/
    └── example simulation logs
```

## Running the prototype

### Requirements

-   Python 3
-   Ollama
-   Qwen3 8B (current prototype)
-   `requests`
-   `pandas`
-   `scikit-learn`
-   `matplotlib`

Install the Python packages:

``` bash
python -m pip install requests pandas scikit-learn matplotlib
```

Install/pull the model in Ollama:

``` bash
ollama pull qwen3:8b
```

Run a single simulation:

``` bash
python simulation.py
```

Run multiple worldlines:

``` bash
python batch_simulation.py
```

Visualize the resulting state-space trajectories:

``` bash
python plot_futures.py
```

## Trial1-1000x production snapshot

Trial1-1000x is the reproducible production snapshot that expands the
earlier 100-worldline Trial1 batch to **1,000 worldlines**.

-   1,000 worldlines × 10 turns
-   3 agents per turn, always ordered USA → China → Japan
-   30,000 LLM decisions
-   7 × NVIDIA RTX A5000
-   7 independent Ollama replicas with worldline-level parallelism
-   Model: Qwen3 8B, Q4_K_M
-   Master seed: `20260821`
-   No missing or duplicate run IDs
-   All runs passed merge validation
-   `merged/all_states.csv`: 11,000 state rows

### Fixed provenance

-   Ollama version: `0.32.15`
-   Ollama image:
    `ollama/ollama@sha256:57d60e686821ea81a7748a3ec8141308c8b8f95b27105713954abf7a6529e700`
-   Qwen3 8B model digest:
    `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
-   Model format and quantization: GGUF, Q4_K_M, 8.2B parameters
-   Master seed: `20260821`
-   Recorded configuration: `production-1000-lock.json`

All seven replicas were verified to use the same Ollama image and model
digest before the production run.

### Results

The lightweight merged results tracked by Git are:

-   `artifacts/production-1000/merged/all_states.csv`
-   `artifacts/production-1000/merged/manifest.json`

The complete 1,000-run JSON dataset is distributed as the GitHub Release
asset `lunar_futures_production1000_results.tar.gz`. Its SHA-256 and the
checksums of every run JSON and merged result are recorded in
`Trial1-1000x.SHA256SUMS`.

### Reproduction

Copy `.env.7gpu.example` to `.env.7gpu`, replace the seven GPU UUID
placeholders, and start the digest-pinned Ollama replicas:

``` bash
sudo docker compose --env-file .env.7gpu \
  -f compose.ollama-7gpu.yaml up -d
```

Ensure that `qwen3:8b` is present in every replica's model volume. Then
run the seven workers and validated merge:

``` bash
NUM_RUNS=1000 \
NUM_TURNS=10 \
MASTER_SEED=20260821 \
MODEL=qwen3:8b \
EXPERIMENT_DIR="$PWD/artifacts/production-1000" \
scripts/run_7_workers.sh
```

Progress can be monitored from another terminal with:

``` bash
scripts/progress.sh artifacts/production-1000
```

After obtaining the Release archive and extracting its result files into
the repository layout, verify the snapshot with:

``` bash
sha256sum -c Trial1-1000x.SHA256SUMS
```

### Summary

Across the 1,000 simulated futures, final geopolitical tension had a mean
of 78.43 and a median of 83; mean trust was 44.60. An exploratory
**composite** cold-war-like criterion was defined as all three of:

-   tension >= 80
-   trust <= 40
-   neutral access <= 10

Under that criterion, 273 of 1,000 worldlines (27.3%) were classified as
cold-war-like outcomes. The 27.3% value is **not a forecast probability**.
It is a simulation outcome specific to this model, agent prompts,
state-transition rules, event probabilities, and production configuration.

USA most frequently expanded surveillance, China most frequently made
territorial claims, and Japan predominantly selected rescue actions. These
results should likewise be interpreted as outputs of this exploratory
simulation configuration, not as predictions of real-world behavior.

## Current status

This repository contains an **early hackathon prototype**.

At this stage, transition rules, agent prompts, event probabilities, and
state variables are intentionally simple and exploratory. Results depend
on both the LLM decisions and hand-defined state-transition rules.

Future versions should therefore include sensitivity analyses, repeated
simulations, alternative prompts/models, better separation between
endogenous agent behavior and exogenous transition rules, and systematic
validation of the simulation design.

## Vision

The ultimate objective is not to ask an LLM to predict the future.

It is to use many interacting AI agents to explore a large landscape of
plausible futures, identify unusual but desirable trajectories, and ask:

> **What had to happen for this future to become possible?**

The lunar south pole provides a concrete testbed, but the same framework
could potentially be applied to other complex systems in which many
actors, institutions, technologies, unexpected events, and feedback
loops interact.

**Generate many futures. Find the rare good ones. Trace them backward.**
