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
