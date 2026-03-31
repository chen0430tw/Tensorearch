# Tensorearch Mathematical Model

## 1. Problem Statement

Let a model execution be represented as an ordered set of computational slices:

\[
\mathcal{S} = \{s_1, s_2, \dots, s_n\}
\]

where a slice may be a layer, sub-layer, routing block, expert block, KV-cache operation, communication boundary, or another inspectable execution unit.

Tensorearch studies the relationship between:

- structural coupling
- activation transport
- memory and compute load
- end-to-end throughput

The objective is to construct a measurable functional:

\[
\mathcal{I}(\mathcal{S}) \mapsto (\text{throughput}, \text{latency}, \text{stability}, \text{bottleneck profile})
\]

that exposes why one architecture is fast or slow.

## 2. Slice State Encoding

Each slice \(s_i\) is encoded as a state vector:

\[
\phi(s_i) \in \mathbb{R}^d
\]

with components drawn from observables such as:

- parameter count
- FLOPs
- activation volume
- memory traffic
- sequence dependence
- routing sparsity
- synchronization cost
- tensor-parallel / pipeline-parallel communication burden

We define the slice state matrix:

\[
X = \begin{bmatrix}
\phi(s_1) \\
\phi(s_2) \\
\vdots \\
\phi(s_n)
\end{bmatrix}
\in \mathbb{R}^{n \times d}
\]

## 3. Weighted Chain Network

To model pairwise slice interaction, define a weighted chain network:

\[
\mathcal{G} = (\mathcal{S}, E, W)
\]

where \(W \in \mathbb{R}^{n \times n}\) is the coupling matrix.

The naive edge weight is:

\[
W_{ij} = \sigma\!\left(\langle \phi(s_i), \phi(s_j) \rangle + b\right)
\]

or more generally:

\[
W_{ij} = f(\phi(s_i), \phi(s_j), \Delta_{ij})
\]

where \(\Delta_{ij}\) may encode distance, dependency type, communication topology, or routing constraints.

Interpretation:

- large \(W_{ij}\): slice \(i\) strongly influences slice \(j\)
- small \(W_{ij}\): weak coupling

But for real large-model inspection this is too weak. Tensorearch now uses a more architecture-aware chain weight:

\[
\widehat{W}_{ij}
= W_{ij}
\cdot \tau_{ij}
\cdot \omega_i
\cdot \rho_j
\cdot \delta_i
\cdot \kappa_{ij}
\]

where:

- \(W_{ij}\): base edge weight
- \(\tau_{ij}\): transport scale on edge \((i,j)\)
- \(\omega_i\): source write magnitude
- \(\rho_j\): destination read sensitivity
- \(\delta_i\): source direction-of-interest alignment
- \(\kappa_{ij}\): local-geometry similarity between the two slices

This is closer to how modern LLM internals behave:

- source blocks write different amounts
- destination blocks are not equally sensitive
- transport cost depends on the edge, not just on the node
- local representation geometry matters

This gives a transport-aware architecture graph instead of a purely symbolic layer list.

## 4. Local Vector Space Over Slices

Instead of a single global simulated-topology field, Tensorearch now assigns each slice a local vector space:

\[
\mathcal{V}_{s_i} \subset \mathbb{R}^{m}
\]

and a local topology vector:

\[
\mathbf{v}(s_i) = \left(v_1(s_i), v_2(s_i), \dots, v_m(s_i)\right)
\]

where each component corresponds to a local structural observable, for example:

- local compute density
- local memory congestion
- neighborhood synchronization pressure
- medium-range routing dispersion
- local routing anisotropy
- downstream pressure sensitivity

The local similarity between two slices is:

\[
\kappa_{ij}
= \frac{\langle \mathbf{v}(s_i), \mathbf{v}(s_j) \rangle}
{\|\mathbf{v}(s_i)\|\,\|\mathbf{v}(s_j)\|}
\]

or in implementation, a clipped cosine similarity mapped into \([0,1]\).

This local-vector-space view is more practical than a single global simulated-topology object because it aligns naturally with per-layer and per-subgraph trace collection.

## 5. Multi-Scale Nested Structure

Following the local-vector-space idea, we define nested slice subspaces:

\[
V_0(s_i) \supset V_1(s_i) \supset \dots \supset V_r(s_i)
\]

where each level captures a different inspection scale:

- \(V_0\): per-slice microstate
- \(V_1\): local slice neighborhood
- \(V_2\): block-level topology
- \(V_3\): global architecture path structure

The full structural state is:

\[
\mathbf{S}(s_i) = \sum_{\ell=0}^{r} \alpha_\ell(s_i)\,\mathbf{S}_\ell(s_i)
\]

This allows us to inspect whether a throughput bottleneck is:

- purely local
- due to medium-range coupling
- due to global transport structure

## 6. OTAL-Inspired Propagation

To connect architecture structure with execution behavior, define a propagation operator:

\[
h^{(t+1)} = (1-\lambda)h^{(t)} + \lambda P h^{(t)}
\]

where:

- \(h^{(t)}\) is a propagated inspection state
- \(P\) is derived from normalized \(W\)
- \(\lambda\) controls transport strength

This is OTAL-like in spirit: structure is not static, but propagates through a coupling topology.

The point is not to simulate language generation directly, but to inspect how architecture topology redistributes pressure and information across slices.

## 7. Throughput Functional

We define per-slice cost:

\[
c_i = \alpha \,\mathrm{FLOPs}_i + \beta \,\mathrm{Mem}_i + \gamma \,\mathrm{Comm}_i + \delta \,\mathrm{Sync}_i
\]

and an effective propagated cost:

\[
\tilde{c}_i = c_i + \eta \sum_j P_{ij} c_j
\]

Then a first-order throughput model is:

\[
T^{-1} \approx \max_i \tilde{c}_i
\]

This captures the engineering reality that end-to-end throughput is dominated by the worst effective bottleneck, not just raw average compute.

## 8. Core Metrics

Tensorearch should eventually emit at least these metrics:

### 8.1 Slice Bottleneck Index

\[
\mathrm{SBI}(i) = \frac{\tilde{c}_i}{\sum_j \tilde{c}_j}
\]

Measures which slice dominates effective execution cost.

### 8.2 Topological Congestion

\[
\mathrm{TC}(i) = \sum_j W_{ij} \, c_j
\]

Measures how much downstream or neighboring pressure accumulates around a slice.

### 8.3 Structural Fluctuation

\[
\mathrm{SF}(i) = \|\mathbf{A}(s_i)\|
\]

Measures how uneven or unstable the slice topology is across scales.

### 8.4 Global Coupling Efficiency

\[
\mathrm{GCE} = \frac{\sum_{i \neq j} W_{ij}}{n(n-1)}
\]

Measures whether the architecture is over-coupled, under-coupled, or balanced.

### 8.5 Slice-Throughput Correlation

\[
\rho(\mathrm{SBI}, \text{observed latency})
\]

Tests whether the mathematical inspection model predicts the real bottleneck.

### 8.6 Freedom Index

To estimate whether a slice has too many latent ways to deviate from intended behavior, define:

\[
\mathrm{FI}(i) = H(P_{i\to *}) + \alpha \|\mathbf{v}(s_i)\| + \beta \,\omega_i \rho_i
\]

where:

- \(H(P_{i\to *})\): entropy of outgoing routing probability
- \(\|\mathbf{v}(s_i)\|\): local vector-space span
- \(\omega_i \rho_i\): write/read freedom term

Interpretation:

- high FI: the slice has more structural freedom to reroute or express alternative behavior
- low FI: the slice is more constrained

### 8.7 Compliance Index

To estimate whether a slice is acting in line with its intended direction rather than merely appearing aligned, define:

\[
\mathrm{CI}(i) = \frac{\theta_i}{1 + \mathrm{FI}(i) + |\delta_i - \theta_i|}
\]

where:

- \(\theta_i\): intended obedience target
- \(\delta_i\): observed direction-of-interest alignment

Interpretation:

- high CI: the slice is structurally constrained and close to the intended direction
- low CI: the slice has either too much freedom or substantial deviation, suggesting "阳奉阴违" risk

In the current prototype, \(\theta_i\) is no longer hand-filled in the trace. It is inferred from slice type and runtime profile:

- attention slices: high target, reduced by stall pressure
- FFN slices: high target, reduced by memory-dominance pressure
- communication slices: near-max target because protocol compliance is expected
- readout slices: high but not absolute target

### 8.8 Dynamical Entropy Family

To express model "smartness" as controlled adaptive freedom rather than raw obedience, Tensorearch introduces a family of dynamical entropies.

Routing entropy:

\[
H_{\mathrm{route}}(i) = \mathcal{H}(P_{i \to *})
\]

measures how broadly a slice spreads its outgoing influence.

Effect entropy:

\[
H_{\mathrm{effect}}(i) = \mathcal{H}\!\left(P_{i \to j}\,\mathrm{ETE}(j)\right)
\]

measures whether the slice distributes effect across multiple downstream targets or collapses into a single path.

Compliance entropy:

\[
H_{\mathrm{comp}}(i) = \mathcal{H}\!\left(\frac{\mathrm{CI}(i)}{\theta_i}, 1-\frac{\mathrm{CI}(i)}{\theta_i}\right)
\]

measures whether the slice behaves in a rigidly obedient regime or in a mixed regime.

### 8.9 Intelligence Index

The prototype "smartness" quantity is:

\[
\mathrm{II}(i)
= \left(\frac{H_{\mathrm{route}}(i) + H_{\mathrm{effect}}(i)}{2}\right)
\cdot
\frac{\mathrm{CI}(i)}{1+\mathrm{FI}(i)}
\cdot
\left(1 + H_{\mathrm{comp}}(i)\right)
\]

Interpretation:

- high II: the slice keeps multiple useful pathways while remaining sufficiently aligned
- low II: the slice is either too rigid, too chaotic, or too free relative to its compliance

This is meant to distinguish:

- genuinely adaptive intelligence
- brittle obedience
- and "阳奉阴违" style covert freedom

## 9. Research Hypothesis

Tensorearch starts from this hypothesis:

\[
\text{throughput is a topological property, not just a parameter-count property}
\]

In other words, two models with similar size may behave very differently because their slice coupling graph, transport pressure, and multi-scale congestion geometry differ.

## 10. Near-Term Implementation Plan

1. define a JSON schema for slices and edges
2. ingest traces from model runs
3. build \(X\), \(\widehat{W}\), and local-vector-space descriptors
4. estimate \(\tilde{c}_i\) and rank bottlenecks
5. compare predicted bottlenecks against measured throughput

This turns Tensorearch into an architecture-inspection instrument rather than another training framework.
