# Transformer Debugger to Tensorearch Mapping

## Why This Mapping Exists

OpenAI's Transformer Debugger inspects how internal model components contribute to a behavioral target.

Tensorearch reuses that workflow, but changes the inspection object:

- from internal representational units
- to architecture execution units

## OpenAI Transformer Debugger: Core Pattern

Transformer Debugger uses a pattern like:

1. define a target behavior
2. define a direction of interest
3. collect component activations
4. estimate contribution via attribution / intervention
5. trace circuits that causally matter

The public repository and README indicate these major objects:

- subject model
- activation server
- neuron viewer
- neurons / attention heads / SAE latents
- direct effect
- estimated total effect
- ablation / intervention

## Tensorearch: Corresponding Objects

| Transformer Debugger | Tensorearch |
|---|---|
| subject model | subject architecture |
| neuron / head / SAE latent | slice / route / transport edge / topology cell |
| behavior target | throughput target / latency target / bottleneck target |
| direction of interest | throughput-gradient direction / bottleneck direction |
| direct effect | direct slice cost contribution |
| estimated total effect | propagated topology-aware bottleneck effect |
| ablation | slice removal / edge masking / routing suppression |
| circuit tracing | bottleneck-path tracing |

## Direction of Interest

In Transformer Debugger, a direction of interest can be formed from the difference between output directions tied to competing tokens.

In Tensorearch, the analogous target is not token choice but execution quality.

Define an inspection objective:

\[
\mathcal{L}_{\text{sys}} = \alpha \cdot \text{latency} - \beta \cdot \text{throughput} + \gamma \cdot \text{stall} + \delta \cdot \text{memory pressure}
\]

Then the direction of interest becomes the gradient of system quality with respect to slice state:

\[
d_i = \nabla_{\phi(s_i)} \mathcal{L}_{\text{sys}}
\]

This gives a per-slice architecture-debug direction.

## Direct Effect

Transformer Debugger uses direct effect to ask how strongly a component writes into a relevant downstream direction.

Tensorearch analog:

\[
\mathrm{DE}(s_i) = \langle \phi(s_i), d_i \rangle
\]

This estimates the direct contribution of slice \(s_i\) to the throughput objective.

## Estimated Total Effect

Transformer Debugger also uses a broader effect estimate that includes mediated downstream influence.

Tensorearch analog:

\[
\mathrm{ETE}(s_i) = \mathrm{DE}(s_i) + \eta \sum_j P_{ij}\mathrm{DE}(s_j)
\]

where \(P\) is a normalized propagation matrix derived from weighted-chain topology.

This is the architecture-level version of circuit-mediated influence.

## Attribution on Edges

Transformer Debugger often uses activation-times-gradient style objects.

For architecture inspection, a similar edge quantity is:

\[
\mathrm{EA}_{ij} = W_{ij}\frac{\partial \mathcal{L}_{\text{sys}}}{\partial W_{ij}}
\]

This measures how much the coupling edge between slices \(i\) and \(j\) contributes to poor or good throughput.

## Intervention

Transformer Debugger uses interventions in the forward pass.

Tensorearch interventions should include:

- remove a slice
- mask an edge
- replace learned routing with fixed local routing
- reduce communication bandwidth
- freeze a subgraph

Then compare:

\[
\Delta \mathcal{L}_{\text{sys}} = \mathcal{L}_{\text{sys}}^{\text{intervened}} - \mathcal{L}_{\text{sys}}^{\text{base}}
\]

This turns bottleneck analysis into a causal experiment.

## Why OTAL Fits

Transformer Debugger traces causal structure through representational pathways.

Tensorearch traces causal structure through topology pathways.

OTAL is relevant because it treats the system as:

- a coupled topology
- with propagated influence
- whose useful behavior depends on graph structure rather than only local state

That matches the architecture-inspection problem directly.

## Practical Outcome

The result is a tool that can answer questions like:

- why is this architecture slower than another one of similar size?
- which slice is the dominant bottleneck?
- which coupling pattern is amplifying stalls?
- does a learned routing topology actually help throughput?
- where should we simplify, fuse, or rewire the graph?
