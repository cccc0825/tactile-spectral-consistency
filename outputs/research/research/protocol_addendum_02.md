# Prospective implementation audit repairs

2026-09-12; still before any classifier fitting or target inference.

The independent reviewer found potential stale-cache, incomplete freeze-lineage and timeout gaps. Source-selection cache reuse now requires matching hashes of code, preprocessing, protocol, PSDs, quality and position audit. Freeze includes those files. The90minute timer is persistent wall time starting at the first source selection, including any intervening pause; checks occur before/after fits and target prediction, including perturbations. Source selection for all planned models precedes target inference; if this entire stage cannot finish, target results remain unexecuted. Once frozen, core primary/replicate models precede distance controls; all incomplete controls will be listed. This clarifies the earlier optimistic priority wording.

A gap crossing a selected-window boundary now fails the same20ms rule as an internal gap. Extracted parts are versioned by extractor hash and will all be regenerated with the repaired check before formal selection.

Synthetic speed-metadata errors change the spatial feature's temporal support:±10% extends it to18–1320Hz, which lies inside the observed0–3000Hz PSD but outside the primary20–1200Hz support. The sensitivity analysis intentionally evaluates the frozen transform at these erroneous inputs and does not extrapolate beyond observed PSD support. It is not a matched-bandwidth causal experiment. No target accuracy has informed this specification.
