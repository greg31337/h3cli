# VPIPE ANE graph emitter

Pinned source: https://github.com/tgo-app-dev/vpipe/tree/f34e2cc3a3adae759eea254419f436f5b7800057/apple-silicon/coreml

Apache-2.0, with original LICENSE and NOTICE retained. Only the self-contained
graph emitter is adapted. The VPIPE CoreML manager dependency and self-test are
removed; h3cli owns loading, validation, policy, cache lifetime and scheduling.
The emitted compiled CoreML format is undocumented. This research dependency
is isolated here and requires numerical validation before model execution.
