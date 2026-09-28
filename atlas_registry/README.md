# Atlas registry

1. Every mask must already occupy the same atlas coordinate space.
2. COVALE does not perform atlas or image registration.
3. Each registry entry is a spatial primitive available to localization.
4. The registry is not intended to enumerate every anatomical phrase.
5. Complex descriptions should be composed from available primitives.
6. Add a primitive by adding its mask and an entry to `registry.json`.
7. Aliases are convenience metadata and do not define a closed vocabulary.

Mask paths are relative to this directory and must remain inside it.
