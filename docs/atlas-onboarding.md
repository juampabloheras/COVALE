# Onboarding a new atlas

Put the atlas metadata, labels, license, and one labeled NIfTI volume in a
single directory:

```text
NewAtlas/
├── atlas.json
├── labels.json
├── LICENSE.txt
└── NewAtlas.nii.gz
```

`atlas.json` contains the atlas ID, name, license identifier, citation, source
URL, version, role, and priority. `labels.json` maps integer voxel values to
labels and may provide stable region IDs. See the runnable example in
[`examples/atlas_onboarding`](../examples/atlas_onboarding).

Run one command from the repository root:

```bash
covale-onboard-atlas examples/atlas_onboarding
```

The command validates the files and voxel labels, canonicalizes the ontology,
copies the bundle under `covale/atlas_registry/atlases/`, and updates
`covale/atlas_registry/registry.json`.

| Mode | Use when | Behavior |
|---|---|---|
| `deterministic` | Labels are simple names | Normalizes names, extracts laterality, infers common structure types, and creates synonyms locally |
| `none` | Labels already contain canonical metadata | Preserves canonical names, synonyms, laterality, confidence, and rationale |
| `openai` | Labels need richer interpretation | Uses the consumer OpenAI API in validated chunks |

Select a mode with `--canonicalization MODE`. See the
[atlas registry documentation](../covale/atlas_registry/README.md) for the
complete file formats and validation rules.

[Back to the main README](../README.md#additional-resources)
