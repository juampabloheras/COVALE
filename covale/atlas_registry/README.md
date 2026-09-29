# Atlas registry

COVALE supports integer-labeled atlas volumes and one-file-per-region binary
masks. The included atlas is synthetic and exists only to demonstrate the
format; replace it before scientific use.

## Requirements

1. Every mask and labeled volume must already occupy the declared coordinate
   space. COVALE does not perform registration.
2. Asset paths are relative to `registry.json` and must stay inside the
   registry directory.
3. Region IDs and atlas IDs must be stable. Expressions use region IDs rather
   than display names.
4. Record atlas licenses, citations, versions, and source URLs before
   redistributing third-party assets.

## Registry structure

```json
{
  "schema_version": 2,
  "space": {"name": "MNI152", "resolution": "1mm"},
  "atlases": {
    "anatomical": {
      "name": "Anatomical Atlas",
      "volume": "atlases/MNI_Anatomical/volumes/anatomical.nii.gz",
      "role": "anatomical",
      "priority": 10,
      "source_url": "https://example.org/atlas",
      "version": "1.0",
      "citation": "Author et al.",
      "license": "CC-BY-4.0",
      "sha256": "64-lowercase-hex-characters"
    }
  },
  "regions": {
    "anatomical:17": {
      "canonical_name": "hippocampus",
      "display_name": "left hippocampus",
      "source_label": "Left-Hippocampus",
      "laterality": "left",
      "parent_anatomy": "temporal lobe",
      "parent_ids": [],
      "structure_type": "subcortical_nucleus",
      "is_abnormality": false,
      "synonyms": ["left hippocampal formation"],
      "confidence": 0.99,
      "rationale": "Canonicalized from the atlas label table.",
      "source": {
        "type": "labels",
        "atlas": "anatomical",
        "values": [17]
      }
    }
  }
}
```

For a binary mask, replace the label source with:

```json
{"type": "mask", "path": "masks/left-hippocampus.nii.gz"}
```

## Search and resolution

Deterministic search considers stable IDs, display and canonical names, source
labels, and synonyms. It preserves laterality and supports filters for
laterality, structure type, atlas, and parent region. Exact unambiguous matches
do not call OpenAI. Ambiguous or compositional descriptions send only a ranked
candidate catalog to the configured model, and returned IDs must belong to
that catalog.

Expressions use stable IDs:

```json
{
  "op": "union",
  "args": [
    {"op": "region", "id": "anatomical:17"},
    {"op": "region", "id": "anatomical:53"}
  ]
}
```

Legacy `{"op": "region", "name": "left hippocampus"}` expressions remain
supported when the name resolves exactly and unambiguously.

## Onboarding a new atlas

Create one self-contained source directory:

```text
NewAtlas/
├── atlas.json
├── labels.json
├── LICENSE.txt
└── NewAtlas.nii.gz
```

The directory must contain exactly one `.nii` or `.nii.gz` volume.

### `atlas.json`

```json
{
  "id": "new-atlas",
  "name": "New Atlas",
  "role": "anatomical",
  "priority": 100,
  "source_url": "https://example.org/new-atlas",
  "version": "1.0",
  "citation": "Author et al.",
  "license": "CC-BY-4.0"
}
```

The `license` field is the license identifier recorded in the registry.
`LICENSE.txt` contains the corresponding license text.

### `labels.json`

Keys are positive integer voxel values. Each value can be a plain label or an
object containing a label and stable ID:

```json
{
  "1": {
    "id": "new-atlas:left-hippocampus",
    "label": "left hippocampus"
  },
  "2": {
    "id": "new-atlas:right-hippocampus",
    "label": "right hippocampus"
  }
}
```

Stable IDs must begin with the atlas ID followed by `:`. If `id` is omitted,
COVALE generates `<atlas-id>:<voxel-value>`. Existing canonical fields such as
`canonical_name`, `synonyms`, `laterality`, `parent_anatomy`, `structure_type`,
`confidence`, and `rationale` are also accepted.

### Run onboarding

The repository includes a complete synthetic example under
[`examples/atlas_onboarding`](../examples/atlas_onboarding):

```bash
covale-onboard-atlas examples/atlas_onboarding
```

This single command:

1. Validates `atlas.json`, `labels.json`, `LICENSE.txt`, and the NIfTI volume.
2. Requires exact agreement between JSON keys and nonzero voxel labels.
3. Canonicalizes every label.
4. Copies the source files and canonical ontology under
   `covale/atlas_registry/atlases/<directory-name>/`.
5. Adds the atlas and stable region IDs to
   `covale/atlas_registry/registry.json`.

| Mode | Use when | Behavior |
|---|---|---|
| `deterministic` | Labels are simple names | Normalizes separators and case, extracts laterality, infers common structure types, and creates synonyms locally |
| `none` | Labels already contain curated canonical fields | Preserves names, synonyms, laterality, structure types, confidence, and rationale |
| `openai` | Labels need richer anatomical interpretation | Uses the consumer OpenAI API in validated chunks; requires `OPENAI_API_KEY` |

Select the mode with:

```bash
covale-onboard-atlas NewAtlas --canonicalization none
```

Use `--model` and `--chunk-size` with `openai`. Use `--registry` when the
registry is not at `covale/atlas_registry/registry.json`. Onboarding refuses to
replace existing files, atlas IDs, or stable region IDs unless `--overwrite`
is supplied.

The registry and output assets must share a directory tree so all volume paths
remain contained within the registry directory.
