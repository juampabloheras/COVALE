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

Use `covale-onboard-atlas` to turn a raw integer-labeled NIfTI volume and label
JSON into the canonical atlas layout:

```text
atlas_registry/atlases/<atlas-name>/
├── <volume-stem>_canonical_names.json
├── <volume-stem>_onboarding_report.json
├── raw_labels/
│   └── <original-label-file>.json
└── volumes/
    └── <original-volume>.nii.gz
```

For a label file that already contains canonical metadata, such as the
NextBrain dictionary format, preserve it with `none`:

```bash
covale-onboard-atlas \
  MNI_NextBrain \
  /path/to/NextBrain_left_right_merged.nii.gz \
  atlas_registry/atlases \
  --labels-json /path/to/NextBrain_left_right_merged_labels.json \
  --canonicalization none
```

For a simple `{label_id: label_name}` mapping, use deterministic
canonicalization:

```bash
covale-onboard-atlas \
  MNI_NewAtlas \
  /path/to/NewAtlas.nii.gz \
  atlas_registry/atlases \
  --labels-json /path/to/NewAtlas_labels.json \
  --canonicalization deterministic
```

Available canonicalization modes are:

- `none`: retain existing canonical names, synonyms, laterality, confidence,
  and rationale.
- `deterministic` (default): normalize separators and case, extract laterality,
  infer common structure types, and generate synonyms locally.
- `openai`: canonicalize labels in validated chunks with the consumer OpenAI
  API. Configure `OPENAI_API_KEY`; optionally pass `--model` and
  `--chunk-size`.

If `--labels-json` is omitted, COVALE generates a one-region label file only
for a binary volume whose sole nonzero value is `1`. Multi-label atlases
require a label JSON. Onboarding fails explicitly when JSON IDs and nonzero
volume labels differ, inputs are malformed, or outputs already exist. Pass
`--overwrite` only when replacement is intentional.

After onboarding:

1. Add the copied volume to the `atlases` object in `registry.json`.
2. Add each canonical region with a stable ID and a `labels` source pointing
   to the atlas ID and integer label value.
3. Copy license, citation, source URL, version, and priority metadata into the
   atlas definition.
4. Load the registry and test representative exact, synonym, and lateralized
   queries before using it for evaluation.
