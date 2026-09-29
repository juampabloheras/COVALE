# Atlas Registry v2

COVALE Registry v2 separates anatomical ontology metadata from atlas storage.
It supports both one-file-per-mask resources and integer-labeled atlas volumes.
The included example atlas is synthetic and exists only to demonstrate the
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
      "volume": "atlases/anatomical.nii.gz",
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

Registry v1 files remain loadable and are converted to v2 in memory.

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

## Convert a MedPrimitives-style atlas directory

The converter scans `*_canonical_names.json` files and NIfTI volumes, verifies
3D integer labels and affines, preserves ontology metadata and synonyms,
creates stable IDs, copies assets into a self-contained registry bundle, and
writes `migration_report.json`.

```bash
covale-migrate-registry \
  /path/to/source/atlases \
  /path/to/output/atlas_registry \
  --space MNI152 \
  --resolution 1mm \
  --metadata /path/to/atlas_metadata.json
```

The optional metadata file maps source atlas names to provenance:

```json
{
  "AnatomicalAtlas": {
    "name": "Anatomical Atlas",
    "role": "anatomical",
    "priority": 10,
    "source_url": "https://example.org/atlas",
    "version": "1.0",
    "citation": "Author et al.",
    "license": "CC-BY-4.0"
  }
}
```

The report explicitly warns about missing licenses and citations. Review those
warnings before distributing the generated bundle. Use `--overwrite` only
when intentionally replacing an existing migration.
