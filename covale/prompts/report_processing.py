EXTRACTION_PROMPT = """
Extract every clinical finding with an anatomical location from the report.

Return only a JSON object with this shape:
{
  "findings": [
    {
      "text": "the source text describing the finding",
      "concept": "a concise normalized clinical finding",
      "anatomy": "the complete anatomical location phrase",
      "assertion": "present, absent, or uncertain"
    }
  ]
}

Use an empty findings list when the report contains no anatomically localized
findings. Keep laterality and spatial modifiers in anatomy. Do not infer
findings or locations that are not stated in the report.
""".strip()

COMPATIBILITY_PROMPT = """
Identify every clinically compatible reference and candidate finding pair.

Two findings are compatible only when they describe the same underlying
clinical observation and have compatible assertion status. Allow clinical
paraphrases, but do not match different abnormalities merely because they
occur in the same anatomical location. Return all compatible combinations;
spatial alignment will choose the final one-to-one matches.

Return only a JSON object with this shape:
{
  "compatible_pairs": [
    {
      "reference_index": 0,
      "candidate_index": 1
    }
  ]
}
""".strip()
