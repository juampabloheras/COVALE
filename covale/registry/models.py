from typing import Annotated, Literal

from pydantic import Field, model_validator

from covale.models import StrictModel

Laterality = Literal[
    "left",
    "right",
    "bilateral",
    "midline",
    "unknown",
]


class AtlasSpace(StrictModel):
    name: str = Field(min_length=1)
    resolution: str | None = None


class AtlasDefinition(StrictModel):
    name: str = Field(min_length=1)
    volume: str = Field(min_length=1)
    role: str = Field(default="anatomical", min_length=1)
    priority: int = Field(default=100, ge=0)
    source_url: str | None = None
    version: str | None = None
    citation: str | None = None
    license: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class MaskSource(StrictModel):
    type: Literal["mask"]
    path: str = Field(min_length=1)


class LabelSource(StrictModel):
    type: Literal["labels"]
    atlas: str = Field(min_length=1)
    values: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def require_positive_unique_labels(self) -> "LabelSource":
        if any(value <= 0 for value in self.values):
            raise ValueError("Atlas label values must be positive integers.")
        if len(set(self.values)) != len(self.values):
            raise ValueError("Atlas label values must be unique.")
        return self


RegionSource = Annotated[
    MaskSource | LabelSource,
    Field(discriminator="type"),
]


class RegionDefinition(StrictModel):
    canonical_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    source_label: str | None = None
    laterality: Laterality = "unknown"
    parent_anatomy: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    structure_type: str = Field(default="unknown", min_length=1)
    is_abnormality: bool = False
    synonyms: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None
    source: RegionSource

    @model_validator(mode="after")
    def normalize_terms(self) -> "RegionDefinition":
        values = []
        seen = set()
        for synonym in self.synonyms:
            synonym = synonym.strip()
            folded = synonym.casefold()
            if synonym and folded not in seen:
                values.append(synonym)
                seen.add(folded)
        self.synonyms = values
        return self


class RegistryV2(StrictModel):
    schema_version: Literal[2]
    space: AtlasSpace
    atlases: dict[str, AtlasDefinition] = Field(default_factory=dict)
    regions: dict[str, RegionDefinition]

    @model_validator(mode="after")
    def validate_references(self) -> "RegistryV2":
        if not self.regions:
            raise ValueError("Registry must define at least one region.")
        for region_id, region in self.regions.items():
            if not region_id.strip():
                raise ValueError("Region IDs must not be empty.")
            if isinstance(region.source, LabelSource):
                if region.source.atlas not in self.atlases:
                    raise ValueError(
                        f"Region '{region_id}' references unknown atlas "
                        f"'{region.source.atlas}'."
                    )
            missing_parents = [
                parent_id
                for parent_id in region.parent_ids
                if parent_id not in self.regions
            ]
            if missing_parents:
                raise ValueError(
                    f"Region '{region_id}' references unknown parents: "
                    f"{missing_parents}"
                )
        return self


class RegionMatch(StrictModel):
    region_id: str
    display_name: str
    matched_term: str
    match_type: Literal[
        "id",
        "display_name",
        "canonical_name",
        "source_label",
        "synonym",
        "approximate",
    ]
    score: float = Field(ge=0, le=1)
    laterality: Laterality
    structure_type: str
