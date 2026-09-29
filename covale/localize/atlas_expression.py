from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from covale.masks import AtlasMask, difference, intersection, load_mask, union
from covale.models import AtlasRegistry, validate_expression
from covale.prompts import LOCALIZATION_PROMPT

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2] / "atlas_registry" / "registry.json"
)
Expression = Mapping[str, Any]
ExpressionBuilder = Callable[[str, Mapping[str, Any]], Expression]


class LocalizationError(ValueError):
    pass


def load_registry(registry_path: str | Path) -> dict[str, Any]:
    path = Path(registry_path)
    try:
        registry = AtlasRegistry.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise LocalizationError(f"Could not read atlas registry: {path}") from error

    return registry.model_dump()


def execute_expression(
    expression: Expression,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> AtlasMask:
    path = Path(registry_path)
    registry = load_registry(path)
    try:
        expression = validate_expression(expression).model_dump()
    except ValidationError as error:
        raise LocalizationError(f"Invalid mask expression: {error}") from error
    regions = registry["regions"]
    registry_root = path.resolve().parent

    def resolve_region(name: str) -> AtlasMask:
        canonical_name = name
        if canonical_name not in regions:
            requested_name = name.casefold()
            canonical_name = next(
                (
                    region_name
                    for region_name, entry in regions.items()
                    if isinstance(entry, dict)
                    and requested_name
                    in {
                        region_name.casefold(),
                        *(
                            alias.casefold()
                            for alias in entry.get("aliases", [])
                            if isinstance(alias, str)
                        ),
                    }
                ),
                "",
            )

        if not canonical_name:
            raise LocalizationError(f"Unknown atlas region: {name}")

        entry = regions[canonical_name]
        if not isinstance(entry, dict):
            raise LocalizationError(f"Malformed atlas region: {canonical_name}")

        relative_path = entry.get("mask")
        if not isinstance(relative_path, str) or not relative_path:
            raise LocalizationError("Atlas region must define a mask path.")

        mask_path = (registry_root / relative_path).resolve()
        if not mask_path.is_relative_to(registry_root):
            raise LocalizationError(
                "Atlas mask path must remain inside atlas_registry."
            )
        if not mask_path.is_file():
            raise LocalizationError(f"Atlas mask does not exist: {relative_path}")
        return load_mask(mask_path)

    def interpret(node: object) -> AtlasMask:
        if not isinstance(node, dict) or not isinstance(node.get("op"), str):
            raise LocalizationError("Mask expression must contain a string op.")

        operator = node["op"]
        if operator == "unresolved":
            if set(node) != {"op"}:
                raise LocalizationError("Malformed unresolved expression.")
            raise LocalizationError("Anatomical description could not be localized.")

        if operator == "region":
            if set(node) != {"op", "name"} or not isinstance(
                node.get("name"), str
            ):
                raise LocalizationError("Region expression requires a string name.")
            return resolve_region(node["name"])

        if operator not in {"union", "intersection", "difference"}:
            raise LocalizationError(f"Unknown mask operator: {operator}")
        if set(node) != {"op", "args"} or not isinstance(node.get("args"), list):
            raise LocalizationError(f"{operator} expression requires an args list.")
        if len(node["args"]) < 2:
            raise LocalizationError(f"{operator} requires at least two arguments.")

        masks = [interpret(argument) for argument in node["args"]]
        if operator == "union":
            return union(*masks)
        if operator == "intersection":
            return intersection(*masks)
        return difference(masks[0], *masks[1:])

    return interpret(expression)


def expression_stats(expression: Expression) -> dict[str, int]:
    def count(node: object, depth: int) -> tuple[int, int, int]:
        if not isinstance(node, Mapping):
            return 0, 0, depth
        operator = node.get("op")
        if operator == "region":
            return 1, 0, depth
        arguments = node.get("args")
        if not isinstance(arguments, list):
            return 0, 0, depth

        regions = 0
        operations = 1
        max_depth = depth
        for argument in arguments:
            child_regions, child_operations, child_depth = count(
                argument, depth + 1
            )
            regions += child_regions
            operations += child_operations
            max_depth = max(max_depth, child_depth)
        return regions, operations, max_depth

    regions, operations, depth = count(expression, 1)
    return {
        "regions": regions,
        "operations": operations,
        "depth": depth,
    }


def localize_with_expression(
    text: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    expression_builder: ExpressionBuilder | None = None,
) -> tuple[Expression, AtlasMask]:
    if not isinstance(text, str) or not text.strip():
        raise LocalizationError("Anatomical description must be non-empty text.")

    registry = load_registry(registry_path)
    if expression_builder is None:
        requested_name = text.strip().casefold()
        expression: Expression = {"op": "unresolved"}
        for name, entry in registry["regions"].items():
            aliases = entry.get("aliases", []) if isinstance(entry, dict) else []
            available_names = [
                name,
                *(alias for alias in aliases if isinstance(alias, str)),
            ]
            if requested_name in {
                available_name.casefold() for available_name in available_names
            }:
                expression = {"op": "region", "name": name}
                break
    else:
        expression = expression_builder(text, registry)

    if not isinstance(expression, Mapping):
        raise LocalizationError("Localization must return a mask expression.")
    try:
        expression = validate_expression(expression).model_dump()
    except ValidationError as error:
        raise LocalizationError(
            f"Localization returned an invalid expression: {error}"
        ) from error
    return expression, execute_expression(expression, registry_path)


def localize(
    text: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    expression_builder: ExpressionBuilder | None = None,
) -> AtlasMask:
    return localize_with_expression(text, registry_path, expression_builder)[1]
