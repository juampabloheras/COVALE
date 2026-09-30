from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from covale.registry import (
    AtlasMask,
    DEFAULT_REGISTRY,
    Registry,
    RegistryError,
    difference,
    intersection,
    union,
)
from covale.registry.models import validate_expression
from covale.localize.spatial_primitives import (
    clip_plane,
    directional_part,
    morphology,
)

Expression = Mapping[str, Any]
ExpressionBuilder = Callable[[str, Mapping[str, Any]], Expression]


class LocalizationError(ValueError):
    pass


def load_registry(registry_path: str | Path) -> dict[str, Any]:
    try:
        return Registry.load(registry_path).model.model_dump(mode="json")
    except RegistryError as error:
        raise LocalizationError(str(error)) from error


def execute_expression(
    expression: Expression,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> AtlasMask:
    try:
        registry = Registry.load(Path(registry_path))
    except RegistryError as error:
        raise LocalizationError(str(error)) from error
    return execute_registry_expression(expression, registry)


def execute_registry_expression(
    expression: Expression,
    registry: Registry,
) -> AtlasMask:
    try:
        expression = validate_expression(expression).model_dump(exclude_none=True)
    except ValidationError as error:
        raise LocalizationError(f"Invalid mask expression: {error}") from error

    def resolve_region(node: Mapping[str, Any]) -> AtlasMask:
        region_id = node.get("id")
        if region_id is None:
            name = node.get("name")
            if not isinstance(name, str):
                raise LocalizationError("Region expression requires an id or name.")
            try:
                region_id = registry.resolve_name(name)
            except RegistryError as error:
                raise LocalizationError(str(error)) from error
            if region_id is None:
                raise LocalizationError(f"Unknown atlas region: {name}")
        try:
            return registry.mask(region_id)
        except ValueError as error:
            raise LocalizationError(str(error)) from error

    def interpret(node: object) -> AtlasMask:
        if not isinstance(node, dict) or not isinstance(node.get("op"), str):
            raise LocalizationError("Mask expression must contain a string op.")

        operator = node["op"]
        if operator == "unresolved":
            if set(node) != {"op"}:
                raise LocalizationError("Malformed unresolved expression.")
            raise LocalizationError("Anatomical description could not be localized.")

        if operator == "region":
            return resolve_region(node)

        if operator in {
            "directional_part",
            "clip_plane",
            "dilate",
            "erode",
        }:
            mask = interpret(node["arg"])
            try:
                if operator == "directional_part":
                    return directional_part(
                        mask,
                        node["direction"],
                        node["fraction"],
                    )
                if operator == "clip_plane":
                    return clip_plane(
                        mask,
                        tuple(node["normal"]),
                        node["offset_mm"],
                        node["side"],
                    )
                return morphology(mask, operator, node["distance_mm"])
            except ValueError as error:
                raise LocalizationError(str(error)) from error

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
        argument = node.get("arg")
        if isinstance(argument, Mapping):
            regions, operations, child_depth = count(argument, depth + 1)
            return regions, operations + 1, max(depth, child_depth)
        arguments = node.get("args")
        if not isinstance(arguments, list):
            return 0, 0, depth

        regions = 0
        operations = 1
        max_depth = depth
        for argument in arguments:
            child_regions, child_operations, child_depth = count(argument, depth + 1)
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

    try:
        registry = Registry.load(registry_path)
    except RegistryError as error:
        raise LocalizationError(str(error)) from error
    if expression_builder is None:
        try:
            region_id = registry.resolve_name(text)
        except RegistryError as error:
            raise LocalizationError(str(error)) from error
        expression: Expression = (
            {"op": "region", "id": region_id}
            if region_id is not None
            else {"op": "unresolved"}
        )
    else:
        expression = expression_builder(
            text,
            registry.model.model_dump(mode="json"),
        )

    if not isinstance(expression, Mapping):
        raise LocalizationError("Localization must return a mask expression.")
    try:
        expression = validate_expression(expression).model_dump(exclude_none=True)
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
