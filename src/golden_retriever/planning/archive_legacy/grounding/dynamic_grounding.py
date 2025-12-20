from __future__ import annotations

from retriever.core.symbolic_structs import Variable
from retriever.core.types import (
    Module,
    ObjectDescriptionDict,
    ObjectSymbol,
)


class DynamicGroundingModule(
    Module[
        tuple[list[Variable], ObjectDescriptionDict],
        list[ObjectSymbol],
    ]
):
    """
    A module that grounds symbolic object variables to concrete object IDs.
    """

    def __call__(
        self, inp: tuple[list[Variable], ObjectDescriptionDict]
    ) -> list[ObjectSymbol]:
        """
        Takes a list of symbolic variables and a dictionary of object
        descriptions, and returns a list of grounded object symbols.

        This implementation performs a simple greedy, case-insensitive search.
        For each variable, it finds the first available perceived object whose
        description contains the variable's type name as a substring.
        """
        object_vars, object_descriptions = inp
        available_object_ids = set(object_descriptions.descriptions.keys())
        groundings: list[ObjectSymbol] = []

        for var in object_vars:
            # Find the first available object that matches the variable type.
            # E.g., var.type.name="cup" matches description="a red cup".
            match_found = False
            # Iterate over a copy of the items to allow modification
            sorted_object_ids = sorted(list(available_object_ids))
            for obj_id in sorted_object_ids:
                description = object_descriptions.descriptions[obj_id]
                if var.type.name.lower() in description.lower():
                    groundings.append(ObjectSymbol(object_id=obj_id))
                    available_object_ids.remove(obj_id)
                    match_found = True
                    break
        
        # Note: If no match is found for a variable, it is simply omitted
        # from the output. A more robust implementation might raise an error.
        return groundings 