from retriever.core.symbolic_structs import Type, Variable
from retriever.core.types import (
    ObjectDescriptionDict,
)
from retriever.planning.grounding.dynamic_grounding import DynamicGroundingModule


def test_dynamic_grounding_module():
    """
    Tests the DynamicGroundingModule with various scenarios.
    """
    # 1. Set up the module and mock perception data.
    module = DynamicGroundingModule()
    object_descriptions = ObjectDescriptionDict(
        descriptions={
            "red_cup_0": "red cup",
            "blue_mug_1": "blue mug",
            "red_cup_2": "a second red cup",
            "green_plate_3": "a green plate",
        }
    )
    
    cup_type = Type("cup")
    mug_type = Type("mug")
    bowl_type = Type("bowl")

    # 2. Test Case: Simple one-to-one grounding.
    variables_1 = [Variable("?c", cup_type), Variable("?m", mug_type)]
    groundings_1 = module((variables_1, object_descriptions))
    expected_ids_1 = ["blue_mug_1", "red_cup_0"]
    assert sorted([g.object_id for g in groundings_1]) == expected_ids_1

    # 3. Test Case: Grounding multiple objects of the same type.
    variables_2 = [Variable("?c1", cup_type), Variable("?c2", cup_type)]
    groundings_2 = module((variables_2, object_descriptions))
    expected_ids_2 = ["red_cup_0", "red_cup_2"]
    assert sorted([g.object_id for g in groundings_2]) == expected_ids_2

    # 4. Test Case: Ungroundable variable.
    variables_3 = [Variable("?b", bowl_type)]
    groundings_3 = module((variables_3, object_descriptions))
    assert len(groundings_3) == 0
    
    # 5. Test Case: More variables than available objects.
    variables_4 = [
        Variable("?c1", cup_type),
        Variable("?c2", cup_type),
        Variable("?c3", cup_type),
    ]
    groundings_4 = module((variables_4, object_descriptions))
    assert len(groundings_4) == 2
    expected_ids_4 = ["red_cup_0", "red_cup_2"]
    assert sorted([g.object_id for g in groundings_4]) == expected_ids_4
    
    # 6. Test Case: Case-insensitivity.
    cup_type_upper = Type("CUP")
    variables_5 = [Variable("?c", cup_type_upper)]
    groundings_5 = module((variables_5, object_descriptions))
    assert len(groundings_5) == 1
    assert groundings_5[0].object_id == "red_cup_0" 