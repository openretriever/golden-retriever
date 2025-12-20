import importlib

import pydantic

# TODO Note
# - implement base class for abstract skill classes and dummy skill classes
# - we follow bilevel planning framework - we have operators and parameterized skills


class OperatorRegistry:
    def __init__(self):
        self.operators = {}

    def register_operator(self, operator_name, module_name, class_name):
        module = importlib.import_module(module_name)
        operators_class = getattr(module, class_name)
        self.operators[operator_name] = operators_class()

    def get_operator(self, operator_name):
        return self.operators.get(operator_name, None)


class SkillRegistry:
    def __init__(self):
        self.skills = {}

    def register_skill(self, skill_name, module_name, class_name):
        module = importlib.import_module(module_name)
        skill_class = getattr(module, class_name)
        self.skills[skill_name] = skill_class()

    def get_skill(self, skill_name):
        return self.skills.get(skill_name, None)


# Operator definition
class AbstractOperator:
    def __init__(self, name, preconditions, effects, skill_name):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects
        self.skill_name = skill_name


class AbstractOpenLoopSkill:
    # Abstract skll class - all skills for different robots should inherit from this class
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class AbstractClosedLoopSkill:
    # Abstract skll class - all skills for different robots should inherit from this class
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class SkillParams(pydantic.BaseModel):
    # Base class for defining parameters for parameterized skills
    pass
