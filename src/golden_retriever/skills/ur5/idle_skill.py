class IdleSkill:
    def __init__(self, name, description, parameters):
        """

        Args:
            name:
            description:
            parameters:
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
