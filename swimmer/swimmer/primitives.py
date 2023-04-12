"""Primitives for the genetic tree."""
from maelstrom.genotype import GeneticTree

OUTPUT = "OUTPUT"
FLOAT = "FLOAT"
GENERAL = "GENERAL"


@GeneticTree.declare_primitive(GENERAL, OUTPUT, (FLOAT, FLOAT))
def create_output(float_1, float_2):
    """Output a tuple of two floats, where each float is coerced to be between -1 and 1."""
    return float_1, float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def add(float_1, float_2):
    """Add two floats."""
    return float_1 + float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def sub(float_1, float_2):
    """Add two floats."""
    return float_1 - float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def mul(float_1, float_2):
    """Add two floats."""
    return float_1 * float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def div(float_1, float_2):
    """Add two floats."""
    return float_1 / float_2 if float_2 != 0 else float_1


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context_0(context):
    """Get the first context value."""
    return context[0]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context_1(context):
    """Get the second context value."""
    return context[1]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context_2(context):
    """Get the third context value."""
    return context[2]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context_3(context):
    """Get the fourth context value."""
    return context[3]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context_4(context):
    """Get the fifth context value."""
    return context[4]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context_5(context):
    """Get the sixth context value."""
    return context[5]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context_6(context):
    """Get the seventh context value."""
    return context[6]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context_7(context):
    """Get the eighth context value."""
    return context[7]
