"""
primitives
"""
from maelstrom.genotype import GeneticTree

GENERAL = "General"
FLOAT = "Float"


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def action(context):
    return context["action"]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context0(context):
    return context["obs"][0]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def context1(context):
    return context["obs"][1]


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def add(float_1, float_2):
    return float_1 + float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def sub(float_1, float_2):
    return float_1 - float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def mul(float_1, float_2):
    return float_1 * float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def div(float_1, float_2):
    return float_1 / float_2 if float_2 != 0 else float_1
