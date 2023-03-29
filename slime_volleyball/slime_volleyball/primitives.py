import random
from maelstrom.genotype import GeneticTree


QVALUES = "qvalues"
GENERAL = "general"
FLOAT = "float"


@GeneticTree.declare_primitive(QVALUES, FLOAT, ())
def action(context):
    return context["action"]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_1(context):
    return context["obs"][0]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_2(context):
    return context["obs"][1]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_3(context):
    return context["obs"][2]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_4(context):
    return context["obs"][3]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_5(context):
    return context["obs"][4]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_6(context):
    return context["obs"][5]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_7(context):
    return context["obs"][6]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_8(context):
    return context["obs"][7]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_9(context):
    return context["obs"][8]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_10(context):
    return context["obs"][9]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_11(context):
    return context["obs"][10]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def obs_12(context):
    return context["obs"][11]


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


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def small_random(context):
    return random.uniform(-3, 3)


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def large_random(context):
    return random.uniform(-20, 20)


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def max_floats(float_1, float_2):
    return float_1 if float_1 > float_2 else float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, (FLOAT, FLOAT))
def min_floats(float_1, float_2):
    return float_1 if float_1 < float_2 else float_2


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def if_then_else1(context):
    if context["obs"][0] > context["obs"][1]:
        return context["obs"][2]
    return context["obs"][3]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def if_then_else2(context):
    if context["obs"][1] > context["obs"][2]:
        return context["obs"][3]
    return context["obs"][4]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def if_then_else3(context):
    if context["obs"][4] > context["obs"][5]:
        return context["obs"][6]
    return context["obs"][7]


@GeneticTree.declare_primitive(GENERAL, FLOAT, ())
def if_then_else4(context):
    if context["obs"][1] > context["obs"][7]:
        return context["obs"][11]
    return context["obs"][3]
