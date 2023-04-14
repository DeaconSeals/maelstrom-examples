from maelstrom.genotype import GeneticTree
import math
import random

import jax
from jax import lax, jit
import jax.numpy as jnp

GENERAL = "General"
ANGLE = "Angle"
DISTANCE = "Distance"

#-------------------------------------- Leaf nodes --------------------------------------
@GeneticTree.declare_primitive(GENERAL, DISTANCE, (), 2, literal_init = True)
def distance_const(maximum):
    return random.random()*maximum

@GeneticTree.declare_primitive(GENERAL, ANGLE, (), 2*math.pi, literal_init = True)
def angle_const(maximum):
    return random.random()*maximum

@GeneticTree.declare_primitive(GENERAL, ANGLE, (), literal_init = True)
def predator_last_move():
    return 'context[0]'

@GeneticTree.declare_primitive(GENERAL, ANGLE, (), literal_init = True)
def prey_last_move():
    return 'context[1]'

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (), literal_init = True)
def distance_to_opponent():
    return 'context[2]'

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (), literal_init = True)
def predator_distance():
    return 'context[3]'

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (), literal_init = True)
def predator_angle():
    return 'context[4]'

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (), literal_init = True)
def prey_distance():
    return 'context[5]'

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (), literal_init = True)
def prey_angle():
    return 'context[6]'

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (), literal_init = True)
def predator_distance_to_wall():
    return 'context[7]'

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (), literal_init = True)
def prey_distance_to_wall():
    return 'context[8]'

@GeneticTree.declare_primitive(GENERAL, ANGLE, (), literal_init = True)
def predator_to_prey_angle():
    return 'context[9]'

@GeneticTree.declare_primitive(GENERAL, ANGLE, (), literal_init = True)
def prey_to_predator_angle():
    return 'context[10]'

#----------------------------------- Functional nodes -----------------------------------
@GeneticTree.declare_primitive(GENERAL, ANGLE, (ANGLE, ANGLE))
@jit
def add_angles(angle_0, angle_1):
    return jnp.mod(angle_0+angle_1+(2*math.pi), 2*math.pi)

@GeneticTree.declare_primitive(GENERAL, ANGLE, (ANGLE,))
@jit
def flip_angle(angle):
    return add_angles(angle, math.pi)

@GeneticTree.declare_primitive(GENERAL, ANGLE, (ANGLE, ANGLE))
@jit
def subtract_angles(angle_0, angle_1):
    return add_angles(angle_0, -angle_1)

@GeneticTree.declare_primitive(GENERAL, ANGLE, (ANGLE, ANGLE))
@jit
def average_angles(angle_0, angle_1):
    return lax.atan2((lax.sin(angle_0)+lax.sin(angle_1))/2,
                     (lax.cos(angle_0)+lax.cos(angle_1))/2)

@GeneticTree.declare_primitive(GENERAL, ANGLE, (ANGLE, DISTANCE))
@jit
def multiply_angle(angle, distance):
    return jnp.mod(angle*distance, 2*math.pi)


@GeneticTree.declare_primitive(GENERAL, DISTANCE, (DISTANCE, DISTANCE))
@jit
def add_distances(distance_0, distance_1):
    return lax.add(distance_0, distance_1)


@GeneticTree.declare_primitive(GENERAL, DISTANCE, (DISTANCE, DISTANCE))
@jit
def subtract_distances(distance_0, distance_1):
    return lax.sub(distance_0, distance_1)

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (DISTANCE, DISTANCE))
@jit
def multiply_distances(distance_0, distance_1):
    return lax.mul(distance_0, distance_1)

@GeneticTree.declare_primitive(GENERAL, DISTANCE, (DISTANCE, DISTANCE))
@jit
def divide_distances(distance_0, distance_1):
    return lax.div(lax.select(lax.eq(distance_1, 0.0), 2.0, distance_0), lax.select(lax.eq(distance_1, 0.0), 2.0, distance_1))
    

@GeneticTree.declare_primitive(GENERAL, ANGLE, (DISTANCE, DISTANCE, ANGLE, ANGLE))
@jit
def if_greater_than(distance_0, distance_1, angle_0, angle_1):
    return lax.select(lax.gt(distance_0, distance_1), angle_0, angle_1)