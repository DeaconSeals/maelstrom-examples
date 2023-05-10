import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
import numpy as np
from maelstrom.population import GeneticProgrammingPopulation

@jit
def cartesian_to_polar(state, r_bound):
    return jnp.array((lax.min(r_bound, lax.sqrt(jnp.sum(lax.mul(state,state)))), lax.atan2(state[0], state[1])))

@jit
def polar_to_cartesian(r, angle):
    return jnp.array((lax.mul(r,lax.cos(angle)), lax.mul(r,lax.sin(angle))))

@jit
def update_point(state, angle, distance, r_bound):
    shift = polar_to_cartesian(distance, angle)
    action_state = cartesian_to_polar(lax.add(state, shift), r_bound)
    return polar_to_cartesian(action_state[0], action_state[1])

@jit
def move(pred_angle, prey_angle, pred_speed, prey_speed, r_bound, state):
    return jnp.concatenate((update_point(state[:2], pred_angle, pred_speed, r_bound), update_point(state[2:], prey_angle, prey_speed, r_bound)))

@jit
def euclidean_distance(A, B):
    return lax.sqrt(lax.add(lax.pow(lax.sub(A[0], B[0]), 2.0 ), lax.pow(lax.sub(A[1], B[1]), 2.0)))

@jit
def agent_distance(state):
    return euclidean_distance(state[:2], state[2:])

@jit
def check_collision(state, agent_radius):
    return jnp.greater(lax.mul(agent_radius, 2.0), agent_distance(state))

@jit
def generate_observations(state, pred_last_action, prey_last_action, r_bound):
    pred_polar = cartesian_to_polar(state[:2], r_bound)
    prey_polar = cartesian_to_polar(state[2:], r_bound)
    return jnp.array((
        pred_last_action,
        prey_last_action,
        agent_distance(state),
        euclidean_distance(jnp.array((0.0,0.0)), state[:2]),
        pred_polar[0],
        pred_polar[1],
        prey_polar[0],
        prey_polar[1],
        lax.sub(1.0, pred_polar[0]),
        lax.sub(1.0, prey_polar[0]),
        lax.atan2(lax.sub(state[3], state[1]), lax.sub(state[2], state[0])),
        lax.atan2(lax.sub(state[1], state[3]), lax.sub(state[0], state[2]))
        ))


map_move = jit(vmap(move, (0, 0, None, None, None, 0)))
map_move2d = jit(vmap(map_move, (0, 0, None, None, None, 0)))
map_collision = jit(vmap(check_collision, (0, None)))
map_collision2d = jit(vmap(map_collision, (0, None)))
map_observations = jit(vmap(generate_observations, (0, 0, 0, None)))
map_observations2d = jit(vmap(map_observations, (0, 0, 0, None)))

def round_robin_evaluation(
    predators: GeneticProgrammingPopulation,
    prey: GeneticProgrammingPopulation,
    agent_radius: float,
    time_limit: int,
    predator_move_speed: float,
    prey_move_speed: float,
    **kwargs
):
    # Instantiate executable lambda functions
    predators.build()
    prey.build()

    # Create JIT-compiled and vectorized versions of controllers
    v_pred = [jax.jit(jax.vmap(jax.jit(ind.func), (0,))) for ind in predators.population]
    v_prey = [jax.jit(jax.vmap(jax.jit(ind.func), (0,))) for ind in prey.population]

    # Define starting game state
    initial_state = jnp.array([-0.5, 0.0, 0.5, 0.0])
    competition_shape = (len(v_pred), len(v_prey))
    state = jnp.broadcast_to(initial_state, (len(v_pred), len(v_prey), initial_state.shape[0]))
    
    # Instantiate intermediate variables
    pred_actions = jnp.zeros((competition_shape))
    prey_actions = jnp.zeros((competition_shape))
    dones = jnp.zeros(competition_shape, dtype=jnp.bool_)
    fitnesses = jnp.full(competition_shape, time_limit)
    r_bound = 1-agent_radius
    
    # Time-limited game loop
    for time in range(time_limit):
        # Generate observations from 2D vectorized JIT-compiled function
        observations = map_observations2d(state, pred_actions, prey_actions, r_bound)

        # Evaluate agent controllers to generate actions
        pred_actions = jnp.array(np.array([fun(curr_observation) for fun, curr_observation in zip(v_pred, observations)]))
        prey_actions = lax.transpose(jnp.array(np.array([fun(curr_observation) for fun, curr_observation in zip(v_prey, lax.transpose(observations, (1,0,2)))])), (1,0))
        
        # Determine next state using 2D vectorized JIT-compiled function
        state = map_move2d(pred_actions, prey_actions, predator_move_speed, prey_move_speed, r_bound, state).block_until_ready()
        # Identify agent collisions using 2D vectorized JIT-compiled function
        collisions = map_collision2d(state, agent_radius)
        # Calculate fitness of games in which collision occured in this timestep
        fitnesses = jnp.where(jnp.bitwise_and(collisions, jnp.bitwise_not(dones)), jnp.full(competition_shape, time), fitnesses)
        dones = jnp.bitwise_or(dones, collisions)

    # Assign negated predator fitness
    for ind, scores in zip(predators.population, fitnesses):
        ind.fitness = time_limit - jnp.mean(scores)

    # Assign prey fitness
    for ind, scores in zip(prey.population, lax.transpose(fitnesses, (1,0))):
        ind.fitness = jnp.mean(scores)

    # Generate data log
    log = {}
    predator_fitnesses = [ind.fitness for ind in predators.population]
    log['avg_predator_fitness'] = sum(predator_fitnesses)/len(predator_fitnesses)
    log['max_predator_fitness'] = max(predator_fitnesses)

    prey_fitnesses = [ind.fitness for ind in prey.population]
    log['avg_prey_fitness'] = sum(prey_fitnesses)/len(prey_fitnesses)
    log['max_prey_fitness'] = max(prey_fitnesses)

    return log, len(predator_fitnesses)*len(prey_fitnesses)