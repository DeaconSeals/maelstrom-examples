import jax
import jax.numpy as jnp
from jax import lax, jit, vmap

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
def euclidean_distance(A, B)
    return lax.sqrt(lax.add(lax.pow(lax.sub(A[0], B[0]), 2 ), lax.pow(lax.sub(A[1], B[1]), 2)))

@jit
def agent_distance(state):
    return euclidean_distance(state[:2], state[2:])

@jit
def check_collision(state, agent_radius):
    return jnp.greater(lax.mul(agent_radius, 2), agent_distance(state))

@jit
def generate_observations(state, pred_last_action, prey_last_action, r_bound):
    pred_polar = cartesian_to_polar(state[:2], r_bound)
    prey_polar = cartesian_to_polar(state[2:], r_bound)
    return jnp.array(
        pred_last_action,
        prey_last_action,
        agent_distance(state),
        euclidean_distance(jnp.array(0,0), state[:2]),
        pred_polar[0],
        pred_polar[1],
        prey_polar[0],
        prey_polar[1],
        lax.sub(1, pred_polar[0]),
        lax.sub(1, prey_polar[0]),
        lax.atan2(lax.sub(state[3], state[1]), lax.sub(state[2], state[0])),
        lax.atan2(lax.sub(state[1], state[3]), lax.sub(state[0], state[2]))
        )


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
    predators.build()
    prey.build()
    v_pred = [jax.jit(jax.vmap(ind.func, (0,))) for ind in predators.population]
    v_prey = [jax.jit(jax.vmap(ind.func, (0,))) for ind in prey.population]
    initial_state = jnp.array([-0.5, 0.0, 0.5, 0.0])
    competition_shape = (len(v_pred), len(v_prey))
    state = jnp.broadcast_to(initial_state, (len(v_pred), len(v_prey), initial_state.shape[0]))
    pred_actions = jnp.zeros((competition_shape))
    prey_actions = jnp.zeros((competition_shape))
    dones = jnp.zeros(competition_shape, dtype=jnp.bool_)
    fitnesses = jnp.full(competition_shape, time_limit)
    r_bound = 1-agent_radius
    for time in range(time_limit):
        observations = map_observations2d(state, pred_actions, prey_actions, r_bound)
        pred_actions = jnp.array(np.array([fun(curr_observation) for fun, curr_observation in zip(v_pred, observations)]))
        prey_actions = lax.transpose(jnp.array(np.array([fun(curr_observation) for fun, curr_observation in zip(v_prey, lax.transpose(observations, (1,0,2)))])), (1,0))
        state = map_move2d(pred_actions, prey_actions, predator_move_speed, prey_move_speed, r_bound, state).block_until_ready()
        collisions = map_collision2d(state, agent_radius)
        fitnesses = fitnesses.where(jnp.bitwise_and(collisions, jnp.notwise_not(dones)), jnp.full(competition_shape, time), dones)
        dones = jnp.bitwise_or(dones, collisions)

    for ind, scores in zip(predators.population, fitnesses):
        ind.fitness = time_limit - jnp.mean(scores)

    for ind, scores in zip(prey.population, lax.transpose(fitnesses, (1,0))):
        ind.fitness = jnp.mean(scores)