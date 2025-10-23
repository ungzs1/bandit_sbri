# # Summary
# 
# Lets fit the models to the actual data.

# ---
# # Setup

# @title Imports
import os
import gymnasium as gym
import pandas as pd
from skopt.space import Real

from popy.io_tools import load_behavior
from popy.behavior_data_tools import convert_column_format
from popy.simulation_tools import WSLSAgent_custom, QLearner, ForagingAgent
from popy.config import PROJECT_PATH_LOCAL
from popy.simulation_helpers import fit_simulate


def get_data_custom(monkey):
    if monkey in ['ka', 'po']:
        behav_monkey = load_behavior(monkey)
    else:
        raise ValueError(f'Unknown monkey: {monkey}')
    behav_monkey = convert_column_format(behav_monkey, original='behavior')

    behav_monkey = behav_monkey.dropna()

    return behav_monkey

def res_to_dataframe(results):
    res_all = pd.DataFrame.from_dict(results, orient='index').reset_index()  # convert to dataframe
    res_all = res_all.rename(columns={'index': 'Model'})  # rename index column to Model

    cols = list(res_all.columns)
    first_columns = ['Model', 'epsilon', 'alpha', 'alpha_unchosen', 'beta', 'V0', 
                     'forgetting_rate', 'forgetting_threshold', 
                     'stickyness_bias', 
                     'b1', 'b2', 'b3', 
                     'abandoned_bias', 'abandoned_decay']
    present = [c for c in first_columns if c in cols]
    rest = [c for c in cols if c not in present]
    new_columns = present + rest

    return res_all[new_columns]

def behavs_to_dataframe(behaviors_simulated, monkey):
    # combine all behaviors into one dataframe
    behavs = []

    # Process simulations: start with the real behavior
    behav_monkey = behaviors_simulated[f'MONKEY {monkey.upper()}']
    behav_monkey = behav_monkey.drop(columns=['switch'])
    behav_monkey['model'] = 'recording'
    behavs.append(behav_monkey)

    # then add the simulated behaviors
    for key, behav_temp in behaviors_simulated.items():
        if key != f'MONKEY {monkey.upper()}':
            behav_temp['monkey'] = monkey
            behav_temp['session'] = 0
            behav_temp['model'] = key
            behavs.append(behav_temp)

    behaviors_simulated_all = pd.concat(behavs, axis=0)

    # reorder columns
    cols = ['monkey', 'model', 'session'] + [col for col in behav_monkey.columns if col not in ['monkey', 'session', 'model']]
    behaviors_simulated_all = behaviors_simulated_all[cols]

    # reset index
    behaviors_simulated_all = behaviors_simulated_all.reset_index(drop=True)

    return behaviors_simulated_all

def save_res_and_behav(results, behaviors_simulated, monkey, floc):
    res_all = res_to_dataframe(results)
    floc_res_temp = os.path.join(floc, f'simulation_results_{monkey}.csv')
    res_all.to_csv(floc_res_temp, index=False)

    # save behaviors
    behaviors_simulated_all = behavs_to_dataframe(behaviors_simulated, monkey)
    floc_simulations_temp = os.path.join(floc, f'simulation_behaviors_{monkey}.pkl')
    behaviors_simulated_all.to_pickle(floc_simulations_temp)


# Init parameters

cv_splits = None
n_initial_points = 100
n_calls = 200
n_jobs = -1
verbose = False
n_simulation_trials = 100_000
make_plots = False

# Define parameter spaces
epsilon_range = Real(.01, .3, name='epsilon')
alpha_range = Real(0.01, 1, name='alpha')
alpha_unchosen_range = Real(0, .5, name='alpha_unchosen')
beta_range = Real(.5, 80.0, name='beta')
stickyness_range = Real(0.0, 50.0, name='stickyness_bias')
forgetting_rate_range = Real(0.0, 1.0, name='forgetting_rate')
forgetting_threshold_range = Real(0.0, 1.0, name='forgetting_threshold')
b1_range = Real(-50.0, 50.0, name='b1')
b2_range = Real(-5.0, 5.0, name='b2')
b3_range = Real(-50.0, 50.0, name='b3')
V0_range = Real(0.05, .4, name='V0')
abandoned_bias_range = Real(-50.0, 0.0, name='abandoned_bias')
abandoned_decay_range = Real(0.0, 1.0, name='abandoned_decay')

# Define file location to save results, create folder if it doesn't exist
floc = os.path.join(PROJECT_PATH_LOCAL, 'data', 'results', 'model_fitting')
os.makedirs(floc, exist_ok=True)

# Run parameter fitting per monkey
for monkey in ['ka', 'po']: 
    print(f'--- {monkey} ---')

    ### Get data
    behav_monkey = get_data_custom(monkey)  # get monkey data    
    env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=n_simulation_trials)  # Create the environment
    results, behaviors_simulated = {}, {f'MONKEY {monkey.upper()}': behav_monkey}  # Set container (to collect pandas series into a dataframe)



    ### Fit models to the data

    # ### Modified WSLS
    model_name = 'WSLS agent'
    agent_class = WSLSAgent_custom
    fixed_params = {}
    param_space = [epsilon_range]

    results[model_name], behaviors_simulated[model_name] = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=cv_splits, make_plots=make_plots, 
                                                                        n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
    print('Fitted WSLS agent')
    save_res_and_behav(results, behaviors_simulated, monkey, floc)



    # ### Simple RL agent
    model_name = 'Standard RL'
    agent_class = QLearner
    fixed_params = {'structure_aware': False}
    param_space = [alpha_range, beta_range]

    results[model_name], behaviors_simulated[model_name] = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=cv_splits, make_plots=make_plots, 
                                                                        n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
    # ### Simple RL agent - stickyness
    model_name = 'Standard RL - stickyness'
    agent_class = QLearner
    fixed_params = {'structure_aware': False}
    param_space = [alpha_range, beta_range, stickyness_range]

    results[model_name], behaviors_simulated[model_name] = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=cv_splits, make_plots=make_plots, 
                                                                        n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
    
    print('Fitted Standard RL agents')
    save_res_and_behav(results, behaviors_simulated, monkey, floc)



    # ### Inferential RL
    model_name = 'Inferential RL'
    agent_class = QLearner
    fixed_params = {'structure_aware': True}
    param_space = [alpha_range, beta_range]

    results[model_name], behaviors_simulated[model_name] = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=cv_splits, make_plots=make_plots, 
                                                                        n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
    # ### Inferential RL - stickyness
    model_name = 'Inferential RL - stickyness'
    agent_class = QLearner
    fixed_params = {'structure_aware': True}
    param_space = [alpha_range, beta_range, stickyness_range]

    results[model_name], behaviors_simulated[model_name] = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=cv_splits, make_plots=make_plots, 
                                                                        n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
    
    print('Fitted Inferential RL agents')
    save_res_and_behav(results, behaviors_simulated, monkey, floc)



    # ### Foraging 
    model_name = 'Foraging'
    agent_class = ForagingAgent
    fixed_params = {'reset_on_switch': True}
    param_space = [alpha_range, beta_range, V0_range]

    results[model_name], behaviors_simulated[model_name] = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=cv_splits, make_plots=make_plots, 
                                                                        n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
    
    print('Fitted Foraging agents')
    save_res_and_behav(results, behaviors_simulated, monkey, floc)
    print('saved all')