import pandas as pd
import numpy as np
import datetime
import os
import concurrent.futures
import traceback
import logging

from popy.neural_value_helpers import load_data_custom, time_resolved_decoder_all_time, add_neural_value_coord
import popy.config as cfg

### Load metadata

def get_all_sessions():
    return [('ka', '210322'), ('ka', '020622'), ('po', '210422'), ('po', '240921')]


### Configure logging

def end_log():
    # start time is the first log entry
    end_time = datetime.datetime.now()
    logging.info(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

def init_io(PARAMS):
    os.makedirs(PARAMS['floc'], exist_ok=True)

    # configure logging
    logging.basicConfig(filename=os.path.join(PARAMS['floc'], 'log.txt'),
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w')  # 'w' mode will overwrite the log file

    start_time = datetime.datetime.now()
    logging.info("PARAMS:")
    for key, value in PARAMS.items():
        logging.info(f'{key}: {value}')
    logging.info(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

### Save results

def save_results(dfs_all, floc, fname):
    dfs_all.to_pickle(os.path.join(floc, fname))

### Run

def run_neural_value_extraction(monkey, session, PARAMS):
    print('running: ', monkey, session)
    #monkey, session, subregion = 'ka', '210322', 'vLPFC'

    neural_dataset = load_data_custom(monkey, session, n_extra_trials=(-1, 0), sr=10)

    dfs = []
    for subregion in np.unique(neural_dataset.subregion.data):
        neural_dataset_temp = neural_dataset.sel(unit=neural_dataset.subregion==subregion)
        
        #project neural data to the neural value space, get across-time decodability matrix
        _, data_projected = time_resolved_decoder_all_time(neural_dataset_temp, target='R_1')

        t_interest_value = (1.5, 3.5)

        # get neural value for the current trial and the previous one
        data_projected = add_neural_value_coord(data_projected, t_interest=t_interest_value)
        '''data_projected = data_projected.rename({'V_t': f'V_t_{t_interest_value[0]}_{t_interest_value[1]}'})
        data_projected = data_projected.rename({'V_t_p1': f'V_t_p1_{t_interest_value[0]}_{t_interest_value[1]}'})
        data_projected = data_projected.rename({'dV': f'dV_{t_interest_value[0]}_{t_interest_value[1]}'})'''

        behav_new = data_projected.mean('time').coords.to_dataset().to_dataframe().reset_index()
        behav_new['monkey'] = monkey
        behav_new['session'] = session
        #behav_new['area'] = neural_dataset.area
        behav_new['subregion'] = subregion
        dfs.append(behav_new)
            
    behav_new = pd.concat(dfs, ignore_index=True)
    behav_new = behav_new[['monkey', 'session', 'subregion'] + [c for c in behav_new.columns if c not in ['monkey', 'session', 'subregion']]]
        
    session_log = []
    return behav_new, session_log


### RUN THIS FOR NEURAL VALUE EXTRACTION
PARAMS = {
    'floc': os.path.join(cfg.PROJECT_PATH_LOCAL, 'data', 'results', 'behav_neural_value'),
    'msg': 'extracting plots',
}

if __name__ == '__main__':
    init_io(PARAMS)  # Initialize logging and create results folder

    sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information

    n_cores = np.min([100, os.cpu_count()-1])  # get number of cores in the machine
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # submit jobs
        futures, future_proxy_mapping = [], {}
        for (monkey, session) in sessions:
            future = executor.submit(run_neural_value_extraction, monkey, session, PARAMS)  # Run decoder for each session
            futures.append(future)
            future_proxy_mapping[future] = (monkey, session)

        # wait for results, save them
        count_good = 0
        count_bad = 0

        dfs_all = []  
        for future in concurrent.futures.as_completed(futures):
            try:
                res, session_log = future.result()
                monkey_fut, session_fut = future_proxy_mapping[future]

                # Append results to existing results and save after each session
                if len(dfs_all) == 0:
                    dfs_all = res
                else:
                    dfs_all = pd.concat([dfs_all, res], ignore_index=True)
            
                # Save results after each session
                save_results(dfs_all, PARAMS['floc'], fname='behav.pkl')  # Save results after each session

                # Log progress
                for line in session_log:
                    logging.info(line)
                logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                count_good += 1

            except Exception as e:  # Catch exceptions and log them
                logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                traceback.print_exc()  # Print traceback (?)

                count_bad += 1

            print(f'Progress: {count_good + count_bad}/{len(sessions)} failed: {count_bad}')


    end_log()
    print(f'Finished all on {datetime.datetime.now()}')
