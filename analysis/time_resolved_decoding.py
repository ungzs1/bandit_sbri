import pandas as pd
import numpy as np
import datetime
import os
import concurrent.futures
import traceback
import xarray as xr
import logging

from popy.population_decoders import run_decoder
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

def save_results(xr, floc):
    xr.to_netcdf(os.path.join(floc, 'scores.nc'))

### Set parameters

PARAMS = {
    'conditions': ['stay_value', 'feedback', 'target'],
    'group_targets': None, #['target', 'target_shuffled'],
    'K_fold': 10,
    'step_len': .1,
    'n_perm': 100, 
    'n_extra_trials': (0, 0),
    'floc': os.path.join(cfg.PROJECT_PATH_LOCAL, 'data', 'results', 'simple_decoder'),
    'msg': 'An example decoder of behavioral variables from neural data.',
}

### Run

if __name__ == '__main__':
    init_io(PARAMS)  # Initialize logging and create results folder

    sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information
    
    n_cores = np.min([10, os.cpu_count()-1])  # get number of cores in the machine
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # submit jobs
        futures, future_proxy_mapping = [], {}
        for monkey, session in sessions:
            #print(f'Running decoder for monkey {monkey} and session {session}...\n')
            future = executor.submit(run_decoder, monkey, session, PARAMS, load_data=False, save_data=False)  # Run decoder for each session
            futures.append(future)
            future_proxy_mapping[future] = (monkey, session)

        # wait for results, save them
        count = 0
        xrs = []  
        for future in concurrent.futures.as_completed(futures):
            try:
                res, session_log = future.result()
                monkey_fut, session_fut = future_proxy_mapping[future]

                # Append results to existing results and save after each session
                if len(xrs) == 0:  # First result - save directly
                    xrs = res
                else:  # Not first result - concatenate to existing results
                    xrs = xr.concat([xrs, res], dim='session')
            
                # Save results after each session
                save_results(xrs, PARAMS['floc'])  # Save results after each session

                # Log progress
                for line in session_log:
                    logging.info(line)
                logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                # print log
                print(f'Progress: {count+1}/{len(sessions)} sessions finished.\n')
                count += 1

            except Exception as e:  # Catch exceptions and log them
                logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                traceback.print_exc()  # Print traceback (?)

    end_log()
    print(f'Finished all on {datetime.datetime.now()}')