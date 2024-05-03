from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render

import numpy as np

from flask import Flask, request
from flask.json import jsonify
from flask_caching import Cache

import uuid

from ax_serving import ax_solver
 
config = {
    "DEBUG":True,
    "CACHE_TYPE":"SimpleCache",
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)


@app.route('/new_experiment', methods = ['POST'])
def new_experiment():

    data = request.json

    n_params = data['n_params']
    n_random_trials = data['n_random_trials']
    n_bo_trials = data['n_bo_trials']

    uniqueid = str(uuid.uuid4().int)

    AxSolver = ax_solver.AxSolver(n_params, n_random_trials, n_bo_trials)
    cache.set(uniqueid, AxSolver)

    return jsonify({'uuid':uniqueid})

@app.route('/complete_trial', methods = ['POST'])
def complete_trial():
    data = request.json

    #TODO: Update API spec so mean, cov gets passed explicitly rather than as a tuple

    uniqueid = data['uuid']
    trial_index = int(data['trial_index'])
    metric = data['metric']
    mean = data['mean']
    std = data['std']
    
    print('trial_index: ', trial_index)
    raw_data = {f'{metric}':(float(mean), float(std))}

    print('raw update data: ', raw_data)

    # get our result values back to tuples 
    #for entry in rawdata:
    #    cleaned_result = ax_solver.clean_results_json(entry['results'])
    #    entry['results'] = cleaned_result
    #    cleaned_data.append(entry)

    AxSolver = cache.get(uniqueid)
    AxSolver.update(trial_index, raw_data)
    cache.set(uniqueid, AxSolver)
    return('Updated experiment data')

@app.route('/check_trials')
def check_trials():
    ax_client = cache.get('ax_client')
    trials = ax_client.get_trials_data_frame()['trial_index'].to_list()
    return(f'Current trials are {trials}')


#@app.route('/complete_trial')
#def complete_trial():
#    ax_client = cache.get('ax_client')
#    data = request.json
#    # add a bunch of json validation stuff here for real version
#    # validate that data has trial index and results with correct values
#    # 1. check that trial index is in RUNNING status
#    # 2. Check that results keys match results expected by ax_client
#    # 3. #

#    trial_index = data['trial_index']
#    results = data['results']
#    ax_client.complete_trial(trial_index, raw_data = ax_setup.clean_results_json(results))
#    cache.set('ax_client', ax_client)
#    return(f'Updated experiment for trial {trial_index}')


@app.route('/get_next_trial', methods = ['POST'])
def get_next_trial():

    data = request.json
    uniqueid = data['uuid']
    # since this is time intensive, should consider how to handle that
    # option 1: Spawn a background thread, user checks back later to get result 

    AxSolver = cache.get(uniqueid)
    parameterization, trial_index = AxSolver.ask()
    cache.set(uniqueid, AxSolver)

    print('trial index when getting trial: ', trial_index)

    data = {'trial_index':trial_index, "parameterization":parameterization}
    return jsonify(data)