import numpy as np

from flask import Flask, request, Response
from flask.json import jsonify
from flask_caching import Cache

import uuid

from Genetic_algorithm_class import GeneticAlgorithm
import json
 
config = {
    "DEBUG":True,
    "CACHE_TYPE":"SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 10000
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)


@app.route('/new_experiment', methods = ['POST'])
def new_experiment():
    data = request.json

    n_params = data['n_params']
    n_pop = data['n_pop']

    uniqueid = str(uuid.uuid4().int)

    GeneticOptimizer = GeneticAlgorithm()

    cache.set(uniqueid, GeneticOptimizer)
    cache.set('most_recent_experiment', uniqueid)

    return jsonify({'uuid':uniqueid})

@app.route('/get_open_experiment', methods = ['GET'])
def get_open_experiment():
    unique_id = cache.get('most_recent_experiment')
    cache.set('most_recent_experiment', unique_id)
    return jsonify({'uuid':unique_id})


@app.route('/complete_trial', methods = ['POST'])
def complete_trial():
    data = request.json
    uniqueid = data['uuid']
    trial_index = int(data['trial_index'])
    metric = data['metric']
    distances = data['distances']

    try:
        extra_data = data['extra_data']
    except KeyError:
        extra_data = None
    #std = data['std']

    #TODO: data cleaning on this
    
    print('trial_index: ', trial_index)


    print(f'updating GASolver for experiment {uniqueid}')
    GeneticSolver = cache.get(uniqueid)
    GeneticSolver.update(trial_index, mean, extra_data = extra_data)
    cache.set(uniqueid, BoTorchSolver)
    print(f'set cache for experiment {uniqueid}')
    cache.set('most_recent_experiment', uniqueid)
    return('Updated experiment data')


#@app.route('/check_trials')
#def check_trials():
#    ax_client = cache.get('ax_client')
#    trials = ax_client.get_trials_data_frame()['trial_index'].to_list()
#    return(f'Current trials are {trials}')


@app.route('/get_next_generation', methods = ['POST'])
def get_next_generation():

    data = request.json
    uniqueid = data['uuid']
    # since this is time intensive, should consider how to handle that
    # option 1: Spawn a background thread, user checks back later to get result 

    GeneticSolver = cache.get(uniqueid)
    parameterization, trial_index = GeneticSolver.get_next_generation()
    cache.set(uniqueid, GeneticSolver)
    print(f'cached GeneticSolver for experiment UUID {uniqueid}')

    print('trial index when getting trial: ', trial_index)

    data = {'trial_index':trial_index, "parameterization":parameterization}
    return jsonify(data)

@app.route('/observability_data', methods = ['POST'])
def get_observability_data():
    data = request.json
    uniqueid = data['uuid']

    #try:
    with open(f'servedata_{uniqueid}.json', 'rt') as f:
        servedata = json.load(f)


    return jsonify(servedata)