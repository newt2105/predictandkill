import pyscipopt as scp
import argparse
import random
import numpy as np
import os
import torch
from pyscipopt import SCIP_PARAMSETTING
import shutil
from helper import get_bg_data
from google.protobuf import text_format
from params.better_scip_fixedvariables_pb2 import ScriptOptions
from multiprocessing import Process, Queue, set_start_method
from threading import Thread, get_native_id
from nn_models.GCN import GNNPolicy_position as GNNPolicy, position_get

# Set the prefered compute device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# sử dụng GPU nếu có sẵn để tăng tốc độ tính toán. Nếu không có GPU, nó sẽ sử dụng CPU để thực hiện tính toán.

def getOptionsFromFile(path):
    _scriptOptions = ScriptOptions()
    with open(path, "r") as f:
        options_str = f.read()
        text_format.Merge(options_str, _scriptOptions)
    return _scriptOptions

def init_dirs(dir):
    try:
        if (os.path.exists(dir)):
            shutil.rmtree(dir)
    except OSError as e:
        print(e)
    try:
        os.mkdir(dir)
    except OSError as e:
        print(e)

def loadPolicy(modelpath):
    # Load the blueprints to the compute device
    policy = GNNPolicy().to(DEVICE)
    # Loas the trained state to compute device
    state = torch.load(modelpath, map_location=torch.device(DEVICE))
    # Merge the state data with the blueprint
    policy.load_state_dict(state)
    return policy

def PredictAndKill(policy, queue, instance_dir, solution_dir, log_dir, ScipTrustregionOptions):
    while(True):
        while queue.qsize():
            # Get the problem to solve3
            name = queue.get()
            # Get bipartite graph from the problem
            instancefile = f"{instance_dir}/{name}"
            A, v_map, v_nodes, c_nodes, b_vars = get_bg_data(instancefile)
            constraints_features = c_nodes.cpu()
            constraints_features[np.isnan(constraints_features)] = 1 # remove non-number values
            variables_features = v_nodes
            variables_features = position_get(variables_features)
            edge_indices = A._indices()
            edge_features = A._values().unsqueeze(1)
            edge_features = torch.ones(edge_features.shape)
            # Prediction
            prediction_result = policy(
                constraints_features.to(DEVICE),
                edge_indices.to(DEVICE),
                edge_features.to(DEVICE),
                variables_features.to(DEVICE),
            ).sigmoid().cpu().squeeze()
            # Align variable name between the prediction_result and the solver
            all_varname = []
            for varname in v_map:
                all_varname.append(varname)
            binary_varname = [all_varname[i] for i in b_vars]
            scores = [] # Get a prediction result list of (index, varname, PROBABILITY, -1, type)
            for i in range(len(v_map)):
                type="C"
                if all_varname[i] in binary_varname:
                    type="BINARY"
                scores.append([i, all_varname[i], prediction_result[i].item(), -1, type])
            scores.sort(key=lambda x:x[2], reverse=False) # Sort prediction list by PROBABILITY
            scores = [x for x in scores if x[4]=="BINARY"] # Get only BINARY vars
            # Generate a list of variables to fix their values
            fixer = 0
            count_1 = 0
            for i in range(len(scores)):
                if count_1 < ScipTrustregionOptions.MaxFixed1VarCount:
                    scores[i][3] = 1
                    count_1 += 1
                    fixer += 1
            count_0 = 0
            for i in range(len(scores)):
                if count_0 < ScipTrustregionOptions.MaxFixed0VarCount:
                    scores[i][3] = 0
                    count_0 += 1
                    fixer += 1
            # Load problem to SCIP
            m = scp.Model()
            m.setParam("limits/time", ScipTrustregionOptions.TimeLimit)
            m.hideOutput(ScipTrustregionOptions.HideOutput)
            m.setParam('randomization/randomseedshift', 0)
            m.setParam('randomization/lpseed', 0)
            m.setParam('randomization/permutationseed', 0)
            m.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)#MIP focus
            m.setLogfile(f"{log_dir}/{name}.log")
            m.readProblem(instancefile)
            # Implement trust region by fixing variables
            m_variables = m.getVars()
            m_v_map = {} # Build a dictionary (var_name, var)
            for var in m_variables:
                m_v_map[var.name] = var
            # Fix the variables values
            for i in range(len(scores)):
                target_variable = m_v_map[scores[i][1]] # Select variable to work on based on prediction results
                x_star = scores[i][3] # ignore variables which is not predicted and fixed
                if x_star < 0:
                    continue
                if x_star == 1:
                    m.fixVar(target_variable, 1)
                if x_star == 0:
                    m.fixVar(target_variable, 0)
            #COUNT UNFIXED VAR
            def count_unfixed_variables(policy, queue, instance_dir, ScipTrustregionOptions):
                unfixed_count = 0

                while queue.qsize():
                    # Get the problem to solve
                    name = queue.get()
                    # Get bipartite graph from the problem
                    instancefile = f"{instance_dir}/{name}"
                    A, v_map, v_nodes, _, _ = get_bg_data(instancefile)
                    constraints_features = v_nodes.cpu()
                    constraints_features[np.isnan(constraints_features)] = 1
                    variables_features = v_nodes
                    variables_features = position_get(variables_features)
                    edge_indices = A._indices()
                    edge_features = torch.ones(edge_indices.shape)
                    # Prediction
                    prediction_result = policy(
                        constraints_features.to(DEVICE),
                        edge_indices.to(DEVICE),
                        edge_features.to(DEVICE),
                        variables_features.to(DEVICE),
                    ).sigmoid().cpu().squeeze()

                    scores = []
                    for i, varname in enumerate(v_map):
                        scores.append([i, varname, prediction_result[i].item()])
                    scores.sort(key=lambda x: x[2], reverse=False)
                    
                    scores = scores[:ScipTrustregionOptions.MaxFixed0VarCount + ScipTrustregionOptions.MaxFixed1VarCount]
                    
                    unfixed_count += sum(1 for score in scores if score[2] < 0)

                return unfixed_count
            
            unfixed_var_count = count_unfixed_variables(policy, queue, instance_dir, ScipTrustregionOptions)
            if (unfixed_var_count <= 0):
                
                break

    # Optimize the problem and write the best soluton
    m.optimize()
    m.writeBestSol(f"{solution_dir}/{name}.sol")
    print(f"[{get_native_id()}] scip_fixedvariables: {name} k0={count_0} k1={count_1} ({queue.qsize()} left)")

def _main(ScriptOptions):
    """
    This is the real main function. The logics are placed in here.
    """
    print("==== better_scip_fixedvariables ====")
    print(ScriptOptions)
    # Set directories
    _instances_dir = f"{ScriptOptions.DataPath}/problems/{ScriptOptions.ProblemName}"
    _base_dir = f"{ScriptOptions.DataPath}/results/{ScriptOptions.ProblemName}@better_scip_fixedvariables"
    _solutions_dir = f"{_base_dir}/solutions"
    _log_dir = f"{_base_dir}/solver_logs"
    _model_path = f"{ScriptOptions.ModelPath}"
    # Initialize required directories
    init_dirs(_base_dir)
    init_dirs(_solutions_dir)
    init_dirs(_log_dir)
    # Seed the randomizers
    random.seed(ScriptOptions.Seed)
    torch.manual_seed(ScriptOptions.Seed)
    torch.cuda.manual_seed(ScriptOptions.Seed)
    # Load the trained model
    policy = loadPolicy(_model_path)
    # Get the new instances (problems) to solve
    instances_files = [fn for fn in os.listdir(_instances_dir)
                        if (fn.endswith("mps"))]
    # Multi-threading
    queue = Queue()
    for instancefile in instances_files:
        queue.put(instancefile)
    ps = []
    for i in range(ScriptOptions.WorkerCount):
        p = Thread(target=scip_solve_fixedvariables, 
                    args=(policy, 
                            queue,
                            _instances_dir, 
                            _solutions_dir, 
                            _log_dir, 
                            ScriptOptions.ScipTrustregionOptions
                            ))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    print("++++ better_scip_fixedvariables: done ++++")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Options for original_scip_solvers")
    parser.add_argument("PATH", type=str,
                        help="Path to Options File or Directory of Option Files.")
    parser.add_argument("--recurse", nargs='?', default=False, type=bool,
                        help="Scan the PATH for Options Files.")
    args = parser.parse_args()
    set_start_method("spawn")
    if (args.recurse):
        file_list = [fn for fn in os.listdir(args.PATH)
                     if (fn.endswith("pb2"))]
        for file in file_list:
            _scriptOptions = getOptionsFromFile(f"{args.PATH}/{file}")
            _main(_scriptOptions)
    else:
        _scriptOptions = getOptionsFromFile(args.PATH)
        _main(_scriptOptions)