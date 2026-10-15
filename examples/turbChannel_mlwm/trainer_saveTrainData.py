# General imports
import sys
import os
import argparse
import logging
import numpy as np
from time import sleep, perf_counter
from os.path import exists
import socket
import datetime

# MPI
from mpi4py import MPI

# SmartRedis imports
from smartredis import Client
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

## Define logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file,mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

## Initialize Redis clients
def init_client(SSDB, args, logger_init):
    if (args.dbnodes==1):
        tic = perf_counter()
        client = Client(address=SSDB,cluster=False)
        toc = perf_counter()
    else:
        tic = perf_counter()
        client = Client(address=SSDB,cluster=True)
        toc = perf_counter()
    if (args.logging=='verbose'):
        logger_init.info('%.8e',toc-tic)
    return client

## Main
def main():
    # MPI import and initialization
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    #rankl = int(os.getenv("PALS_LOCAL_RANKID"))
    #rankl = int(os.getenv("SLURM_LOCALID"))
    rankl = rank
    print(f'Rank {rank}/{size}, local rank {rankl} says hello from {name}', flush=True)
    comm.Barrier()
    if (rank == 0):
        print("MPI COMM initialized\n", flush=True)

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    parser.add_argument('--device',default='cpu',help='Device to run on')
    parser.add_argument('--ppn',default=1,type=int,help='Number of processes per node')
    parser.add_argument('--logging',default='no',help='Level of performance logging')
    args = parser.parse_args()

    comm.Barrier()
    if (rank == 0):
        print("Parse Arguments initialized\n", flush=True)

    # Create log files
    time_meta = 0.
    if (args.logging=='verbose'):
        logger_init = setup_logger('client_init', f'client_init_ml_{rank}.log')
        logger_meta = setup_logger('meta', f'meta_data_ml_{rank}.log')
        logger_data = setup_logger('train_data', f'train_data_ml_{rank}.log')
    else:
        logger_init = None
        logger_data = None

    comm.Barrier()
    if (rank == 0):
        print("Create Logs initialized\n", flush=True)

    # Initialize Torch Distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    master_addr = socket.gethostname() if rank == 0 else None
    master_addr = comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(2345)
    if (args.device=='cpu'): backend = 'gloo'
    elif (args.device=='cuda'): backend = 'nccl'

    comm.Barrier()

    # Initialize Redis clients on each rank
    address = os.getenv('SSDB')
    client = init_client(address, args, logger_init)
    comm.Barrier()
    if (rank == 0):
        print("All Python clients initialized\n", flush=True)
        print("Running Script for Saving Training Data\n", flush=True)

    # Pull metadata from database
    tensor_key = f"tensorInfo.{rankl}"    
    while True:
        if (client.poll_tensor(tensor_key,0,1)):
            tic = perf_counter()
            dataSizeInfo = client.get_tensor(tensor_key).astype('int32')
            toc = perf_counter()
            time_meta = time_meta + (toc-tic)
            break
    comm.Barrier()

    npts = dataSizeInfo[0]
    num_tot_tensors = dataSizeInfo[1]
    num_db_tensors = dataSizeInfo[2]
    head_rank = dataSizeInfo[3]
    ndIn = dataSizeInfo[4]
    ndOut = dataSizeInfo[5]

    #if (rank == 0):
    print("Retreived metadata from DB:")
    print(f"[{rankl}] Number of samples per tensor: {npts}")
    print(f"[{rankl}] Number of total tensors in all DB: {num_tot_tensors}")
    print(f"[{rankl}] Number of tensors in local DB: {num_db_tensors}")
    print(f"[{rankl}] Number of inputs and outputs to model: {ndIn},{ndOut}", flush=True)

    # Training setup and variable initialization
    istep = -1 # initialize the simulation step number to -1
    step_list = [] # initialize an empty list containing all the steps sent

    # While loop that checks when training data is available on database
    if (rank == 0):
        print("Starting training loop ... \n")

    target_MB = 100;
    bytes_per_sample = ndIn * 4 + ndOut * 4   # float32
    samples_per_batch = npts
    batch_size_MB = (bytes_per_sample * samples_per_batch) / 1024**2
    chunk_size = max(1, int(target_MB / batch_size_MB))
    if rank == 0:
      print(f"Batch size: {batch_size_MB:.2f} MB")
      print(f"Chunk size: {chunk_size}")
      print(f"Estimated file size: {chunk_size * batch_size_MB:.2f} MB")
    #chunk_size = 10
    chunk_id = 0
    buffer_X = []
    buffer_Y = []
    while True:
        # check to see if the time step number has been sent to database, if not cycle
        if (client.poll_tensor("step",0,1)):
            tic = perf_counter()
            tmp = client.get_tensor('step').astype('int32')
            toc = perf_counter()
            time_meta = time_meta + (toc-tic)
        else:
            #sleep(0.001)
            continue

        # new data is available in database so update Dataset and DataLoader
        if (istep != tmp[0]): 
            istep = tmp[0]
            step_list.append(istep)
            batch =  int(num_db_tensors*len(step_list)/args.ppn)
            if (rank == 0):
                print("\nGetting new training data from DB ...")
                print(f"Added time step {istep} to training data\n", flush=True)
            
            #Load data from DB 
            data_key = f"x.{rankl}.{istep}"    
            input_tmp = client.get_tensor(data_key).astype(np.float32)
            inputs  = input_tmp[:,:ndIn]
            outputs = input_tmp[:,ndIn:]

            #Concatenate and save to disk
            buffer_X.append(inputs)
            buffer_Y.append(outputs)

            if len(buffer_X) == chunk_size:
              X = np.concatenate(buffer_X)
              Y = np.concatenate(buffer_Y)

              if (rank == 0):
                print("\nSaving chunk of training data from DB to disk ...")

              np.savez(f"channel_rank{rankl:04d}_chunk{chunk_id:04d}.npz",
                       features=X, target=Y)

              buffer_X, buffer_Y = [], []
              chunk_id += 1

            #Remove data from DB once loaded
            client.delete_tensor(data_key)  
            if (rank == 0):
                print(f"Deleted time step {istep} from database.\n", flush=True)


    if (args.logging=='verbose'):
        logger_meta.info('%.8e',time_meta)    

    # Exit and tell data loader to exit too
    comm.Barrier()
    if (rank%args.ppn == 0):
        print(f"[{rank}]: Telling SOD2D to quit ... \n")
        arrMLrun = np.int32(np.zeros(2))
        client.put_tensor("check-run",arrMLrun)

    if (rank==0):
        print("Exiting ...")
    

###
if __name__ == '__main__':
    main()
