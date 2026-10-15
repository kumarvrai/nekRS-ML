#ifdef ENABLE_SMARTREDIS

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "smartRedis.hpp"
#include "client.h"
#include <string>
#include <vector>

smartredis_data *sr = new smartredis_data;
SmartRedis::Client *client_ptr = nullptr;

// Initialize the SmartRedis client and the smartredis struct
void smartredis::init_client(nrs_t *nrs)
{
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;

  if (client_ptr)
    return;

  sr->npts_per_tensor = nrs->fieldOffset;
  sr->num_tot_tensors = size;
  sr->num_db_tensors = size;
  sr->head_rank = 0;
  sr->db_nodes = 1;

  if (rank == 0)
    printf("\nInitializing client ...\n");
  bool cluster_mode = (sr->db_nodes > 1);
  std::string logger_name("Client");
  client_ptr = new SmartRedis::Client(cluster_mode, logger_name);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n");
}

void smartredis::del_client(nrs_t *nrs)
{
  int rank = platform->comm.mpiRank;
  if (rank == 0)
    printf("\nDeleting client ...\n");
  delete client_ptr;
  client_ptr = nullptr;
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n");
}

// Check value of check-run variable to know when to quit
int smartredis::check_run()
{
  int rank = platform->comm.mpiRank;
  int exit_val = 1;
  std::string run_key = "check-run";
  int check_run[2] = {1, 1};

  if (rank == 0) {
    client_ptr->unpack_tensor(run_key, check_run, {2}, SRTensorTypeInt32, SRMemLayoutContiguous);
    exit_val = check_run[0];
  }
  MPI_Bcast(&exit_val, 1, MPI_INT, 0, platform->comm.mpiComm);
  if (exit_val == 0 && rank == 0)
    printf("\nML training says time to quit ...\n");
  return exit_val;
}

// Initialize the check-run tensor on DB
void smartredis::init_check_run()
{
  int rank = platform->comm.mpiRank;
  std::vector<int> check_run(2, 1);
  std::string run_key = "check-run";

  if (rank == 0)
    client_ptr->put_tensor(run_key, check_run.data(), {2}, SRTensorTypeInt32, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Put check-run in DB\n\n");
}

// Put step number in DB
void smartredis::put_step_num(int tstep)
{
  std::string key = "step";
  std::vector<double> step_num(1, tstep);
  if (platform->comm.mpiRank == 0)
    client_ptr->put_tensor(key, step_num.data(), {1}, SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
}

// --------------------------------------------------- //
// ----- Wall-Shear Stress Model --------------------- //
// --------------------------------------------------- //
// Metadata per rank, same tensorInfo.<rank> layout as SOD2D
void smartredis::init_wallModel(int npts, int num_inputs, int num_outputs)
{
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;
  std::vector<int> tensor_info = {npts, size, size, 0, num_inputs, num_outputs};
  std::string info_key = "tensorInfo." + std::to_string(rank);
  client_ptr->put_tensor(info_key, tensor_info.data(), {6}, SRTensorTypeInt32, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("wallModel: sent tensorInfo (n_inputs %d, n_outputs %d)\n", num_inputs, num_outputs);
}

// Static wall data (written once) for the witness sink
void smartredis::put_wallModel_static(int npts, int ncols, const double *data, const std::vector<double> &meta)
{
  int rank = platform->comm.mpiRank;
  if (npts > 0) {
    std::string key = "wm_static." + std::to_string(rank);
    client_ptr->put_tensor(key, (void *)data, {(size_t)npts, (size_t)ncols}, SRTensorTypeDouble,
                           SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  // wm_meta last: once it exists, every wm_static.<rank> is in the DB
  if (rank == 0)
    client_ptr->put_tensor("wm_meta", (void *)meta.data(), {meta.size()}, SRTensorTypeDouble,
                           SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
}

// One training sample: inputs then outputs per wall node
void smartredis::put_wallModel_sample(int isample, int tstep, double time, int npts, int ncols, const double *data)
{
  int rank = platform->comm.mpiRank;
  if (npts > 0) {
    std::string key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
    client_ptr->put_tensor(key, (void *)data, {(size_t)npts, (size_t)ncols}, SRTensorTypeDouble,
                           SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  // Sample index last: once wm_sample.<i> exists, all x.<rank>.<step> are in the DB
  if (rank == 0) {
    std::vector<double> st = {(double)tstep, time};
    client_ptr->put_tensor("wm_sample." + std::to_string(isample), st.data(), {2}, SRTensorTypeDouble,
                           SRMemLayoutContiguous);
    std::vector<double> step_num(1, tstep);
    client_ptr->put_tensor("step", step_num.data(), {1}, SRTensorTypeDouble, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("wallModel: sent sample %d (step %d, time %g)\n", isample, tstep, time);
}

void smartredis::put_wallModel_done(int nsamples)
{
  if (platform->comm.mpiRank == 0) {
    std::vector<double> n(1, nsamples);
    client_ptr->put_tensor("wm_done", n.data(), {1}, SRTensorTypeDouble, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
}

// Inference. Only ranks that own wall points talk to the DB, each with its
// own keys, so there is no MPI collective in here (a barrier inside the
// npts > 0 branch would deadlock, as in SOD2D before its fix). Keys carry
// the step and are deleted after use so the DB does not grow.
void smartredis::run_wallModel(int tstep, int npts, int num_inputs, const double *inputs, int num_outputs,
                               double *outputs)
{
  if (npts == 0)
    return;
  int rank = platform->comm.mpiRank;
  std::string in_key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
  std::string out_key = "y." + std::to_string(rank) + "." + std::to_string(tstep);

  client_ptr->put_tensor(in_key, (void *)inputs, {(size_t)npts, (size_t)num_inputs}, SRTensorTypeDouble,
                         SRMemLayoutContiguous);
  client_ptr->run_model("model", {in_key}, {out_key});
  client_ptr->unpack_tensor(out_key, outputs, {(size_t)npts, (size_t)num_outputs}, SRTensorTypeDouble,
                            SRMemLayoutContiguous);
  client_ptr->delete_tensor(in_key);
  client_ptr->delete_tensor(out_key);
}

// --------------------------------------------------- //
// ----- Pressure Model ------------------------------ //
// --------------------------------------------------- //
// Initialize training for the velocity-pressure model
void smartredis::init_velNpres_train(nrs_t *nrs)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;

  if (rank == 0) 
    printf("\nSending training metadata for velocity-pressure model ...\n");

  // Create and send metadata for training
  std::vector<int> tensor_info(6,0);
  tensor_info[0] = sr->npts_per_tensor;
  tensor_info[1] = sr->num_tot_tensors;
  tensor_info[2] = sr->num_db_tensors;
  tensor_info[3] = sr->head_rank;
  tensor_info[4] = 3; // number of model inputs
  tensor_info[5] = 1; // number of model outputs
  std::string info_key = "tensorInfo";
  if (rank%sr->num_db_tensors == 0) {
    client_ptr->put_tensor(info_key, tensor_info.data(), {6},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);

  if (rank == 0)
    printf("Done\n\n");
}

// Put velocity and pressure data in DB
void smartredis::put_velNpres_data(nrs_t *nrs, int tstep)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  std::string key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
  dfloat *train_data = new dfloat[nrs->fieldOffset * 4]();
  int size_U = nrs->fieldOffset * 3;
  int size_P = nrs->fieldOffset;

  // Concatenate velocity (inputs) and pressure (output)
  std::copy(nrs->U,nrs->U+size_U,train_data);
  std::copy(nrs->P,nrs->P+size_P,train_data+size_U);

  // Send training data to DB
  if (rank == 0)
    printf("\nSending field with key %s \n",key.c_str());
  client_ptr->put_tensor(key, train_data, {nrs->fieldOffset,4},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
}

// Run a ML model for inference
void smartredis::run_pressure_model(nrs_t *nrs, int tstep)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  int npts = nrs->fieldOffset;
  std::string in_key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
  std::string out_key = "y." + std::to_string(rank) + "." + std::to_string(tstep);
  dfloat *outputs = new dfloat[npts]();
  
  // Send input data
  if (rank == 0)
    printf("\nSending field with key %s \n",in_key.c_str());
  client_ptr->put_tensor(in_key, nrs->U, {npts,3},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");

  // Run ML model on input data
  if (rank == 0)
    printf("\nRunning ML model ...\n");
  client_ptr->run_model("model", {in_key}, {out_key});
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");

  // Retrieve model pedictions
  if (rank == 0)
    printf("\nRetrieving field with key %s \n",out_key.c_str());
  client_ptr->unpack_tensor(out_key, outputs, {npts},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
  
  // Compute error in prediction
  double error = 0.0;
  for (int n=0; n<npts; n++) {
    error = error + (outputs[n] - nrs->P[n])*(outputs[n] - nrs->P[n]);
    //printf("True, Pred, Error: %f, %f, %f \n",nrs->P[n],outputs[n],error);
  }
  error = error / npts;
  printf("[%d]: Mean Squared Error in pressure field = %f\n\n",rank,error);
  fflush(stdout);
}

#endif
