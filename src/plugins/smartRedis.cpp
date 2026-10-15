#ifdef ENABLE_SMARTREDIS

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "smartRedis.hpp"
#include "client.h"
#include <string>
#include <vector>

smartredis_data *sr = new smartredis_data;
wallModel_data *wm = new wallModel_data;
SmartRedis::Client *client_ptr;

// Initialize the SmartRedis client and the smartredis struct
void smartredis::init_client(nrs_t *nrs)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;

  // Replace this with variable in .par file
  sr->npts_per_tensor = nrs->fieldOffset;
  sr->num_tot_tensors = size;
  sr->num_db_tensors = size;
  sr->head_rank = 0;
  sr->db_nodes = 1;

  // Initialize SR client
  if (rank == 0)
    printf("\nInitializing client ...\n");
  bool cluster_mode;
  if (sr->db_nodes > 1)
    cluster_mode = true;
  else
    cluster_mode = false;
  std::string logger_name("Client");
  client_ptr = new SmartRedis::Client(cluster_mode, logger_name); // allocates on heap
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n");
}

void smartredis::del_client(nrs_t *nrs)
{

  // Initialize local variables
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;

  // Delete SR client
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
  int exit_val;
  std::string run_key = "check-run";
  int *check_run = new int[2]();

  // Check value of check-run tensor in DB from head rank
  if (rank%sr->num_db_tensors == 0) {
    client_ptr->unpack_tensor(run_key, check_run, {2},
                       SRTensorTypeInt32, SRMemLayoutContiguous);
    exit_val = check_run[0];
  }

  // Broadcast exit value and return it
  MPI_Bcast(&exit_val, 1, MPI_INT, 0, platform->comm.mpiComm);
  if (exit_val==0 && platform->comm.mpiRank==0) {
    printf("\nML training says time to quit ...\n");
  }
  return exit_val;
}

// Initialize the check-run tensor on DB
void smartredis::init_check_run()
{
  int rank = platform->comm.mpiRank;
  std::vector<int> check_run(2,0);
  check_run[0] = 1; check_run[1] = 1;
  std::string run_key = "check-run";

  if (rank%sr->num_db_tensors == 0) {
    client_ptr->put_tensor(run_key, check_run.data(), {2},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Put check-run in DB\n\n");
}

// Put step number in DB
void smartredis::put_step_num(int tstep)
{
  // Initialize local variables
  int rank = platform->comm.mpiRank;
  std::string key = "step";
  std::vector<double> step_num(1,0);
  step_num[0] = tstep;

  // Send time step to DB
  if (rank == 0)
    printf("\nSending time step number ...\n");
  client_ptr->put_tensor(key, step_num.data(), {1},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (rank == 0)
    printf("Done\n\n");
}

// --------------------------------------------------- //
// ----- Wall-Shear Stress Model --------------------- //
// --------------------------------------------------- //
// Initialize training for the wall shear stress model
void smartredis::init_wallModel_train(nrs_t *nrs, int Npart)
{
  int rank = platform->comm.mpiRank;
  int size = platform->comm.mpiCommSize;
  std::vector<int> tensor_info(6,0);
  auto mesh = nrs->meshV;

  if (rank == 0) 
    printf("\nSending training metadata for wall shear stress model ...\n");

  int wall_node_ct = Npart;
  int owall_node_ct = Npart;
  printf("[Rank%d]: Found %d wall nodes and %d off-wall nodes\n",rank,wall_node_ct, owall_node_ct);
  assert(wall_node_ct == owall_node_ct);
  sr->npts_per_tensor = wall_node_ct;
  wm->num_samples = wall_node_ct;

  // Create and send metadata for training
  tensor_info[0] = sr->npts_per_tensor;
  tensor_info[1] = sr->num_tot_tensors;
  tensor_info[2] = sr->num_db_tensors;
  tensor_info[3] = sr->head_rank;
  tensor_info[4] = wm->num_inputs;
  tensor_info[5] = wm->num_outputs;
  std::string info_key = "tensorInfo."+ std::to_string(rank);
  //if (rank%sr->num_db_tensors == 0) {
  client_ptr->put_tensor(info_key, tensor_info.data(), {6},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  //}
  MPI_Barrier(platform->comm.mpiComm);

  if (rank == 0)
    printf("Done\n\n");
}
//// Initialize training for the wall shear stress model
//void smartredis::init_wallModel_train(nrs_t *nrs)
//{
//  int rank = platform->comm.mpiRank;
//  int size = platform->comm.mpiCommSize;
//  std::vector<int> tensor_info(6,0);
//  auto mesh = nrs->meshV;
//
//  if (rank == 0) 
//    printf("\nSending training metadata for wall shear stress model ...\n");
//
//  // Loop over mesh coordinates to find wall and off-wall node indices
//  std::vector<int> ind_owall_nodes_raw;
//  for (int i=0; i<mesh->Nlocal; i++) {
//    const auto y = abs(mesh->y[i]);
//    if (y >= wm->wall_height-wm->eps && y <= wm->wall_height+wm->eps) {
//      wm->ind_wall_nodes.push_back(i);
//    }
//    if (y >= wm->off_wall_height-wm->eps && y <= wm->off_wall_height+wm->eps) {
//      ind_owall_nodes_raw.push_back(i);
//    }
//  }
//  int wall_node_ct = wm->ind_wall_nodes.size();
//  int owall_node_ct = ind_owall_nodes_raw.size();
//  printf("Found %d wall nodes and %d off-wall nodes\n",wall_node_ct, owall_node_ct);
//  assert(wall_node_ct == owall_node_ct);
//  sr->npts_per_tensor = wall_node_ct;
//  wm->num_samples = wall_node_ct;
//
//  // Pair the wall nodes with the off-wall nodes based on the x and z coordinate
//  // to pair the inputs and outputs of the model
//  std::vector<int> ind_owall_nodes_tmp(wall_node_ct,0);
//  ind_owall_nodes_tmp = ind_owall_nodes_raw;
//  int pairs = 0;
//  for (int i=0; i<wall_node_ct; i++) {
//    int ind_w = wm->ind_wall_nodes[i];
//    const auto x_wall = mesh->x[ind_w];
//    const auto z_wall = mesh->z[ind_w];
//    for (int j=0; j<ind_owall_nodes_tmp.size(); j++) {
//      int ind_ow = ind_owall_nodes_tmp[j];
//      const auto x_owall = mesh->x[ind_ow];
//      const auto z_owall = mesh->z[ind_ow];
//      if ((x_owall>=x_wall-wm->eps && x_owall<=x_wall+wm->eps) && 
//          (z_owall>=z_wall-wm->eps && z_owall<=z_wall+wm->eps)) {
//        wm->ind_owall_nodes_matched.push_back(ind_ow);
//        ind_owall_nodes_tmp.erase(ind_owall_nodes_tmp.begin() + j);
//        pairs++;
//      }
//    }
//  }
//  printf("Found %d paris of wall nodes and off-wall nodes\n",pairs);
//
//  // Create and send metadata for training
//  tensor_info[0] = sr->npts_per_tensor;
//  tensor_info[1] = sr->num_tot_tensors;
//  tensor_info[2] = sr->num_db_tensors;
//  tensor_info[3] = sr->head_rank;
//  tensor_info[4] = wm->num_inputs;
//  tensor_info[5] = wm->num_outputs;
//  std::string info_key = "tensorInfo";
//  if (rank%sr->num_db_tensors == 0) {
//    client_ptr->put_tensor(info_key, tensor_info.data(), {6},
//                    SRTensorTypeInt32, SRMemLayoutContiguous);
//  }
//  MPI_Barrier(platform->comm.mpiComm);
//
//  if (rank == 0)
//    printf("Done\n\n");
//}
//

// Put training data for wall shear stress model in DB -- channel
void smartredis::put_wallModel_data_channel(nrs_t *nrs, 
                                            int tstep, 
                                            int Npart, 
                                            int *BdryToV, 
                                            double *Upart,
                                            double *Ppart,
                                            double *Rpart,
                                            double *Norml,
                                            double *Tang1,
                                            double *Tang2)
{
  int rank = platform->comm.mpiRank;
  int num_samples = wm->num_samples;
  int num_cols = wm->num_inputs+wm->num_outputs;
  dfloat avg_utau, avg_ut;
  dfloat xx, yy, zz;  
  dfloat ux, uy, uz;  
  dfloat px, py, pz;  
  dfloat nx, ny, nz;  
  dfloat t1x, t1y, t1z;
  dfloat t2x, t2y, t2z;
  dfloat umag, ymag, mue, utau;
  dfloat uxt, uyn, uzt;
  dfloat dpdxt, dpdyn, dpdz;
  std::string key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
  //std::string key = "x." + std::to_string(rank);
  dfloat *train_data = new dfloat[num_samples * num_cols]();
  dfloat *inputs = new dfloat[num_samples * wm->num_inputs]();
  dfloat *outputs = new dfloat[num_samples * wm->num_outputs]();

  assert(num_samples == Npart);
  platform->options.getArgs("VISCOSITY",mue);

  // Extract velocity at off-wall nodes (inputs)
  // Extract shear at wall nodes (outputs)
  avg_utau = 0.0;
  avg_ut   = 0.0;
  for (int n = 0; n < Npart; ++n) {
    const int v = BdryToV[n];
    xx  = Rpart[0*Npart + n];
    yy  = Rpart[1*Npart + n];
    zz  = Rpart[2*Npart + n];
    ux  = Upart[0*Npart + n];
    uy  = Upart[1*Npart + n];
    uz  = Upart[2*Npart + n];
    px  = Ppart[0*Npart + n];
    py  = Ppart[1*Npart + n];
    pz  = Ppart[2*Npart + n];
    nx  = -Norml[0*Npart + n];
    ny  = -Norml[1*Npart + n];
    nz  = -Norml[2*Npart + n];
    t1x = Tang1[0*Npart + n];
    t1y = Tang1[1*Npart + n];
    t1z = Tang1[2*Npart + n];
    t2x = Tang2[0*Npart + n];
    t2y = Tang2[1*Npart + n];
    t2z = Tang2[2*Npart + n];
    utau = nrs->usrwrk[v+0*nrs->fieldOffset];

    //derived quantities
    ymag  = sqrt(xx*xx+yy*yy+zz*zz);
    umag  = sqrt(ux*ux+uy*uy+uz*uz);
    uxt   = ux*t1x+uy*t1y+uz*t1z;
    uyn   = ux*nx+uy*ny+uz*nz;
    uzt   = ux*t2x+uy*t2y+uz*t2z;
    dpdxt = px*t1x+py*t1y+pz*t1z;
    dpdyn = px*nx+py*ny+pz*nz;
    dpdz  = px*t2x+py*t2y+pz*t2z;

    //Radhakrishnan et al (2021,2024)
    //inputs and outputs for offline training
    //inputs[n*wm->num_inputs+0]   = uxt/(Smag*ymag);  // input
    inputs[n*wm->num_inputs+0]   = uxt*ymag/mue;     // input
    outputs[n*wm->num_outputs+0] = utau;    //output

    avg_ut += uxt;
    avg_utau += utau;
  }
  printf("Train[%d]: utau_avg usmpl_avg %E %E \n",rank,sqrt(abs(avg_utau/Npart)),avg_ut/Npart);

  // Concatenate inputs and outputs
  for (int i=0; i<num_samples; i++) {
    //Inputs
    for (int j=0; j<wm->num_inputs; j++) {
      train_data[i*num_cols+j] = inputs[i*wm->num_inputs + j];
    }
    //Outputs
    for (int j=0; j<wm->num_outputs; j++) {
      train_data[i*num_cols+wm->num_inputs+j] = outputs[i*wm->num_outputs + j];
    }
  }

  // Send training data to DB
  if (rank == 0)
    printf("Train: Sending field with key %s \n",key.c_str());
  client_ptr->put_tensor(key, train_data, {num_samples,num_cols},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);

  delete[] train_data;
  delete[] inputs;
  delete[] outputs;

  if (rank == 0)
    printf("Done\n");
}

// Put training data for wall shear stress model in DB
void smartredis::put_wallModel_data(nrs_t *nrs, 
                                    int tstep, 
                                    int Npart, 
                                    int *BdryToV, 
                                    double *Upart,
                                    double *Ppart,
                                    double *Rpart,
                                    double *Norml,
                                    double *Tang1,
                                    double *Tang2)
{
  int rank = platform->comm.mpiRank;
  int num_samples = wm->num_samples;
  int num_cols = wm->num_inputs+wm->num_outputs;
  dfloat avg_utau, avg_ut;
  dfloat xx, yy, zz;  
  dfloat ux, uy, uz;  
  dfloat px, py, pz;  
  dfloat nx, ny, nz;  
  dfloat t1x, t1y, t1z;
  dfloat t2x, t2y, t2z;
  dfloat umag, ymag, mue, utau;
  dfloat uxt, uyn, uzt;
  dfloat dpdxt, dpdyn, dpdz;
  std::string key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
  //std::string key = "x." + std::to_string(rank);
  dfloat *train_data = new dfloat[num_samples * num_cols]();
  dfloat *inputs = new dfloat[num_samples * wm->num_inputs]();
  dfloat *outputs = new dfloat[num_samples * wm->num_outputs]();

  assert(num_samples == Npart);
  platform->options.getArgs("VISCOSITY",mue);

  // Extract velocity at off-wall nodes (inputs)
  // Extract shear at wall nodes (outputs)
  avg_utau = 0.0;
  avg_ut   = 0.0;
  for (int n = 0; n < Npart; ++n) {
    const int v = BdryToV[n];
    xx  = Rpart[0*Npart + n];
    yy  = Rpart[1*Npart + n];
    zz  = Rpart[2*Npart + n];
    ux  = Upart[0*Npart + n];
    uy  = Upart[1*Npart + n];
    uz  = Upart[2*Npart + n];
    px  = Ppart[0*Npart + n];
    py  = Ppart[1*Npart + n];
    pz  = Ppart[2*Npart + n];
    nx  = -Norml[0*Npart + n];
    ny  = -Norml[1*Npart + n];
    nz  = -Norml[2*Npart + n];
    t1x = Tang1[0*Npart + n];
    t1y = Tang1[1*Npart + n];
    t1z = Tang1[2*Npart + n];
    t2x = Tang2[0*Npart + n];
    t2y = Tang2[1*Npart + n];
    t2z = Tang2[2*Npart + n];
    utau = nrs->usrwrk[v+0*nrs->fieldOffset];

    //derived quantities
    ymag  = sqrt(xx*xx+yy*yy+zz*zz);
    umag  = sqrt(ux*ux+uy*uy+uz*uz);
    uxt   = ux*t1x+uy*t1y+uz*t1z;
    uyn   = ux*nx+uy*ny+uz*nz;
    uzt   = ux*t2x+uy*t2y+uz*t2z;
    dpdxt = px*t1x+py*t1y+pz*t1z;
    dpdyn = px*nx+py*ny+pz*nz;
    dpdz  = px*t2x+py*t2y+pz*t2z;

    //inputs and outputs for offline training
    inputs[n*wm->num_inputs+0]   = ymag;     // input
    inputs[n*wm->num_inputs+1]   = uxt;     // input
    inputs[n*wm->num_inputs+2]   = uyn;     // input
    inputs[n*wm->num_inputs+3]   = uzt;     // input
    inputs[n*wm->num_inputs+4]   = dpdxt;   // input
    inputs[n*wm->num_inputs+5]   = dpdyn;   // input
    inputs[n*wm->num_inputs+6]   = dpdz;   // input
    outputs[n*wm->num_outputs+0] = utau;   //output

    ////inputs and outputs for online training
    ////inputs[n*wm->num_inputs+0]   = utau*ymag/mue;     // input
    //inputs[n*wm->num_inputs+0]   = umag*ymag/mue;     // input
    //inputs[n*wm->num_inputs+1]   = umag/cbrt(fabs(mue*dpdxt) + 1e-12);   // input
    //outputs[n*wm->num_outputs+0] = utau;   //output

    avg_ut += uxt;
    avg_utau += utau;
  }
  printf("Train[%d]: utau_avg usmpl_avg %E %E \n",rank,sqrt(abs(avg_utau/Npart)),avg_ut/Npart);

  // Concatenate inputs and outputs
  for (int i=0; i<num_samples; i++) {
    //Inputs
    for (int j=0; j<wm->num_inputs; j++) {
      train_data[i*num_cols+j] = inputs[i*wm->num_inputs + j];
    }
    //Outputs
    for (int j=0; j<wm->num_outputs; j++) {
      train_data[i*num_cols+wm->num_inputs+j] = outputs[i*wm->num_outputs + j];
    }
  }

  // Send training data to DB
  if (rank == 0)
    printf("Train: Sending field with key %s \n",key.c_str());
  client_ptr->put_tensor(key, train_data, {num_samples,num_cols},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);

  delete[] train_data;
  delete[] inputs;
  delete[] outputs;

  if (rank == 0)
    printf("Done\n");
}

//// Put training data for wall shear stress model in DB
//void smartredis::put_wallModel_data(nrs_t *nrs, int tstep)
//{
//  int rank = platform->comm.mpiRank;
//  int num_samples = wm->num_samples;
//  int num_cols = wm->num_inputs+wm->num_outputs;
//  dfloat mue;
//  std::string key = "x." + std::to_string(rank) + "." + std::to_string(tstep);
//  dfloat *train_data = new dfloat[num_samples*num_cols]();
//  dfloat *inputs = new dfloat[num_samples]();
//  dfloat *outputs = new dfloat[num_samples]();
//
//  // Extract velocity at off-wall nodes (inputs)
//  for (int i=0; i<num_samples; i++) {
//    int ind = wm->ind_owall_nodes_matched[i];
//    inputs[i] = nrs->U[ind+0*nrs->fieldOffset];
//  }
//
//  // Extract strain rate at wall and multiply by viscosity to obtain stress
//  platform->options.getArgs("VISCOSITY",mue);
//  for (int i=0; i<num_samples; i++) {
//    int ind = wm->ind_wall_nodes[i];
//    //outputs[i] = mue * nrs->cds->S[ind+0*nrs->fieldOffset];
//    //outputs[i] = 0.055;
//    outputs[i] = nrs->usrwrk[ind+0*nrs->fieldOffset];
//  }
//
//  // Concatenate inputs and outputs
//  for (int i=0; i<num_samples; i++) {
//    train_data[i*num_cols] = inputs[i];
//    train_data[i*num_cols+1] = outputs[i];
//  }
//
//  // Send training data to DB
//  if (rank == 0)
//    printf("\nSending field with key %s \n",key.c_str());
//  client_ptr->put_tensor(key, train_data, {num_samples,num_cols},
//                    SRTensorTypeDouble, SRMemLayoutContiguous);
//  MPI_Barrier(platform->comm.mpiComm);
//  if (rank == 0)
//    printf("Done\n\n");
//}


// Run ML model for inference
void smartredis::run_wallModel(nrs_t *nrs, 
                               int tstep, 
                               int Npart, 
                               int *BdryToV, 
                               double *Upart,
                               double *Ppart,
                               double *Rpart,
                               double *Norml,
                               double *Tang1,
                               double *Tang2)
{
  int rank = platform->comm.mpiRank;
  int num_samples = wm->num_samples;
  dfloat xx, yy, zz;  
  dfloat ux, uy, uz;  
  dfloat px, py, pz;  
  dfloat nx, ny, nz;  
  dfloat t1x, t1y, t1z;
  dfloat t2x, t2y, t2z;
  dfloat umag, ymag, mue, utau;
  dfloat uxt, uyn, uzt, dpdxt;
  std::string in_key = "x." + std::to_string(rank);
  std::string out_key = "y." + std::to_string(rank);
  dfloat *inputs = new dfloat[num_samples * wm->num_inputs]();
  dfloat *outputs = new dfloat[num_samples * wm->num_outputs]();
  //dfloat *targets = new dfloat[num_samples * wm->num_outputs]();

  assert(num_samples == Npart);
  platform->options.getArgs("VISCOSITY",mue);

  // Extract velocity at off-wall nodes (inputs)
  for (int n = 0; n < Npart; ++n) {
    const int v = BdryToV[n];
    xx  = Rpart[0*Npart + n];
    yy  = Rpart[1*Npart + n];
    zz  = Rpart[2*Npart + n];
    ux  = Upart[0*Npart + n];
    uy  = Upart[1*Npart + n];
    uz  = Upart[2*Npart + n];
    px  = Ppart[0*Npart + n];
    py  = Ppart[1*Npart + n];
    pz  = Ppart[2*Npart + n];
    nx  = -Norml[0*Npart + n];
    ny  = -Norml[1*Npart + n];
    nz  = -Norml[2*Npart + n];
    t1x = Tang1[0*Npart + n];
    t1y = Tang1[1*Npart + n];
    t1z = Tang1[2*Npart + n];
    t2x = Tang2[0*Npart + n];
    t2y = Tang2[1*Npart + n];
    t2z = Tang2[2*Npart + n];

    //derived quantities
    ymag = sqrt(xx*xx+yy*yy+zz*zz);
    umag = sqrt(ux*ux+uy*uy+uz*uz);
    uxt = ux*t1x+uy*t1y+uz*t1z;
    uyn = ux*nx+uy*ny+uz*nz;
    uzt = ux*t2x+uy*t2y+uz*t2z;
    dpdxt = px*t1x+py*t1y+pz*t1z;

    //inputs and outputs
    //inputs[n*wm->num_inputs+0] = utau*ymag/mue;     // input
    inputs[n*wm->num_inputs+0] = umag*ymag/mue;     // input
    inputs[n*wm->num_inputs+1] = umag/cbrt(fabs(mue*dpdxt) + 1e-12);     // input
  }

  //// Extract strain rate at wall and multiply by viscosity to obtain stress
  //for (int i=0; i<num_samples; i++) {
  //  int ind = wm->ind_wall_nodes[i];
  //  //targets[i] = mue * nrs->cds->S[ind+0*nrs->fieldOffset];
  //  targets[i] = 0.0507;
  //}
  
  // Send input data
  //if (rank == 0)
  //  printf("\nSending field with key %s \n",in_key.c_str());
  client_ptr->put_tensor(in_key, inputs, {num_samples,wm->num_inputs},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  //if (rank == 0)
  //  printf("Done\n\n");

  // Run ML model on input data
  //if (rank == 0)
  //  printf("\nInfer: Running ML model ...\n");
  client_ptr->run_model("model", {in_key}, {out_key});
  MPI_Barrier(platform->comm.mpiComm);
  //if (rank == 0)
  //  printf("Done\n\n");

  // Retrieve model pedictions
  //if (rank == 0)
  //  printf("Infer: Retrieving field with key %s \n",out_key.c_str());
  client_ptr->unpack_tensor(out_key, outputs, {num_samples,wm->num_outputs},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  //if (rank == 0)
  //  printf("Done\n");

  // Apply traction BC model
  for (int n = 0; n < Npart; ++n) {
    const int v = BdryToV[n];
    for (int j=0; j<wm->num_outputs; j++) {
      nrs->usrwrk[v + j*nrs->fieldOffset] = outputs[n*wm->num_outputs + j];
    }  
  }
  //if (rank == 0)
  //  printf("usrwrk array filled\n\n");  

  delete[] inputs;
  delete[] outputs;
  
  //// Compute error in prediction
  //double error = 0.0;
  //double avg_ut = 0.0;
  //double avg_utau = 0.0;
  //for (int i=0; i<num_samples; i++) {
  //  avg_ut += inputs[i]/Npart;
  //  avg_utau += outputs[i]/Npart;
  //  error += sqrt((outputs[i] - targets[i])*(outputs[i] - targets[i]))/Npart;
  //  //printf("True, Pred, Error: %f, %f, %f \n",nrs->P[n],outputs[n],error);
  //}
  //error = error / num_samples;
  //if (rank == 0)
  //  printf("ML-INFER: UAVG UTAVG MRSE= %E,%E,%E\n",avg_ut,sqrt(abs(avg_utau)),error);
  //  fflush(stdout);
}

//// Run ML model for inference
//void smartredis::run_wallModel(nrs_t *nrs, int tstep)
//{
//  int rank = platform->comm.mpiRank;
//  int num_samples = wm->num_samples;
//  dfloat mue;
//  std::string in_key = "x." + std::to_string(rank);
//  std::string out_key = "y." + std::to_string(rank);
//  dfloat *outputs = new dfloat[num_samples]();
//  dfloat *inputs = new dfloat[num_samples]();
//  dfloat *targets = new dfloat[num_samples]();
//
//  // Extract velocity at off-wall nodes (inputs)
//  for (int i=0; i<num_samples; i++) {
//    int ind = wm->ind_owall_nodes_matched[i];
//    inputs[i] = nrs->U[ind+0*nrs->fieldOffset];
//  }
//
//  // Extract strain rate at wall and multiply by viscosity to obtain stress
//  platform->options.getArgs("VISCOSITY",mue);
//  for (int i=0; i<num_samples; i++) {
//    int ind = wm->ind_wall_nodes[i];
//    //targets[i] = mue * nrs->cds->S[ind+0*nrs->fieldOffset];
//    targets[i] = 0.055;
//  }
//  
//  // Send input data
//  if (rank == 0)
//    printf("\nSending field with key %s \n",in_key.c_str());
//  client_ptr->put_tensor(in_key, inputs, {num_samples,1},
//                    SRTensorTypeDouble, SRMemLayoutContiguous);
//  MPI_Barrier(platform->comm.mpiComm);
//  if (rank == 0)
//    printf("Done\n\n");
//
//  // Run ML model on input data
//  if (rank == 0)
//    printf("\nRunning ML model ...\n");
//  client_ptr->run_model("model", {in_key}, {out_key});
//  MPI_Barrier(platform->comm.mpiComm);
//  if (rank == 0)
//    printf("Done\n\n");
//
//  // Retrieve model pedictions
//  if (rank == 0)
//    printf("\nRetrieving field with key %s \n",out_key.c_str());
//  client_ptr->unpack_tensor(out_key, outputs, {num_samples},
//                       SRTensorTypeDouble, SRMemLayoutContiguous);
//  MPI_Barrier(platform->comm.mpiComm);
//  if (rank == 0)
//    printf("Done\n\n");
//
//  // Vishal -- Apply traction BC model
//  for (int i=0; i<num_samples; i++) {
//    int ind = wm->ind_wall_nodes[i];
//    nrs->usrwrk[ind+0*nrs->fieldOffset] = outputs[i];
//  }
//  if (rank == 0)
//    printf("usrwrk array filled\n\n");  
//  
//  // Compute error in prediction
//  double error = 0.0;
//  for (int i=0; i<num_samples; i++) {
//    error = error + (outputs[i] - targets[i])*(outputs[i] - targets[i]);
//    //printf("True, Pred, Error: %f, %f, %f \n",nrs->P[n],outputs[n],error);
//  }
//  error = error / num_samples;
//  printf("[%d]: Mean Squared Error in wall shear stress field = %E\n\n",rank,error);
//  fflush(stdout);
//}

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
