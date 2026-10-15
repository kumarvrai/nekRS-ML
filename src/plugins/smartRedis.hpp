#if !defined(nekrs_smartredis_hpp_)
#define nekrs_smartredis_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include <vector>

struct smartredis_data {
  int npts_per_tensor; // number of points (samples) per tensor being sent to DB
  int num_tot_tensors; // number of total tensors being sent to all DB
  int num_db_tensors; // number of tensors being sent to each DB
  int db_nodes; // number of DB nodes (always 1 for co-located DB)
  int head_rank; // rank ID of the head rank on each node (metadata transfer with co-DB)
};

// Database keys of the wall-model data (same names as SOD2D mod_smartredis
// where SOD2D has them):
//   tensorInfo.<rank>  int32[6]  npts, nranks, nranks, head rank, n_inputs, n_outputs
//   wm_meta            double    contract version, nu, K, n_inputs, n_outputs,
//                                instantaneous(1)/walave(0), period_walave,
//                                project_normal, nranks, witness distance
//   wm_static.<rank>   double[npts, 12+K]  wall xyz, n, t1, t2, y_1..y_K
//   x.<rank>.<step>    double[npts, n_inputs+n_outputs]  training sample
//   wm_sample.<i>      double[2]  step, time of the i-th training sample
//   wm_done            double[1]  number of samples sent (end of run)
//   x/y.<rank>.<step>  inference input/output, deleted after use
// Ranks without wall points send no x.*, wm_static.* or y.* tensors.
namespace smartredis
{
  void init_client(nrs_t *nrs);
  void del_client(nrs_t *nrs);
  int check_run();
  void init_check_run();
  void put_step_num(int tstep);

  // Wall-shear stress model (data assembled by the wallModelML plugin)
  void init_wallModel(int npts, int num_inputs, int num_outputs);
  void put_wallModel_static(int npts, int ncols, const double *data, const std::vector<double> &meta);
  void put_wallModel_sample(int isample, int tstep, double time, int npts, int ncols, const double *data);
  void put_wallModel_done(int nsamples);
  void run_wallModel(int tstep, int npts, int num_inputs, const double *inputs, int num_outputs, double *outputs);

  // Velocity-pressure model
  void init_velNpres_train(nrs_t *nrs);
  void put_velNpres_data(nrs_t *nrs, int tstep);
  void run_pressure_model(nrs_t *nrs, int tstep);
}

#endif
