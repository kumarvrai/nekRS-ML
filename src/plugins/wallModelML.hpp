#if !defined(nekrs_wallmodelml_hpp_)
#define nekrs_wallmodelml_hpp_

// ML wall model: witness (off-wall) sampling, training data and wall-shear BC.
//
// The data layout is the SOD2D one (sod2d_smart, branch
// 203-ml-for-wall-models), so the same trainer, analysis scripts and
// exported model work with both codes. Per wall node and per off-wall point
// k = 1..K, placed at y_k = (k/K) * wm_witness_dist along the inward normal,
// there are seven raw input columns
//   [u_1 u_2 u_3 | dp/dx_1 dp/dx_2 dp/dx_3 | u_tau^Reichardt(|u_t|, y_k, nu)]
// (point k occupies columns 7(k-1) .. 7k-1) and three output columns, the
// tangential wall traction tau_w = (2 mu_eff S . n_in), normal part removed.
// Vectors are in global axes, or in the wall frame (t1, n, t2) when
// wm_project_normal is set; t1 = global x projected on the wall (global z
// where |n_x| > 0.9), n into the fluid, t2 = t1 x n.
//
// Inputs and target come from the same field: the instantaneous u/p, or the
// "walave" exponential filter dt/(dt+period_walave) as in SOD2D.
//
// .par section (keys as in the SOD2D JSON where they exist):
//   [WALLMODEL]
//   wm_mode               = train | infer | none
//   wm_boundary_ids       = 1            (comma separated bIDs)
//   wm_witness_dist       = 0.1
//   wm_num_offwall_points = 1
//   wm_sample_field       = instantaneous | walave
//   period_walave         = 1.0
//   wm_project_normal     = false
//   wm_leapwitsave        = 10           (training: steps between samples)
//   wm_infer_stride       = 1            (inference: steps between updates)
//   wm_inference_backend  = smartredis | reichardt
//   wm_infer_dtype        = float32 | float64  (tensors sent to the model; float32
//                           as SOD2D, so SOD2D-exported models run unchanged)
//   Training data outputs (either or both):
//   wm_smartredis_transfer  = true        stream samples to the SmartSim DB
//   wm_witness_save         = false       write wm_witness.h5 directly (parallel
//                                         HDF5, -DENABLE_HDF5=ON; witnessH5 plugin)
//   wm_witness_h5_file_name = wm_witness.h5
//   wm_continue_witness     = false       append to an existing file on restart
//   wm_witness_flush        = 10          samples buffered between writes
//   wm_witness_dtype        = float32 | float64
//
// The wall-shear BC is written to nrs->o_usrwrk slots 0..2 as the traction
// vector on the fluid in global axes; the .oudf projects it on its face
// tangents (bc->tr1 = usrwrk0*t1x + ..., see examples/turbChannel_mlwm).

#include "nrs.hpp"
#include <string>
#include <vector>

namespace wallModelML
{
constexpr int contractVersion = 1;
constexpr int inputsPerPoint = 7;
constexpr int colVel = 0;
constexpr int colGradP = 3;
constexpr int colUtauR = 6;
constexpr int numOutputs = 3;
constexpr int numInferOutputs = 1;
// Static per-wall-node columns: wall xyz, n, t1, t2, then y_k (K columns)
constexpr int numStaticFixedCols = 12;

struct config_t {
  std::string mode = "none";
  std::vector<int> bIDs = {1};
  dfloat witnessDist = 0.1;
  int numOffwallPoints = 1;
  std::string sampleField = "instantaneous";
  dfloat periodWalave = 1.0;
  bool projectNormal = false;
  int leapWitSave = 10;
  int inferStride = 1;
  std::string inferenceBackend = "smartredis";
  std::string inferDtype = "float32";
  bool smartredisTransfer = true;
  bool witnessSave = false;
  std::string witnessFile = "wm_witness.h5";
  bool continueWitness = false;
  int witnessFlush = 10;
  std::string witnessDtype = "float32";
};

void setup(nrs_t *nrs);
// Call at the end of every UDF_ExecuteStep: updates the walave filter,
// samples/sends training data or runs inference and updates the BC.
void executeStep(nrs_t *nrs, dfloat time, int tstep);
// Call once after the last step (tells the witness sink the run is over).
void finalize(nrs_t *nrs);

// Building blocks used by executeStep, exposed for custom udf workflows.
void updateWalave(nrs_t *nrs);
void sampleInputs(nrs_t *nrs);
void sampleTarget(nrs_t *nrs);
void applyWallShear(nrs_t *nrs, const std::vector<dfloat> &utau);

dfloat reichardtUtau(dfloat Us, dfloat y, dfloat nu);

const config_t &config();
int numPoints();                // local wall nodes (rows)
int numInputs();                // 7 * K
dfloat nu();
const std::vector<dfloat> &inputs();   // numPoints x numInputs, row major
const std::vector<dfloat> &outputs();  // numPoints x numOutputs, row major
const std::vector<dfloat> &staticData(); // numPoints x numStaticCols
const std::vector<dlong> &wallVolumeIndex(); // volume node index of each row
int numStaticCols();
} // namespace wallModelML

#endif
