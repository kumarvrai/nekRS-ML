#if !defined(nekrs_sgsmodel_hpp_)
#define nekrs_sgsmodel_hpp_

// Eddy-viscosity subgrid-scale models for LES.
//
// The model sets the momentum viscosity through the properties callback:
//   mu_eff = mu + rho * nu_t,  rho = DENSITY (constant-property flows).
// nu_t is evaluated from the velocity handed to the callback, so it is
// consistent with the step being solved (no one-step lag).
//
// .par section:
//   [SGSMODEL]
//   model     = vreman | none
//   vreman_cs = 0.1    (Vreman constant c = 2.5 * Cs^2)
//
// Vreman (2004): nu_t = c * sqrt(B_beta / (alpha_ij alpha_ij)),
// alpha_ij = du_j/dx_i, beta_ij = Delta_m^2 alpha_mi alpha_mj, with the
// filter width taken per reference direction of the spectral element
// (Delta_m = (2/N) dx/dr_m), i.e. alpha_mi computed with d/dr_m scaled
// by 2/N. nu_t = 0 where B_beta <= 0 or alpha_ij alpha_ij = 0.
//
// Usage in the .udf:
//   #include "plugins/sgsModel.hpp"           (kernel is auto-loaded)
//   UDF_Setup:  sgsModel::setup(nrs);          (installs udf.properties)
//   optional:   sgsModel::o_nut() to output or average nu_t
// A udf that needs its own properties callback calls
// sgsModel::updateProperties from it instead of using the installed one.
// New models: add a kernel in kernels/plugins and a branch in sgsModel.cpp.

#include "nrs.hpp"
#include <string>

namespace sgsModel
{
void buildKernel(occa::properties kernelInfo);
void setup(nrs_t *nrs, bool installPropertiesCallback = true);
void computeNut(nrs_t *nrs, occa::memory o_U);
void updateProperties(nrs_t *nrs, dfloat time, occa::memory o_U, occa::memory o_S,
                      occa::memory o_UProp, occa::memory o_SProp);
occa::memory o_nut();
const std::string &model();
} // namespace sgsModel

#endif
