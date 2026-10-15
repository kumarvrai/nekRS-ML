#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "udf.hpp"
#include "sgsModel.hpp"

namespace
{
std::string modelName = "none";
dfloat vremanCs = 0.1;
dfloat mu0 = 0, rho0 = 1;
bool initialized = false;

occa::kernel vremanKernel;
occa::memory o_nutField;

// mu_eff = mu0 + rho0 * nu_t, rho = rho0
occa::kernel propertiesKernel;
} // namespace

namespace sgsModel
{
const std::string &model() { return modelName; }
occa::memory o_nut() { return o_nutField; }

void buildKernel(occa::properties kernelInfo)
{
  const std::string path = getenv("NEKRS_KERNEL_DIR") + std::string("/plugins/");
  vremanKernel = platform->device.buildKernel(path + "sgsVremanHex3D.okl", kernelInfo, true);
  propertiesKernel = platform->device.buildKernel(path + "sgsProperties.okl", kernelInfo, true);
}

void setup(nrs_t *nrs, bool installPropertiesCallback)
{
  platform->par->extract("sgsmodel", "model", modelName);
  platform->par->extract("sgsmodel", "vreman_cs", vremanCs);
  nrsCheck(modelName != "none" && modelName != "vreman", platform->comm.mpiComm, EXIT_FAILURE,
           "[SGSMODEL] unknown model %s (vreman | none)\n", modelName.c_str());

  platform->options.getArgs("VISCOSITY", mu0);
  platform->options.getArgs("DENSITY", rho0);

  o_nutField = platform->device.malloc(nrs->fieldOffset * sizeof(dfloat));
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_nutField);

  if (installPropertiesCallback)
    udf.properties = &updateProperties;
  initialized = true;

  if (platform->comm.mpiRank == 0) {
    printf("sgsModel: model=%s", modelName.c_str());
    if (modelName == "vreman")
      printf("  Cs=%g (c=2.5Cs^2=%g)", vremanCs, 2.5 * vremanCs * vremanCs);
    printf("\n");
  }
}

void computeNut(nrs_t *nrs, occa::memory o_U)
{
  mesh_t *mesh = nrs->meshV;
  if (modelName == "vreman") {
    const dfloat c = 2.5 * vremanCs * vremanCs;
    const dfloat dd = (2.0 / mesh->N) * (2.0 / mesh->N);
    vremanKernel(mesh->Nelements, nrs->fieldOffset, c, dd, mesh->o_vgeo, mesh->o_D, o_U, o_nutField);
  }
  else {
    platform->linAlg->fill(mesh->Nlocal, 0.0, o_nutField);
  }
}

void updateProperties(nrs_t *nrs, dfloat time, occa::memory o_U, occa::memory o_S,
                      occa::memory o_UProp, occa::memory o_SProp)
{
  nrsCheck(!initialized, platform->comm.mpiComm, EXIT_FAILURE, "%s\n",
           "sgsModel::updateProperties called before sgsModel::setup");
  mesh_t *mesh = nrs->meshV;
  computeNut(nrs, o_U);
  propertiesKernel(mesh->Nlocal, nrs->fieldOffset, mu0, rho0, o_nutField, o_UProp);
}
} // namespace sgsModel
