#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "pointInterpolation.hpp"
#include "wallModelML.hpp"
#ifdef ENABLE_SMARTREDIS
#include "smartRedis.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <sstream>

namespace
{
wallModelML::config_t cfg;
bool initialized = false;

int nWall = 0;   // local wall nodes
int nPts = 0;    // local witness points = nWall * K
int K = 1;
dfloat nuRef = 0;

std::vector<dlong> wallToV;       // volume index (element-local) of each wall node
std::vector<dfloat> wallXYZ;      // nWall x 3
std::vector<dfloat> wallN;        // nWall x 3, unit normal into the fluid
std::vector<dfloat> wallT1;       // nWall x 3
std::vector<dfloat> wallT2;       // nWall x 3
std::vector<dfloat> wallYs;       // nWall x K
std::vector<dfloat> witX, witY, witZ; // nPts, point p = n*K + k

std::vector<dfloat> inputData;    // nWall x 7K
std::vector<dfloat> outputData;   // nWall x 3
std::vector<dfloat> staticCols;   // nWall x (12 + K)

pointInterpolation_t *interp = nullptr;

occa::memory o_Uwal, o_Pwal;      // walave filter of u and p
occa::memory o_gradP;             // 3 x fieldOffset
occa::memory o_Sij;               // 6 x fieldOffset (element-local, not smoothed)
occa::memory o_witU, o_witGP;     // 3 x max(nPts,1)
bool walaveStarted = false;

int sampleIndex = 0;

std::vector<int> parseIntList(const std::string &s)
{
  std::vector<int> out;
  std::string item;
  std::stringstream ss(s);
  while (std::getline(ss, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
    if (!item.empty())
      out.push_back(std::stoi(item));
  }
  return out;
}

void readConfig()
{
  auto par = platform->par;
  const std::string sec = "wallmodel";
  par->extract(sec, "wm_mode", cfg.mode);
  std::string ids;
  if (par->extract(sec, "wm_boundary_ids", ids))
    cfg.bIDs = parseIntList(ids);
  par->extract(sec, "wm_witness_dist", cfg.witnessDist);
  par->extract(sec, "wm_num_offwall_points", cfg.numOffwallPoints);
  par->extract(sec, "wm_sample_field", cfg.sampleField);
  par->extract(sec, "period_walave", cfg.periodWalave);
  par->extract(sec, "wm_project_normal", cfg.projectNormal);
  par->extract(sec, "wm_leapwitsave", cfg.leapWitSave);
  par->extract(sec, "wm_infer_stride", cfg.inferStride);
  par->extract(sec, "wm_inference_backend", cfg.inferenceBackend);

  auto check = [](bool ok, const std::string &msg) {
    nrsCheck(!ok, platform->comm.mpiComm, EXIT_FAILURE, "[WALLMODEL] %s\n", msg.c_str());
  };
  check(cfg.mode == "train" || cfg.mode == "infer" || cfg.mode == "none",
        "wm_mode must be train, infer or none");
  check(cfg.sampleField == "instantaneous" || cfg.sampleField == "walave",
        "wm_sample_field must be instantaneous or walave");
  check(cfg.inferenceBackend == "smartredis" || cfg.inferenceBackend == "reichardt",
        "wm_inference_backend must be smartredis or reichardt");
  check(cfg.numOffwallPoints >= 1, "wm_num_offwall_points must be >= 1");
  check(cfg.witnessDist > 0, "wm_witness_dist must be > 0");
  check(cfg.leapWitSave >= 1 && cfg.inferStride >= 1, "wm_leapwitsave and wm_infer_stride must be >= 1");
  check(!cfg.bIDs.empty(), "wm_boundary_ids is empty");
#ifndef ENABLE_SMARTREDIS
  check(cfg.mode != "train", "wm_mode = train needs nekRS built with ENABLE_SMARTREDIS");
  check(!(cfg.mode == "infer" && cfg.inferenceBackend == "smartredis"),
        "wm_inference_backend = smartredis needs nekRS built with ENABLE_SMARTREDIS");
#endif
}

// Same frame as SOD2D wm_wall_basis: t1 = reference axis projected on the
// wall plane (x, or z where the wall is nearly normal to x), t2 = t1 x n.
// A fixed reference axis keeps the frame identical along a wall.
void wallBasis(const dfloat n[3], dfloat t1[3], dfloat t2[3])
{
  dfloat ref[3] = {1, 0, 0};
  if (std::abs(n[0]) >= 0.9) {
    ref[0] = 0;
    ref[2] = 1;
  }
  const dfloat rn = ref[0] * n[0] + ref[1] * n[1] + ref[2] * n[2];
  for (int d = 0; d < 3; d++)
    t1[d] = ref[d] - rn * n[d];
  const dfloat m = std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
  for (int d = 0; d < 3; d++)
    t1[d] /= m;
  t2[0] = t1[1] * n[2] - t1[2] * n[1];
  t2[1] = t1[2] * n[0] - t1[0] * n[2];
  t2[2] = t1[0] * n[1] - t1[1] * n[0];
}

// v in global axes -> (v.t1, v.n, v.t2) when wm_project_normal
void toOutputFrame(int n, const dfloat v[3], dfloat out[3])
{
  if (!cfg.projectNormal) {
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];
    return;
  }
  const dfloat *t1 = &wallT1[3 * n];
  const dfloat *nn = &wallN[3 * n];
  const dfloat *t2 = &wallT2[3 * n];
  out[0] = v[0] * t1[0] + v[1] * t1[1] + v[2] * t1[2];
  out[1] = v[0] * nn[0] + v[1] * nn[1] + v[2] * nn[2];
  out[2] = v[0] * t2[0] + v[1] * t2[1] + v[2] * t2[2];
}

void setupWallPoints(nrs_t *nrs)
{
  mesh_t *mesh = nrs->meshV;
  std::vector<dfloat> sgeo(mesh->o_sgeo.size() / sizeof(dfloat));
  mesh->o_sgeo.copyTo(sgeo.data());

  K = cfg.numOffwallPoints;
  for (dlong e = 0; e < mesh->Nelements; e++) {
    for (int f = 0; f < mesh->Nfaces; f++) {
      const int bID = mesh->EToB[f + mesh->Nfaces * e];
      if (std::find(cfg.bIDs.begin(), cfg.bIDs.end(), bID) == cfg.bIDs.end())
        continue;
      for (int m = 0; m < mesh->Nfp; m++) {
        const dlong sk = e * mesh->Nfaces * mesh->Nfp + f * mesh->Nfp + m;
        const dlong idM = mesh->vmapM[sk];
        const dlong sid = mesh->Nsgeo * sk;
        // sgeo normal points out of the element, i.e. into the wall
        dfloat n[3] = {-sgeo[sid + NXID], -sgeo[sid + NYID], -sgeo[sid + NZID]};
        const dfloat nm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int d = 0; d < 3; d++)
          n[d] /= nm;
        dfloat t1[3], t2[3];
        wallBasis(n, t1, t2);

        const dfloat xw[3] = {mesh->x[idM], mesh->y[idM], mesh->z[idM]};
        wallToV.push_back(idM);
        for (int d = 0; d < 3; d++) {
          wallXYZ.push_back(xw[d]);
          wallN.push_back(n[d]);
          wallT1.push_back(t1[d]);
          wallT2.push_back(t2[d]);
        }
        for (int k = 0; k < K; k++) {
          const dfloat yk = (static_cast<dfloat>(k + 1) / K) * cfg.witnessDist;
          wallYs.push_back(yk);
          witX.push_back(xw[0] + yk * n[0]);
          witY.push_back(xw[1] + yk * n[1]);
          witZ.push_back(xw[2] + yk * n[2]);
        }
      }
    }
  }
  nWall = wallToV.size();
  nPts = nWall * K;

  staticCols.assign(static_cast<size_t>(nWall) * wallModelML::numStaticCols(), 0);
  const int ns = wallModelML::numStaticCols();
  for (int n = 0; n < nWall; n++) {
    for (int d = 0; d < 3; d++) {
      staticCols[n * ns + 0 + d] = wallXYZ[3 * n + d];
      staticCols[n * ns + 3 + d] = wallN[3 * n + d];
      staticCols[n * ns + 6 + d] = wallT1[3 * n + d];
      staticCols[n * ns + 9 + d] = wallT2[3 * n + d];
    }
    for (int k = 0; k < K; k++)
      staticCols[n * ns + wallModelML::numStaticFixedCols + k] = wallYs[n * K + k];
  }
}

void setupInterpolation(nrs_t *nrs)
{
  mesh_t *mesh = nrs->meshV;
  interp = new pointInterpolation_t(nrs);
  interp->setPoints(nPts, witX.data(), witY.data(), witZ.data());
  interp->find(pointInterpolation_t::VerbosityLevel::None);

  const dlong nAlloc = std::max(nPts, 1);
  o_witU = platform->device.malloc(3 * nAlloc * sizeof(dfloat));
  o_witGP = platform->device.malloc(3 * nAlloc * sizeof(dfloat));

  // Self-check: interpolate the coordinates back to the witness points.
  // A point findpts could not locate (outside the mesh) shows up here.
  occa::memory o_xyz = platform->device.malloc(3 * nrs->fieldOffset * sizeof(dfloat));
  o_xyz.copyFrom(mesh->o_x, mesh->Nlocal * sizeof(dfloat), 0 * nrs->fieldOffset * sizeof(dfloat));
  o_xyz.copyFrom(mesh->o_y, mesh->Nlocal * sizeof(dfloat), 1 * nrs->fieldOffset * sizeof(dfloat));
  o_xyz.copyFrom(mesh->o_z, mesh->Nlocal * sizeof(dfloat), 2 * nrs->fieldOffset * sizeof(dfloat));
  interp->eval(3, nrs->fieldOffset, o_xyz, nAlloc, o_witU);
  std::vector<dfloat> xyz(3 * nAlloc);
  o_witU.copyTo(xyz.data());
  o_xyz.free();

  dfloat errMax = 0;
  for (int p = 0; p < nPts; p++) {
    errMax = std::max(errMax, std::abs(xyz[0 * nAlloc + p] - witX[p]));
    errMax = std::max(errMax, std::abs(xyz[1 * nAlloc + p] - witY[p]));
    errMax = std::max(errMax, std::abs(xyz[2 * nAlloc + p] - witZ[p]));
  }
  MPI_Allreduce(MPI_IN_PLACE, &errMax, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm);
  if (platform->comm.mpiRank == 0)
    printf("wallModelML: witness interpolation check max|x_interp - x_wit| = %.3e\n", errMax);
  nrsCheck(errMax > 1e-6 * cfg.witnessDist + 1e-10, platform->comm.mpiComm, EXIT_FAILURE,
           "wallModelML: witness points not found in the mesh (max error %g)\n", errMax);
}

// Fields the sample is taken from: instantaneous u/p or the walave filter.
void sampleFields(nrs_t *nrs, occa::memory &o_U, occa::memory &o_P)
{
  if (cfg.sampleField == "walave") {
    o_U = o_Uwal;
    o_P = o_Pwal;
  }
  else {
    o_U = nrs->o_U;
    o_P = nrs->o_P;
  }
}
} // namespace

namespace wallModelML
{
const config_t &config() { return cfg; }
int numPoints() { return nWall; }
int numInputs() { return inputsPerPoint * K; }
int numStaticCols() { return numStaticFixedCols + K; }
dfloat nu() { return nuRef; }
const std::vector<dfloat> &inputs() { return inputData; }
const std::vector<dfloat> &outputs() { return outputData; }
const std::vector<dfloat> &staticData() { return staticCols; }

// Reichardt law-of-the-wall solve for u_tau, a line-by-line port of SOD2D
// reichardt_utau (mod_wall_model.f90) so the u_tau^R input column is the
// same function in both codes.
dfloat reichardtUtau(dfloat Us, dfloat y, dfloat nu)
{
  if (!(y > 0 && Us > 1.0e-10))
    return 0;
  dfloat utau = std::sqrt(Us * nu / y);
  if (utau * y / nu > 5.0) {
    const dfloat vkinv = 1.0 / 0.41;
    const dfloat onovu = 1.0 / Us;
    const dfloat xmuit = y / nu;
    const dfloat oneoe = 1.0 / 11.0;
    dfloat parco = 1.0;
    int itera = 0;
    while (parco >= 1.0e-6 && itera < 100) {
      itera++;
      utau = std::max(utau, static_cast<dfloat>(0));
      const dfloat yplus = utau * xmuit;
      const dfloat ypele = yplus * oneoe;
      const dfloat ypel2 = std::min(ypele, static_cast<dfloat>(20));
      const dfloat expye = std::exp(-ypel2);
      const dfloat yplu2 = std::min(yplus, static_cast<dfloat>(70));
      const dfloat expyt = std::exp(-yplu2 * 0.33);
      const dfloat firsl = vkinv * std::log(1.0 + 0.4 * yplus);
      const dfloat fdvfr = utau * (firsl + 7.8 * (1.0 - expye - ypele * expyt)) - Us;
      const dfloat diffd = firsl + vkinv * 0.4 * yplus / (1.0 + 0.4 * yplus) +
                           7.8 * (1.0 - expye * (1.0 - ypele) - ypele * expyt * (2.0 - yplus * 0.33));
      const dfloat devfr = -fdvfr / diffd;
      parco = std::abs(devfr * onovu);
      utau = utau + devfr;
    }
  }
  return utau;
}

void setup(nrs_t *nrs)
{
  readConfig();
  if (cfg.mode == "none")
    return;

  dfloat mu = 0, rho = 1;
  platform->options.getArgs("VISCOSITY", mu);
  platform->options.getArgs("DENSITY", rho);
  nuRef = mu / rho;

  setupWallPoints(nrs);
  setupInterpolation(nrs);

  const auto fieldBytes = nrs->fieldOffset * sizeof(dfloat);
  o_gradP = platform->device.malloc(3 * fieldBytes);
  if (cfg.mode == "train")
    o_Sij = platform->device.malloc(6 * fieldBytes);
  if (cfg.sampleField == "walave") {
    o_Uwal = platform->device.malloc(nrs->NVfields * fieldBytes);
    o_Pwal = platform->device.malloc(fieldBytes);
  }
  inputData.assign(static_cast<size_t>(nWall) * numInputs(), 0);
  outputData.assign(static_cast<size_t>(nWall) * numOutputs, 0);

  // BC: traction vector (global axes) in usrwrk slots 0..2
  if (cfg.mode == "infer") {
    if (nrs->o_usrwrk.size() < 3 * fieldBytes) {
      nrs->usrwrk = (dfloat *)calloc(3 * nrs->fieldOffset, sizeof(dfloat));
      nrs->o_usrwrk = platform->device.malloc(3 * fieldBytes, nrs->usrwrk);
    }
  }

  long long nWallGlobal = nWall;
  MPI_Allreduce(MPI_IN_PLACE, &nWallGlobal, 1, MPI_LONG_LONG, MPI_SUM, platform->comm.mpiComm);
  if (platform->comm.mpiRank == 0) {
    printf("wallModelML: mode=%s  bIDs=", cfg.mode.c_str());
    for (auto b : cfg.bIDs)
      printf("%d ", b);
    printf(" wall nodes=%lld  K=%d  d=%g  field=%s  project_normal=%d  nu=%g\n",
           nWallGlobal, K, cfg.witnessDist, cfg.sampleField.c_str(), (int)cfg.projectNormal, nuRef);
  }

#ifdef ENABLE_SMARTREDIS
  if (cfg.mode == "train" || cfg.inferenceBackend == "smartredis") {
    smartredis::init_client(nrs);
    smartredis::init_check_run();
    const int nOut = (cfg.mode == "train") ? numOutputs : numInferOutputs;
    smartredis::init_wallModel(nWall, numInputs(), nOut);
    if (cfg.mode == "train") {
      std::vector<double> meta = {(double)contractVersion,
                                  (double)nuRef,
                                  (double)K,
                                  (double)numInputs(),
                                  (double)numOutputs,
                                  cfg.sampleField == "instantaneous" ? 1.0 : 0.0,
                                  (double)cfg.periodWalave,
                                  cfg.projectNormal ? 1.0 : 0.0,
                                  (double)platform->comm.mpiCommSize,
                                  (double)cfg.witnessDist};
      smartredis::put_wallModel_static(nWall, numStaticCols(), staticCols.data(), meta);
    }
  }
#endif
  initialized = true;
}

void updateWalave(nrs_t *nrs)
{
  if (cfg.sampleField != "walave")
    return;
  mesh_t *mesh = nrs->meshV;
  if (!walaveStarted) {
    o_Uwal.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));
    o_Pwal.copyFrom(nrs->o_P, nrs->fieldOffset * sizeof(dfloat));
    walaveStarted = true;
    return;
  }
  const dfloat dt = nrs->dt[0];
  const dfloat dtfact = dt / (dt + cfg.periodWalave);
  const dfloat avwei = 1.0 - dtfact;
  platform->linAlg->axpbyMany(mesh->Nlocal, nrs->NVfields, nrs->fieldOffset, dtfact, nrs->o_U, avwei, o_Uwal);
  platform->linAlg->axpby(mesh->Nlocal, dtfact, nrs->o_P, avwei, o_Pwal);
}

void sampleInputs(nrs_t *nrs)
{
  mesh_t *mesh = nrs->meshV;
  occa::memory o_U, o_P;
  sampleFields(nrs, o_U, o_P);

  // grad(p): element gradient, then averaged across elements (as SOD2D's
  // eval_gradient on the nodal field)
  nrs->gradientVolumeKernel(mesh->Nelements, mesh->o_vgeo, mesh->o_D, nrs->fieldOffset, o_P, o_gradP);
  oogs::startFinish(o_gradP, nrs->NVfields, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);
  platform->linAlg->axmyVector(mesh->Nlocal, nrs->fieldOffset, 0, 1.0, mesh->o_invLMM, o_gradP);

  // findpts eval is collective: every rank calls it, also with 0 points
  const dlong nAlloc = std::max(nPts, 1);
  interp->eval(nrs->NVfields, nrs->fieldOffset, o_U, nAlloc, o_witU);
  interp->eval(nrs->NVfields, nrs->fieldOffset, o_gradP, nAlloc, o_witGP);
  std::vector<dfloat> u(3 * nAlloc), gp(3 * nAlloc);
  o_witU.copyTo(u.data());
  o_witGP.copyTo(gp.data());

  const int nIn = numInputs();
  for (int n = 0; n < nWall; n++) {
    const dfloat *nn = &wallN[3 * n];
    for (int k = 0; k < K; k++) {
      const int p = n * K + k;
      const dfloat uk[3] = {u[0 * nAlloc + p], u[1 * nAlloc + p], u[2 * nAlloc + p]};
      const dfloat gk[3] = {gp[0 * nAlloc + p], gp[1 * nAlloc + p], gp[2 * nAlloc + p]};
      const dfloat un = uk[0] * nn[0] + uk[1] * nn[1] + uk[2] * nn[2];
      const dfloat ut[3] = {uk[0] - un * nn[0], uk[1] - un * nn[1], uk[2] - un * nn[2]};
      const dfloat Us = std::sqrt(ut[0] * ut[0] + ut[1] * ut[1] + ut[2] * ut[2]);

      dfloat *row = &inputData[static_cast<size_t>(n) * nIn + k * inputsPerPoint];
      toOutputFrame(n, uk, row + colVel);
      toOutputFrame(n, gk, row + colGradP);
      row[colUtauR] = reichardtUtau(Us, wallYs[p], nuRef);
    }
  }
}

void sampleTarget(nrs_t *nrs)
{
  mesh_t *mesh = nrs->meshV;
  occa::memory o_U, o_P;
  sampleFields(nrs, o_U, o_P);

  // Element-local strain rate at the wall node of the wall element, as in
  // SOD2D wmodel_surf (no averaging across elements): S_ij = (du_i/dx_j + du_j/dx_i)/2,
  // layout S11 S22 S33 S12 S23 S13
  nrs->SijOijKernel(mesh->Nelements, nrs->fieldOffset, 0, 0, mesh->o_vgeo, mesh->o_D, o_U, o_Sij);

  std::vector<dfloat> S(6 * nrs->fieldOffset), mue(nrs->fieldOffset);
  o_Sij.copyTo(S.data(), 6 * nrs->fieldOffset * sizeof(dfloat));
  nrs->o_mue.copyTo(mue.data(), nrs->fieldOffset * sizeof(dfloat));

  const dlong off = nrs->fieldOffset;
  for (int n = 0; n < nWall; n++) {
    const dlong v = wallToV[n];
    const dfloat *nn = &wallN[3 * n];
    const dfloat s11 = S[v + 0 * off], s22 = S[v + 1 * off], s33 = S[v + 2 * off];
    const dfloat s12 = S[v + 3 * off], s23 = S[v + 4 * off], s13 = S[v + 5 * off];
    // tau.n_in with tau = 2 mu_eff S (incompressible: no divergence term);
    // mue holds the effective viscosity of the momentum equation (incl. SGS)
    const dfloat twoMu = 2 * mue[v];
    const dfloat tv[3] = {twoMu * (s11 * nn[0] + s12 * nn[1] + s13 * nn[2]),
                          twoMu * (s12 * nn[0] + s22 * nn[1] + s23 * nn[2]),
                          twoMu * (s13 * nn[0] + s23 * nn[1] + s33 * nn[2])};
    const dfloat tn = tv[0] * nn[0] + tv[1] * nn[1] + tv[2] * nn[2];
    const dfloat tt[3] = {tv[0] - tn * nn[0], tv[1] - tn * nn[1], tv[2] - tn * nn[2]};
    toOutputFrame(n, tt, &outputData[static_cast<size_t>(n) * numOutputs]);
  }
}

// Traction on the fluid from u_tau at each wall node, direction from the
// tangential velocity at the first off-wall point:
//   tau_w = u_tau^2 * u_t/|u_t|, or nu |u_t|/y below y+ = 5 (as SOD2D
//   evalWallModelML); the BC traction is -tau_w (rho = 1).
void applyWallShear(nrs_t *nrs, const std::vector<dfloat> &utau)
{
  std::vector<dfloat> u(3 * std::max(nPts, 1));
  o_witU.copyTo(u.data());
  const dlong nAlloc = std::max(nPts, 1);
  const dlong off = nrs->fieldOffset;
  dfloat rho = 1;
  platform->options.getArgs("DENSITY", rho);

  for (int n = 0; n < nWall; n++) {
    const int p = n * K; // k = 1
    const dfloat *nn = &wallN[3 * n];
    const dfloat uk[3] = {u[0 * nAlloc + p], u[1 * nAlloc + p], u[2 * nAlloc + p]};
    const dfloat un = uk[0] * nn[0] + uk[1] * nn[1] + uk[2] * nn[2];
    const dfloat ut[3] = {uk[0] - un * nn[0], uk[1] - un * nn[1], uk[2] - un * nn[2]};
    const dfloat Us = std::sqrt(ut[0] * ut[0] + ut[1] * ut[1] + ut[2] * ut[2]);
    const dfloat y = wallYs[p];
    const dfloat ut1 = utau[n];

    dfloat tmag = 0;
    if (Us > 1.0e-10) {
      tmag = (y * ut1 / nuRef < 5.0) ? rho * nuRef / y : rho * ut1 * ut1 / Us;
    }
    const dlong v = wallToV[n];
    for (int d = 0; d < 3; d++)
      nrs->usrwrk[v + d * off] = -tmag * ut[d];
  }
  nrs->o_usrwrk.copyFrom(nrs->usrwrk, 3 * off * sizeof(dfloat));
}

void executeStep(nrs_t *nrs, dfloat time, int tstep)
{
  if (!initialized)
    return;

  updateWalave(nrs);

  if (cfg.mode == "train") {
    if (tstep > 0 && tstep % cfg.leapWitSave == 0) {
      sampleInputs(nrs);
      sampleTarget(nrs);
      {
        // monitor: wall-node means of the target and of u_tau^R at point 1
        double sums[5] = {0, 0, 0, 0, (double)nWall};
        for (int n = 0; n < nWall; n++) {
          for (int j = 0; j < numOutputs; j++)
            sums[j] += outputData[static_cast<size_t>(n) * numOutputs + j];
          sums[3] += inputData[static_cast<size_t>(n) * numInputs() + colUtauR];
        }
        MPI_Allreduce(MPI_IN_PLACE, sums, 5, MPI_DOUBLE, MPI_SUM, platform->comm.mpiComm);
        if (platform->comm.mpiRank == 0)
          printf("wallModelML: sample %d step %d time %.6e  mean tau_w (%.6e %.6e %.6e)  mean u_tau^R %.6e\n",
                 sampleIndex, tstep, time, sums[0] / sums[4], sums[1] / sums[4], sums[2] / sums[4],
                 sums[3] / sums[4]);
      }
#ifdef ENABLE_SMARTREDIS
      std::vector<dfloat> rows(static_cast<size_t>(nWall) * (numInputs() + numOutputs));
      const int nc = numInputs() + numOutputs;
      for (int n = 0; n < nWall; n++) {
        std::copy_n(&inputData[static_cast<size_t>(n) * numInputs()], numInputs(), &rows[static_cast<size_t>(n) * nc]);
        std::copy_n(&outputData[static_cast<size_t>(n) * numOutputs], numOutputs,
                    &rows[static_cast<size_t>(n) * nc + numInputs()]);
      }
      smartredis::put_wallModel_sample(sampleIndex, tstep, time, nWall, nc, rows.data());
#endif
      sampleIndex++;
#ifdef ENABLE_SMARTREDIS
      // The witness sink (or a trainer) sets check-run to 0 when it has enough
      if (smartredis::check_run() == 0)
        platform->options.setArgs("END TIME", to_string_f(-10.0));
#endif
    }
    if (nrs->lastStep)
      finalize(nrs);
  }
  else if (cfg.mode == "infer") {
    if (tstep > 0 && tstep % cfg.inferStride == 0) {
      sampleInputs(nrs);
      std::vector<dfloat> utau(nWall, 0);
      if (cfg.inferenceBackend == "reichardt") {
        for (int n = 0; n < nWall; n++)
          utau[n] = inputData[static_cast<size_t>(n) * numInputs() + colUtauR];
      }
      else {
#ifdef ENABLE_SMARTREDIS
        smartredis::run_wallModel(tstep, nWall, numInputs(), inputData.data(), numInferOutputs, utau.data());
#endif
      }
      applyWallShear(nrs, utau);

      // monitor: wall-node means of the model u_tau, of the applied |tau_w|
      // and of u_tau^R (input column at point 1), and max |u_tau - u_tau^R|
      if ((tstep / cfg.inferStride) % 10 == 0) {
        double sums[4] = {0, 0, 0, (double)nWall};
        double dmax = 0;
        for (int n = 0; n < nWall; n++) {
          const dlong v = wallToV[n];
          const dfloat utauR = inputData[static_cast<size_t>(n) * numInputs() + colUtauR];
          sums[0] += utau[n];
          sums[1] += std::sqrt(nrs->usrwrk[v] * nrs->usrwrk[v] +
                               nrs->usrwrk[v + nrs->fieldOffset] * nrs->usrwrk[v + nrs->fieldOffset] +
                               nrs->usrwrk[v + 2 * nrs->fieldOffset] * nrs->usrwrk[v + 2 * nrs->fieldOffset]);
          sums[2] += utauR;
          dmax = std::max(dmax, (double)std::abs(utau[n] - utauR));
        }
        MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_DOUBLE, MPI_SUM, platform->comm.mpiComm);
        MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm);
        if (platform->comm.mpiRank == 0)
          printf("wallModelML: step %d time %.6e  mean u_tau %.10e  mean |tau_w| %.10e  "
                 "mean u_tau^R %.10e  max|u_tau-u_tau^R| %.3e\n",
                 tstep, time, sums[0] / sums[3], sums[1] / sums[3], sums[2] / sums[3], dmax);
      }
    }
  }
}

void finalize(nrs_t *nrs)
{
  static bool done = false;
  if (done)
    return;
  done = true;
#ifdef ENABLE_SMARTREDIS
  if (initialized && cfg.mode == "train")
    smartredis::put_wallModel_done(sampleIndex);
#endif
}
} // namespace wallModelML
