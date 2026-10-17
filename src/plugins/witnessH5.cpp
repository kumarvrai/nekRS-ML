#include "nrs.hpp"
#include "platform.hpp"
#include "witnessH5.hpp"

#ifdef ENABLE_HDF5

#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace
{
void h5check(herr_t status, const char *what)
{
  nrsCheck(status < 0, MPI_COMM_SELF, EXIT_FAILURE, "witnessH5: %s failed\n", what);
}

hid_t h5id(hid_t id, const char *what)
{
  nrsCheck(id < 0, MPI_COMM_SELF, EXIT_FAILURE, "witnessH5: %s failed\n", what);
  return id;
}
} // namespace

struct witnessH5_t::impl_t {
  spec_t S;
  MPI_Comm comm;
  int rank = 0, nranks = 1;
  long long ntot = 0, off = 0; // global points, first global row of this rank

  hid_t fid = -1, dxpl = -1;
  hid_t ftype = -1;            // file and memory datatype of the real data
  std::vector<hid_t> dsSeries;
  hid_t dsTime = -1, dsStep = -1;

  // buffers, layout [p * flushEvery + s]
  std::vector<std::vector<double>> buf;
  std::vector<double> bufTime;
  std::vector<int> bufStep;

  // real data in the file datatype: no conversion inside a collective write
  const void *asFileType(const std::vector<double> &v, std::vector<float> &tmp) const
  {
    if (!S.float32)
      return v.data();
    tmp.assign(v.begin(), v.end());
    return tmp.data();
  }

  // Collective write of this rank's rows [off, off + nLocal) x cols [c0, c0 + nc)
  // from a row-major buffer nLocal x ldm (first nc columns used)
  void writeRows(hid_t ds, hid_t memType, const void *b, long long c0, int nc, int ldm) const
  {
    hid_t fs = h5id(H5Dget_space(ds), "H5Dget_space");
    const hsize_t mdims[2] = {(hsize_t)std::max(S.nLocal, 1), (hsize_t)std::max(ldm, 1)};
    hid_t ms = h5id(H5Screate_simple(2, mdims, nullptr), "H5Screate_simple");
    if (S.nLocal > 0 && nc > 0) {
      const hsize_t fstart[2] = {(hsize_t)off, (hsize_t)c0}, cnt[2] = {(hsize_t)S.nLocal, (hsize_t)nc};
      const hsize_t mstart[2] = {0, 0};
      h5check(H5Sselect_hyperslab(fs, H5S_SELECT_SET, fstart, nullptr, cnt, nullptr), "select file rows");
      h5check(H5Sselect_hyperslab(ms, H5S_SELECT_SET, mstart, nullptr, cnt, nullptr), "select memory rows");
    }
    else {
      h5check(H5Sselect_none(fs), "select none");
      h5check(H5Sselect_none(ms), "select none");
    }
    h5check(H5Dwrite(ds, memType, ms, fs, dxpl, b), "H5Dwrite rows");
    H5Sclose(ms);
    H5Sclose(fs);
  }

  // Collective write of a 1-D range [c0, c0 + n) by rank 0
  void writeRange1D(hid_t ds, hid_t memType, const void *b, long long c0, int n) const
  {
    hid_t fs = h5id(H5Dget_space(ds), "H5Dget_space");
    const hsize_t mdim = (hsize_t)std::max(n, 1);
    hid_t ms = h5id(H5Screate_simple(1, &mdim, nullptr), "H5Screate_simple");
    if (rank == 0 && n > 0) {
      const hsize_t start = c0, cnt = n, mstart = 0;
      h5check(H5Sselect_hyperslab(fs, H5S_SELECT_SET, &start, nullptr, &cnt, nullptr), "select range");
      h5check(H5Sselect_hyperslab(ms, H5S_SELECT_SET, &mstart, nullptr, &cnt, nullptr), "select range");
    }
    else {
      h5check(H5Sselect_none(fs), "select none");
      h5check(H5Sselect_none(ms), "select none");
    }
    h5check(H5Dwrite(ds, memType, ms, fs, dxpl, b), "H5Dwrite range");
    H5Sclose(ms);
    H5Sclose(fs);
  }

  void writeStatic(const staticData_t &st)
  {
    const hsize_t dims[2] = {(hsize_t)ntot, (hsize_t)st.ncols};
    hid_t sp = h5id(H5Screate_simple(2, dims, nullptr), "H5Screate_simple");
    hid_t ds = h5id(H5Dcreate2(fid, st.name.c_str(), ftype, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    st.name.c_str());
    H5Sclose(sp);
    std::vector<double> v(st.data.begin(), st.data.end());
    if (v.empty())
      v.resize(1);
    std::vector<float> tmp;
    writeRows(ds, ftype, asFileType(v, tmp), 0, st.ncols, st.ncols);
    H5Dclose(ds);
  }

  template <typename T> void writeSmall(const std::string &name, hid_t type, const std::vector<T> &v)
  {
    const hsize_t n = v.size();
    hid_t sp = h5id(H5Screate_simple(1, &n, nullptr), "H5Screate_simple");
    hid_t ds = h5id(H5Dcreate2(fid, name.c_str(), type, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name.c_str());
    H5Sclose(sp);
    writeRange1D(ds, type, v.data(), 0, (int)n);
    H5Dclose(ds);
  }

  // Attributes are written collectively: every rank passes the same value
  void attrNumber(const std::string &name, double v)
  {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = h5id(H5Acreate2(fid, name.c_str(), H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT), name.c_str());
    h5check(H5Awrite(a, H5T_NATIVE_DOUBLE, &v), name.c_str());
    H5Aclose(a);
    H5Sclose(sp);
  }

  void attrString(const std::string &name, const std::string &v)
  {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, std::max(v.size(), (size_t)1));
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = h5id(H5Acreate2(fid, name.c_str(), t, sp, H5P_DEFAULT, H5P_DEFAULT), name.c_str());
    h5check(H5Awrite(a, t, v.c_str()), name.c_str());
    H5Aclose(a);
    H5Sclose(sp);
    H5Tclose(t);
  }

  // fixed-length string array, as numpy "S" arrays written by h5py
  void attrStringArray(const std::string &name, const std::vector<std::string> &v)
  {
    size_t len = 1;
    for (auto &s : v)
      len = std::max(len, s.size());
    std::vector<char> b(len * std::max(v.size(), (size_t)1), '\0');
    for (size_t i = 0; i < v.size(); i++)
      std::copy(v[i].begin(), v[i].end(), b.begin() + i * len);
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, len);
    H5Tset_strpad(t, H5T_STR_NULLPAD);
    const hsize_t n = std::max(v.size(), (size_t)1);
    hid_t sp = H5Screate_simple(1, &n, nullptr);
    hid_t a = h5id(H5Acreate2(fid, name.c_str(), t, sp, H5P_DEFAULT, H5P_DEFAULT), name.c_str());
    h5check(H5Awrite(a, t, b.data()), name.c_str());
    H5Aclose(a);
    H5Sclose(sp);
    H5Tclose(t);
  }

  hid_t createSeries2D(const std::string &name)
  {
    const hsize_t dims[2] = {(hsize_t)ntot, 0}, maxd[2] = {(hsize_t)ntot, H5S_UNLIMITED};
    // one chunk per sample, as SOD2D ({npts, 1})
    const hsize_t chunk[2] = {(hsize_t)std::min(ntot, (long long)(1 << 24)), 1};
    hid_t sp = h5id(H5Screate_simple(2, dims, maxd), "H5Screate_simple");
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    h5check(H5Pset_chunk(dcpl, 2, chunk), "H5Pset_chunk");
    hid_t ds = h5id(H5Dcreate2(fid, name.c_str(), ftype, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT), name.c_str());
    H5Pclose(dcpl);
    H5Sclose(sp);
    return ds;
  }

  hid_t createSeries1D(const char *name, hid_t type)
  {
    const hsize_t dims = 0, maxd = H5S_UNLIMITED, chunk = 256;
    hid_t sp = h5id(H5Screate_simple(1, &dims, &maxd), "H5Screate_simple");
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    h5check(H5Pset_chunk(dcpl, 1, &chunk), "H5Pset_chunk");
    hid_t ds = h5id(H5Dcreate2(fid, name, type, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT), name);
    H5Pclose(dcpl);
    H5Sclose(sp);
    return ds;
  }

  void setExtent(long long nt)
  {
    for (auto ds : dsSeries) {
      const hsize_t d[2] = {(hsize_t)ntot, (hsize_t)nt};
      h5check(H5Dset_extent(ds, d), "H5Dset_extent");
    }
    const hsize_t d1 = nt;
    h5check(H5Dset_extent(dsTime, &d1), "H5Dset_extent");
    h5check(H5Dset_extent(dsStep, &d1), "H5Dset_extent");
  }

  hid_t fileAccess() const
  {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    h5check(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL), "H5Pset_fapl_mpio");
    return fapl;
  }

  void createFile()
  {
    hid_t fapl = fileAccess();
    fid = h5id(H5Fcreate(S.fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl), S.fileName.c_str());
    H5Pclose(fapl);

    for (auto &n : S.seriesNames)
      dsSeries.push_back(createSeries2D(n));
    dsTime = createSeries1D("time", ftype);
    dsStep = createSeries1D("istep", H5T_NATIVE_INT);

    for (auto &st : S.statics)
      writeStatic(st);

    std::vector<int> nwitPar(nranks), nwitOffset(nranks, 0);
    int nl = S.nLocal;
    MPI_Allgather(&nl, 1, MPI_INT, nwitPar.data(), 1, MPI_INT, comm);
    for (int r = 1; r < nranks; r++)
      nwitOffset[r] = nwitOffset[r - 1] + nwitPar[r - 1];
    writeSmall("nwitPar", H5T_NATIVE_INT, nwitPar);
    writeSmall("nwitOffset", H5T_NATIVE_INT, nwitOffset);
    for (auto &d : S.intData)
      writeSmall(d.first, H5T_NATIVE_INT, d.second);
    for (auto &d : S.realData) {
      if (S.float32)
        writeSmall(d.first, ftype, std::vector<float>(d.second.begin(), d.second.end()));
      else
        writeSmall(d.first, ftype, d.second);
    }

    for (auto &a : S.attrStrings)
      attrString(a.first, a.second);
    for (auto &a : S.attrStringArrays)
      attrStringArray(a.first, a.second);
    for (auto &a : S.attrNumbers)
      attrNumber(a.first, a.second);
  }

  // returns the samples kept
  long long openExisting()
  {
    hid_t fapl = fileAccess();
    fid = h5id(H5Fopen(S.fileName.c_str(), H5F_ACC_RDWR, fapl), S.fileName.c_str());
    H5Pclose(fapl);

    auto fail = [&](const std::string &msg) {
      nrsAbort(comm, EXIT_FAILURE, "witnessH5: cannot continue %s: %s\n", S.fileName.c_str(), msg.c_str());
    };
    for (auto &n : S.seriesNames) {
      if (H5Lexists(fid, n.c_str(), H5P_DEFAULT) <= 0)
        fail("it has no dataset " + n);
      dsSeries.push_back(h5id(H5Dopen2(fid, n.c_str(), H5P_DEFAULT), n.c_str()));
    }
    dsTime = h5id(H5Dopen2(fid, "time", H5P_DEFAULT), "open time");
    dsStep = h5id(H5Dopen2(fid, "istep", H5P_DEFAULT), "open istep");

    hsize_t d[2] = {0, 0};
    {
      hid_t sp = H5Dget_space(dsSeries[0]);
      H5Sget_simple_extent_dims(sp, d, nullptr);
      H5Sclose(sp);
    }
    if ((long long)d[0] != ntot)
      fail("it has " + std::to_string(d[0]) + " points, this run " + std::to_string(ntot));
    const long long nt = d[1];

    // same partition (each rank appends its own rows)
    std::vector<int> nwitPar(nranks);
    {
      hid_t ds = h5id(H5Dopen2(fid, "nwitPar", H5P_DEFAULT), "open nwitPar");
      hid_t s = H5Dget_space(ds);
      hsize_t n = 0;
      H5Sget_simple_extent_dims(s, &n, nullptr);
      H5Sclose(s);
      if ((int)n != nranks)
        fail("it was written by " + std::to_string(n) + " ranks, this run has " + std::to_string(nranks));
      h5check(H5Dread(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, nwitPar.data()), "read nwitPar");
      H5Dclose(ds);
    }
    int bad = (nwitPar[rank] != S.nLocal);

    // same points: compare the checkStatic dataset (float32 file: 1e-5 relative)
    if (!S.checkStatic.empty()) {
      const staticData_t *st = nullptr;
      for (auto &s : S.statics)
        if (s.name == S.checkStatic)
          st = &s;
      nrsCheck(!st, comm, EXIT_FAILURE, "witnessH5: checkStatic %s is not a static dataset\n",
               S.checkStatic.c_str());
      if (!bad && S.nLocal > 0) {
        hid_t ds = h5id(H5Dopen2(fid, st->name.c_str(), H5P_DEFAULT), st->name.c_str());
        hid_t fs = H5Dget_space(ds);
        const hsize_t start[2] = {(hsize_t)off, 0}, cnt[2] = {(hsize_t)S.nLocal, (hsize_t)st->ncols};
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
        hid_t ms = H5Screate_simple(2, cnt, nullptr);
        std::vector<double> v((size_t)S.nLocal * st->ncols);
        h5check(H5Dread(ds, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT, v.data()), "read checkStatic");
        H5Sclose(ms);
        H5Sclose(fs);
        H5Dclose(ds);
        for (size_t i = 0; i < v.size() && !bad; i++)
          if (std::abs(v[i] - st->data[i]) > 1e-5 * std::max(1.0, std::abs((double)st->data[i])))
            bad = 1;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad)
      fail("its points differ from this run's (mesh, partition or sampling set changed)");

    // drop samples after the restart time: the run recomputes them. The one at
    // the restart time is kept (the restarted run samples from its step 1 on)
    std::vector<double> t(std::max(nt, 1LL));
    if (nt > 0)
      h5check(H5Dread(dsTime, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, t.data()), "read time");
    long long keep = 0;
    // float32 time: compare with a tolerance of a few ulps
    const double tol = 1e-6 * std::max(1.0, std::abs(S.restartTime));
    while (keep < nt && t[keep] <= S.restartTime + tol)
      keep++;
    if (keep < nt)
      setExtent(keep);
    if (rank == 0)
      printf("witnessH5: continuing %s: %lld samples kept, %lld dropped (time > %g)\n", S.fileName.c_str(), keep,
             nt - keep, S.restartTime);
    return keep;
  }
};

bool witnessH5_t::available() { return true; }

witnessH5_t::~witnessH5_t()
{
  if (opened)
    close();
  delete impl;
}

void witnessH5_t::open(const spec_t &spec)
{
  nrsCheck(opened, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "witnessH5: file already open");
  nrsCheck(spec.seriesNames.empty(), platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "witnessH5: no series");
  delete impl;
  impl = new impl_t;
  auto &I = *impl;
  I.S = spec;
  I.S.flushEvery = std::max(I.S.flushEvery, 1);
  I.comm = platform->comm.mpiComm;
  MPI_Comm_rank(I.comm, &I.rank);
  MPI_Comm_size(I.comm, &I.nranks);
  long long nl = I.S.nLocal;
  MPI_Allreduce(&nl, &I.ntot, 1, MPI_LONG_LONG, MPI_SUM, I.comm);
  MPI_Exscan(&nl, &I.off, 1, MPI_LONG_LONG, MPI_SUM, I.comm);
  if (I.rank == 0)
    I.off = 0;
  nrsCheck(I.ntot == 0, I.comm, EXIT_FAILURE, "witnessH5: %s has no points\n", I.S.fileName.c_str());
  for (auto &st : I.S.statics)
    nrsCheck(st.data.size() != (size_t)I.S.nLocal * st.ncols, MPI_COMM_SELF, EXIT_FAILURE,
             "witnessH5: static %s has %zu values, expected nLocal x %d\n", st.name.c_str(), st.data.size(),
             st.ncols);

  I.ftype = I.S.float32 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
  I.dxpl = H5Pcreate(H5P_DATASET_XFER);
  h5check(H5Pset_dxpl_mpio(I.dxpl, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio");

  int exists = 0;
  if (I.rank == 0) {
    FILE *f = fopen(I.S.fileName.c_str(), "r");
    exists = (f != nullptr);
    if (f)
      fclose(f);
  }
  MPI_Bcast(&exists, 1, MPI_INT, 0, I.comm);
  if (I.S.continueFile && exists) {
    ntFile = I.openExisting();
  }
  else {
    I.createFile();
    ntFile = 0;
  }

  const size_t nb = (size_t)std::max(I.S.nLocal, 1) * I.S.flushEvery;
  I.buf.assign(I.S.seriesNames.size(), std::vector<double>(nb));
  I.bufTime.assign(I.S.flushEvery, 0);
  I.bufStep.assign(I.S.flushEvery, 0);
  nbuf = 0;
  h5check(H5Fflush(I.fid, H5F_SCOPE_GLOBAL), "H5Fflush");
  opened = true;
  if (I.rank == 0)
    printf("witnessH5: %s  %lld points  %zu series  %s  flush every %d samples\n", I.S.fileName.c_str(), I.ntot,
           I.S.seriesNames.size(), I.S.float32 ? "float32" : "float64", I.S.flushEvery);
}

void witnessH5_t::append(int tstep, double time, const dfloat *data)
{
  nrsCheck(!opened, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "witnessH5::append before open");
  auto &I = *impl;
  const int B = I.S.flushEvery;
  const int ns = I.S.seriesNames.size();
  for (int p = 0; p < I.S.nLocal; p++)
    for (int j = 0; j < ns; j++)
      I.buf[j][(size_t)p * B + nbuf] = data[(size_t)p * ns + j];
  I.bufTime[nbuf] = time;
  I.bufStep[nbuf] = tstep;
  nbuf++;
  if (nbuf == B)
    flush();
}

void witnessH5_t::flush()
{
  if (!opened || nbuf == 0)
    return;
  auto &I = *impl;
  const int B = I.S.flushEvery;
  I.setExtent(ntFile + nbuf);
  std::vector<float> tmp;
  for (size_t j = 0; j < I.dsSeries.size(); j++)
    I.writeRows(I.dsSeries[j], I.ftype, I.asFileType(I.buf[j], tmp), ntFile, nbuf, B);
  I.writeRange1D(I.dsTime, I.ftype, I.asFileType(I.bufTime, tmp), ntFile, nbuf);
  I.writeRange1D(I.dsStep, H5T_NATIVE_INT, I.bufStep.data(), ntFile, nbuf);
  h5check(H5Fflush(I.fid, H5F_SCOPE_GLOBAL), "H5Fflush");
  ntFile += nbuf;
  nbuf = 0;
}

void witnessH5_t::close()
{
  if (!opened)
    return;
  flush();
  auto &I = *impl;
  for (auto ds : I.dsSeries)
    H5Dclose(ds);
  H5Dclose(I.dsTime);
  H5Dclose(I.dsStep);
  H5Pclose(I.dxpl);
  H5Fclose(I.fid);
  I.dsSeries.clear();
  opened = false;
  if (I.rank == 0)
    printf("witnessH5: closed %s with %lld samples\n", I.S.fileName.c_str(), ntFile);
}

#else // ENABLE_HDF5

struct witnessH5_t::impl_t {
};

bool witnessH5_t::available() { return false; }
witnessH5_t::~witnessH5_t() { delete impl; }
void witnessH5_t::open(const spec_t &)
{
  nrsAbort(platform->comm.mpiComm, EXIT_FAILURE, "%s\n",
           "witnessH5: needs nekRS built with -DENABLE_HDF5=ON (parallel HDF5)");
}
void witnessH5_t::append(int, double, const dfloat *) {}
void witnessH5_t::flush() {}
void witnessH5_t::close() {}

#endif // ENABLE_HDF5
