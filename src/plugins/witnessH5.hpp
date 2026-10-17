#if !defined(nekrs_witnessh5_hpp_)
#define nekrs_witnessh5_hpp_

// Parallel HDF5 writer for witness (probe) time series, in the layout of
// SOD2D's witness files: a set of points distributed over the ranks, static
// per-point arrays, and per-point time series that grow by one column per
// sample. Usable by any sampling plugin (wall-model witnesses, probe lines,
// planes ...); the caller names the datasets.
//
// File layout (npts = points over all ranks, nt = samples):
//   <series name>        (npts, nt)  one per series column, nt extendible
//   time (nt), istep (nt)
//   <static name>        (npts, ncols) per-point arrays (coordinates, ...)
//   <int/real dataset>   small arrays, written as given by rank 0
//   nwitPar, nwitOffset  (nranks) points per rank and their first global row
//   file attributes      strings, string arrays, numbers
// Points are ordered by rank, then in local order; each rank writes its rows
// with collective MPI-IO. Samples are buffered and written every flushEvery
// samples (and on flush/close).
//
// continueFile: reopen an existing file, check the layout and the static
// dataset checkStatic (e.g. the coordinates) against this run, drop samples
// with time > restartTime (they are recomputed; the one at the restart time
// stays, a restarted run samples from its step 1 on) and append after the
// rest. istep restarts with the run's step counter.
//
// Needs nekRS built with -DENABLE_HDF5=ON (parallel HDF5); otherwise
// available() is false and open() aborts.

#include "nrs.hpp"
#include <string>
#include <utility>
#include <vector>

class witnessH5_t
{
public:
  struct staticData_t {
    std::string name;
    int ncols;
    std::vector<dfloat> data; // nLocal x ncols, row major
  };

  struct spec_t {
    std::string fileName;
    bool continueFile = false;
    double restartTime = 0;
    std::string checkStatic;        // static dataset compared on continue
    bool float32 = true;            // file datatype of the real data, float32 as SOD2D (rp = 4)
    int flushEvery = 10;            // samples buffered between writes
    int nLocal = 0;                 // local points (rows)
    std::vector<std::string> seriesNames;
    std::vector<staticData_t> statics;
    std::vector<std::pair<std::string, std::vector<int>>> intData;
    std::vector<std::pair<std::string, std::vector<double>>> realData;
    std::vector<std::pair<std::string, std::string>> attrStrings;
    std::vector<std::pair<std::string, std::vector<std::string>>> attrStringArrays;
    std::vector<std::pair<std::string, double>> attrNumbers;
  };

  static bool available();

  witnessH5_t() = default;
  ~witnessH5_t();
  witnessH5_t(const witnessH5_t &) = delete;
  witnessH5_t &operator=(const witnessH5_t &) = delete;

  void open(const spec_t &spec);
  // data: nLocal x seriesNames.size(), row major
  void append(int tstep, double time, const dfloat *data);
  void flush();
  void close();
  bool isOpen() const { return opened; }
  long long numSamples() const { return ntFile + nbuf; }

private:
  struct impl_t;
  impl_t *impl = nullptr;
  bool opened = false;
  long long ntFile = 0;
  int nbuf = 0;
};

#endif
