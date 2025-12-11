/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "property.cuh"
#include "compute_chunk.cuh"
#include "measure.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class AveChunk : public Property
{
public:
  AveChunk(const char** param, int num_param, Measure& measure);

  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  virtual void process(
    const int number_of_steps,
    int step,
    const int fixed_group,
    const int move_group,
    const double global_time,
    const double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force) override;

  virtual void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  // Timing control
  int nevery_;              // Sample interval
  int nrepeat_;             // Number of samples per output
  int nfreq_;               // Output interval
  int irepeat_;             // Current repeat counter

  // Averaging mode
  enum AveMode { ONE, RUNNING, WINDOW };
  AveMode ave_mode_;
  int nwindow_;             // Window size for WINDOW mode

  // Properties to compute
  std::vector<std::string> property_names_;
  int nvalues_;             // Number of properties

  // Flags for what to compute
  bool compute_density_number_;
  bool compute_density_mass_;
  bool compute_vx_, compute_vy_, compute_vz_;
  bool compute_temp_;
  bool compute_fx_, compute_fy_, compute_fz_;
  bool compute_mass_;

  // Reference to compute_chunk
  ComputeChunk* compute_chunk_;
  int nchunk_;

  // Accumulation arrays (GPU)
  GPU_Vector<int> count_one_;       // Atom count per chunk for current sample
  GPU_Vector<double> values_one_;   // [nchunk * nvalues] for current sample
  GPU_Vector<int> count_many_;      // Accumulated over nrepeat samples
  GPU_Vector<double> values_many_;  // [nchunk * nvalues] accumulated

  // Cross-epoch storage (CPU)
  std::vector<int> count_total_;        // For RUNNING/WINDOW modes
  std::vector<double> values_total_;    // [nchunk * nvalues]
  std::vector<std::vector<int>> count_list_;      // For WINDOW mode
  std::vector<std::vector<double>> values_list_;  // For WINDOW mode
  int iwindow_;             // Current window index

  // For output
  std::vector<double> count_avg_output_;  // Averaged count for current output

  // Output
  FILE* fid_;
  std::string filename_;

  void parse(const char** param, int num_param, Measure& measure);
  void parse_property(const std::string& prop);
  void allocate_arrays();
  void sample(const Atom& atom, const Box& box);
  void compute_final_averages();
  void write_output(int step);
};
