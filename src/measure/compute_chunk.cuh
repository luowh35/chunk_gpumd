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
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class ComputeChunk : public Property
{
public:
  ComputeChunk(const char** param, int num_param, Box& box);

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

  // Public interface for ave_chunk to access chunk data
  int get_nchunk() const { return nchunk_; }
  const GPU_Vector<int>& get_ichunk() const { return ichunk_; }
  const std::vector<double>& get_chunk_volume() const { return chunk_volume_cpu_; }
  const std::vector<double>& get_chunk_coords() const { return chunk_coords_cpu_; }
  int get_ncoord() const { return ncoord_; }

  void assign_chunks(const Atom& atom, const Box& box);

private:
  // Chunk configuration
  int dim_;                      // 1 = bin/1d, 2 = bin/2d, 3 = bin/3d
  int axis_[3];                  // Which axes to bin: 0=x, 1=y, 2=z
  double origin_[3];             // Bin origin coordinates
  double delta_[3];              // Bin width per dimension
  double invdelta_[3];           // 1/delta for fast binning
  int nlayers_[3];               // Number of bins per dimension
  int nchunk_;                   // Total number of chunks
  int ncoord_;                   // Number of coordinates per chunk (= dim_)

  // Chunk assignments
  GPU_Vector<int> ichunk_;       // ichunk_[i] = chunk ID for atom i (0 to nchunk-1)
  std::vector<double> chunk_volume_cpu_;  // Volume of each chunk
  std::vector<double> chunk_coords_cpu_;  // Center coordinates of each chunk

  // Compression support (for future implementation)
  bool compress_;

  void parse(const char** param, int num_param, Box& box);
  void calculate_chunk_volumes(const Box& box);
  void calculate_chunk_coords(const Box& box);
};
