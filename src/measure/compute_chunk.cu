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

/*----------------------------------------------------------------------------80
Compute chunk/atom: assign atoms to spatial bins for chunk-based analysis
------------------------------------------------------------------------------*/

#include "compute_chunk.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

// GPU kernel to assign atoms to 1D bins
static __global__ void assign_atoms_to_bins_1d(
  int N,
  const double* x,
  const double* y,
  const double* z,
  int axis,                    // 0=x, 1=y, 2=z
  double origin,
  double invdelta,
  int nlayers,
  Box box,
  int* ichunk)                 // Output: chunk ID per atom
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    // Get position based on axis
    double pos;
    if (axis == 0) {
      pos = x[n];
    } else if (axis == 1) {
      pos = y[n];
    } else {
      pos = z[n];
    }

    // Compute bin index
    int bin_id = static_cast<int>((pos - origin) * invdelta);

    // Clamp to valid range [0, nlayers-1]
    if (bin_id < 0) bin_id = 0;
    if (bin_id >= nlayers) bin_id = nlayers - 1;

    ichunk[n] = bin_id;
  }
}

// GPU kernel to assign atoms to 2D bins
static __global__ void assign_atoms_to_bins_2d(
  int N,
  const double* x,
  const double* y,
  const double* z,
  int axis0,
  int axis1,
  double origin0,
  double origin1,
  double invdelta0,
  double invdelta1,
  int nlayers0,
  int nlayers1,
  int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    // Get positions for both axes
    double pos0, pos1;
    if (axis0 == 0) pos0 = x[n];
    else if (axis0 == 1) pos0 = y[n];
    else pos0 = z[n];

    if (axis1 == 0) pos1 = x[n];
    else if (axis1 == 1) pos1 = y[n];
    else pos1 = z[n];

    // Compute bin indices
    int bin_id0 = static_cast<int>((pos0 - origin0) * invdelta0);
    int bin_id1 = static_cast<int>((pos1 - origin1) * invdelta1);

    // Clamp to valid range
    if (bin_id0 < 0) bin_id0 = 0;
    if (bin_id0 >= nlayers0) bin_id0 = nlayers0 - 1;
    if (bin_id1 < 0) bin_id1 = 0;
    if (bin_id1 >= nlayers1) bin_id1 = nlayers1 - 1;

    // 2D chunk ID: row-major layout
    ichunk[n] = bin_id0 + nlayers0 * bin_id1;
  }
}

// GPU kernel to assign atoms to 3D bins
static __global__ void assign_atoms_to_bins_3d(
  int N,
  const double* x,
  const double* y,
  const double* z,
  double origin_x,
  double origin_y,
  double origin_z,
  double invdelta_x,
  double invdelta_y,
  double invdelta_z,
  int nlayers_x,
  int nlayers_y,
  int nlayers_z,
  int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    // Compute bin indices for all three dimensions
    int bin_x = static_cast<int>((x[n] - origin_x) * invdelta_x);
    int bin_y = static_cast<int>((y[n] - origin_y) * invdelta_y);
    int bin_z = static_cast<int>((z[n] - origin_z) * invdelta_z);

    // Clamp to valid range
    if (bin_x < 0) bin_x = 0;
    if (bin_x >= nlayers_x) bin_x = nlayers_x - 1;
    if (bin_y < 0) bin_y = 0;
    if (bin_y >= nlayers_y) bin_y = nlayers_y - 1;
    if (bin_z < 0) bin_z = 0;
    if (bin_z >= nlayers_z) bin_z = nlayers_z - 1;

    // 3D chunk ID: bin_x + nlayers_x * (bin_y + nlayers_y * bin_z)
    ichunk[n] = bin_x + nlayers_x * (bin_y + nlayers_y * bin_z);
  }
}

ComputeChunk::ComputeChunk(const char** param, int num_param, Box& box)
{
  property_name = "compute_chunk";

  // Initialize defaults
  dim_ = 0;
  nchunk_ = 0;
  ncoord_ = 0;
  compress_ = false;

  for (int i = 0; i < 3; i++) {
    axis_[i] = 0;
    origin_[i] = 0.0;
    delta_[i] = 1.0;
    invdelta_[i] = 1.0;
    nlayers_[i] = 1;
  }

  parse(param, num_param, box);
}

void ComputeChunk::parse(const char** param, int num_param, Box& box)
{
  if (num_param < 2) {
    PRINT_INPUT_ERROR("compute_chunk requires at least 1 argument.");
  }

  // Parse chunk style: bin/1d, bin/2d, bin/3d
  if (strcmp(param[1], "bin/1d") == 0) {
    dim_ = 1;
    ncoord_ = 1;

    // Syntax: compute_chunk bin/1d <dim> <origin> <delta>
    if (num_param != 5) {
      PRINT_INPUT_ERROR("compute_chunk bin/1d requires 3 arguments: dim origin delta");
    }

    // Parse axis (x, y, or z)
    if (strcmp(param[2], "x") == 0) {
      axis_[0] = 0;
    } else if (strcmp(param[2], "y") == 0) {
      axis_[0] = 1;
    } else if (strcmp(param[2], "z") == 0) {
      axis_[0] = 2;
    } else {
      PRINT_INPUT_ERROR("dim must be x, y, or z");
    }

    // Get box length along this axis
    double box_length;
    if (axis_[0] == 0) box_length = box.cpu_h[0];
    else if (axis_[0] == 1) box_length = box.cpu_h[4];
    else box_length = box.cpu_h[8];

    // Parse origin (default to lower = 0.0)
    if (strcmp(param[3], "lower") == 0) {
      origin_[0] = 0.0;
    } else if (strcmp(param[3], "center") == 0) {
      origin_[0] = -box_length * 0.5;
    } else if (strcmp(param[3], "upper") == 0) {
      origin_[0] = -box_length;
    } else {
      // Parse as coordinate value
      if (!is_valid_real(param[3], &origin_[0])) {
        PRINT_INPUT_ERROR("origin must be lower, center, upper, or a number");
      }
    }

    // Parse delta (bin width)
    if (!is_valid_real(param[4], &delta_[0])) {
      PRINT_INPUT_ERROR("delta must be a positive number");
    }
    if (delta_[0] <= 0.0) {
      PRINT_INPUT_ERROR("delta must be positive");
    }
    invdelta_[0] = 1.0 / delta_[0];

    // Auto-calculate nlayers from box size and delta
    nlayers_[0] = static_cast<int>(box_length / delta_[0]);
    if (nlayers_[0] <= 0) nlayers_[0] = 1;

    nchunk_ = nlayers_[0];

  } else if (strcmp(param[1], "bin/2d") == 0) {
    dim_ = 2;
    ncoord_ = 2;

    // Syntax: compute_chunk bin/2d <dim> <origin> <delta> <dim> <origin> <delta>
    if (num_param != 8) {
      PRINT_INPUT_ERROR("compute_chunk bin/2d requires 6 arguments: dim origin delta dim origin delta");
    }

    // Parse both axes
    double box_lengths[2];
    for (int i = 0; i < 2; i++) {
      int idx = 2 + i * 3;  // param indices: 2,3,4 and 5,6,7

      // Parse dim
      if (strcmp(param[idx], "x") == 0) {
        axis_[i] = 0;
        box_lengths[i] = box.cpu_h[0];
      } else if (strcmp(param[idx], "y") == 0) {
        axis_[i] = 1;
        box_lengths[i] = box.cpu_h[4];
      } else if (strcmp(param[idx], "z") == 0) {
        axis_[i] = 2;
        box_lengths[i] = box.cpu_h[8];
      } else {
        PRINT_INPUT_ERROR("dim must be x, y, or z");
      }

      // Parse origin
      if (strcmp(param[idx + 1], "lower") == 0) {
        origin_[i] = 0.0;
      } else if (strcmp(param[idx + 1], "center") == 0) {
        origin_[i] = -box_lengths[i] * 0.5;
      } else if (strcmp(param[idx + 1], "upper") == 0) {
        origin_[i] = -box_lengths[i];
      } else {
        if (!is_valid_real(param[idx + 1], &origin_[i])) {
          PRINT_INPUT_ERROR("origin must be lower, center, upper, or a number");
        }
      }

      // Parse delta
      if (!is_valid_real(param[idx + 2], &delta_[i])) {
        PRINT_INPUT_ERROR("delta must be a positive number");
      }
      if (delta_[i] <= 0.0) {
        PRINT_INPUT_ERROR("delta must be positive");
      }
      invdelta_[i] = 1.0 / delta_[i];

      // Auto-calculate nlayers
      nlayers_[i] = static_cast<int>(box_lengths[i] / delta_[i]);
      if (nlayers_[i] <= 0) nlayers_[i] = 1;
    }

    // Check axes are different
    if (axis_[0] == axis_[1]) {
      PRINT_INPUT_ERROR("bin/2d requires two different axes");
    }

    nchunk_ = nlayers_[0] * nlayers_[1];

  } else if (strcmp(param[1], "bin/3d") == 0) {
    dim_ = 3;
    ncoord_ = 3;

    // Syntax: compute_chunk bin/3d <dim> <origin> <delta> <dim> <origin> <delta> <dim> <origin> <delta>
    if (num_param != 11) {
      PRINT_INPUT_ERROR("compute_chunk bin/3d requires 9 arguments: dim origin delta (x3)");
    }

    // For 3D, parse all three axes
    double box_lengths[3] = {box.cpu_h[0], box.cpu_h[4], box.cpu_h[8]};

    for (int i = 0; i < 3; i++) {
      int idx = 2 + i * 3;  // param indices: 2,3,4 and 5,6,7 and 8,9,10

      // Parse dim
      if (strcmp(param[idx], "x") == 0) {
        axis_[i] = 0;
      } else if (strcmp(param[idx], "y") == 0) {
        axis_[i] = 1;
      } else if (strcmp(param[idx], "z") == 0) {
        axis_[i] = 2;
      } else {
        PRINT_INPUT_ERROR("dim must be x, y, or z");
      }

      // Parse origin
      if (strcmp(param[idx + 1], "lower") == 0) {
        origin_[i] = 0.0;
      } else if (strcmp(param[idx + 1], "center") == 0) {
        origin_[i] = -box_lengths[axis_[i]] * 0.5;
      } else if (strcmp(param[idx + 1], "upper") == 0) {
        origin_[i] = -box_lengths[axis_[i]];
      } else {
        if (!is_valid_real(param[idx + 1], &origin_[i])) {
          PRINT_INPUT_ERROR("origin must be lower, center, upper, or a number");
        }
      }

      // Parse delta
      if (!is_valid_real(param[idx + 2], &delta_[i])) {
        PRINT_INPUT_ERROR("delta must be a positive number");
      }
      if (delta_[i] <= 0.0) {
        PRINT_INPUT_ERROR("delta must be positive");
      }
      invdelta_[i] = 1.0 / delta_[i];

      // Auto-calculate nlayers
      nlayers_[i] = static_cast<int>(box_lengths[axis_[i]] / delta_[i]);
      if (nlayers_[i] <= 0) nlayers_[i] = 1;
    }

    nchunk_ = nlayers_[0] * nlayers_[1] * nlayers_[2];

  } else {
    PRINT_INPUT_ERROR("chunk style must be bin/1d, bin/2d, or bin/3d");
  }

  printf("compute_chunk: initialized %dD binning with %d chunks\n", dim_, nchunk_);
}

void ComputeChunk::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  // Allocate GPU memory for chunk assignments
  ichunk_.resize(atom.number_of_atoms);

  // Calculate chunk volumes and coordinates
  calculate_chunk_volumes(box);
  calculate_chunk_coords(box);

  printf("compute_chunk: preprocessed with %d atoms and %d chunks\n",
         atom.number_of_atoms, nchunk_);
}

void ComputeChunk::process(
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
  Force& force)
{
  // Assign atoms to chunks every timestep
  assign_chunks(atom, box);
}

void ComputeChunk::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  printf("compute_chunk: postprocessed\n");
}

void ComputeChunk::assign_chunks(const Atom& atom, const Box& box)
{
  const int N = atom.number_of_atoms;
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;

  // Get position data
  const double* x = atom.position_per_atom.data();
  const double* y = atom.position_per_atom.data() + N;
  const double* z = atom.position_per_atom.data() + 2 * N;

  if (dim_ == 1) {
    assign_atoms_to_bins_1d<<<grid_size, block_size>>>(
      N, x, y, z,
      axis_[0],
      origin_[0],
      invdelta_[0],
      nlayers_[0],
      box,
      ichunk_.data()
    );
    GPU_CHECK_KERNEL
  } else if (dim_ == 2) {
    assign_atoms_to_bins_2d<<<grid_size, block_size>>>(
      N, x, y, z,
      axis_[0], axis_[1],
      origin_[0], origin_[1],
      invdelta_[0], invdelta_[1],
      nlayers_[0], nlayers_[1],
      ichunk_.data()
    );
    GPU_CHECK_KERNEL
  } else if (dim_ == 3) {
    assign_atoms_to_bins_3d<<<grid_size, block_size>>>(
      N, x, y, z,
      origin_[0], origin_[1], origin_[2],
      invdelta_[0], invdelta_[1], invdelta_[2],
      nlayers_[0], nlayers_[1], nlayers_[2],
      ichunk_.data()
    );
    GPU_CHECK_KERNEL
  }
}

void ComputeChunk::calculate_chunk_volumes(const Box& box)
{
  chunk_volume_cpu_.resize(nchunk_);

  if (dim_ == 1) {
    // For 1D binning, volume is the bin width times the cross-sectional area
    double box_vol = box.get_volume();
    double length_along_axis;

    if (axis_[0] == 0) {
      length_along_axis = box.cpu_h[0];
    } else if (axis_[0] == 1) {
      length_along_axis = box.cpu_h[4];
    } else {
      length_along_axis = box.cpu_h[8];
    }

    double chunk_length = delta_[0];
    double chunk_volume = (box_vol / length_along_axis) * chunk_length;

    // All chunks have same volume for uniform 1D binning
    for (int i = 0; i < nchunk_; i++) {
      chunk_volume_cpu_[i] = chunk_volume;
    }
  } else if (dim_ == 2) {
    // For 2D binning, volume is chunk area times thickness in 3rd dimension
    double box_vol = box.get_volume();

    // Find which dimension is not binned
    int third_axis = 3 - axis_[0] - axis_[1]; // 0+1+2=3, so third = 3-sum
    double thickness;
    if (third_axis == 0) thickness = box.cpu_h[0];
    else if (third_axis == 1) thickness = box.cpu_h[4];
    else thickness = box.cpu_h[8];

    double chunk_area = delta_[0] * delta_[1];
    double chunk_volume = chunk_area * thickness;

    for (int i = 0; i < nchunk_; i++) {
      chunk_volume_cpu_[i] = chunk_volume;
    }
  } else if (dim_ == 3) {
    // For 3D binning, volume is just delta_x * delta_y * delta_z
    double chunk_volume = delta_[0] * delta_[1] * delta_[2];

    for (int i = 0; i < nchunk_; i++) {
      chunk_volume_cpu_[i] = chunk_volume;
    }
  }
}

void ComputeChunk::calculate_chunk_coords(const Box& box)
{
  chunk_coords_cpu_.resize(nchunk_ * ncoord_);

  if (dim_ == 1) {
    // For 1D, store the center coordinate of each bin
    for (int i = 0; i < nchunk_; i++) {
      chunk_coords_cpu_[i] = origin_[0] + (i + 0.5) * delta_[0];
    }
  } else if (dim_ == 2) {
    // For 2D, store both center coordinates
    int idx = 0;
    for (int j = 0; j < nlayers_[1]; j++) {
      for (int i = 0; i < nlayers_[0]; i++) {
        chunk_coords_cpu_[idx * 2 + 0] = origin_[0] + (i + 0.5) * delta_[0];
        chunk_coords_cpu_[idx * 2 + 1] = origin_[1] + (j + 0.5) * delta_[1];
        idx++;
      }
    }
  } else if (dim_ == 3) {
    // For 3D, store all three center coordinates
    int idx = 0;
    for (int k = 0; k < nlayers_[2]; k++) {
      for (int j = 0; j < nlayers_[1]; j++) {
        for (int i = 0; i < nlayers_[0]; i++) {
          chunk_coords_cpu_[idx * 3 + 0] = origin_[0] + (i + 0.5) * delta_[0];
          chunk_coords_cpu_[idx * 3 + 1] = origin_[1] + (j + 0.5) * delta_[1];
          chunk_coords_cpu_[idx * 3 + 2] = origin_[2] + (k + 0.5) * delta_[2];
          idx++;
        }
      }
    }
  }
}
