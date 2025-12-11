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
Ave/chunk: Time-average properties per chunk
------------------------------------------------------------------------------*/

#include "ave_chunk.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

#define DIM 3
#define K_B 8.617333262e-5  // Boltzmann constant in eV/K

// GPU kernel: Count atoms per chunk
static __global__ void count_atoms_per_chunk(
  int N,
  const int* ichunk,
  int nchunk,
  int* count)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int chunk_id = ichunk[n];
    if (chunk_id >= 0 && chunk_id < nchunk) {
      atomicAdd(&count[chunk_id], 1);
    }
  }
}

// GPU kernel: Sum scalar property per chunk
static __global__ void sum_property_per_chunk(
  int N,
  const int* ichunk,
  const double* property,
  int nchunk,
  int value_offset,
  int nvalues,
  double* sum_per_chunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int chunk_id = ichunk[n];
    if (chunk_id >= 0 && chunk_id < nchunk) {
      atomicAdd(&sum_per_chunk[chunk_id * nvalues + value_offset], property[n]);
    }
  }
}

// GPU kernel: Sum velocity per chunk
static __global__ void sum_velocity_per_chunk(
  int N,
  const int* ichunk,
  const double* vx,
  const double* vy,
  const double* vz,
  int nchunk,
  int value_offset,
  int nvalues,
  double* sum_per_chunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int chunk_id = ichunk[n];
    if (chunk_id >= 0 && chunk_id < nchunk) {
      atomicAdd(&sum_per_chunk[chunk_id * nvalues + value_offset + 0], vx[n]);
      atomicAdd(&sum_per_chunk[chunk_id * nvalues + value_offset + 1], vy[n]);
      atomicAdd(&sum_per_chunk[chunk_id * nvalues + value_offset + 2], vz[n]);
    }
  }
}

// GPU kernel: Compute temperature per chunk
static __global__ void compute_temp_per_chunk(
  int N,
  const int* ichunk,
  const double* mass,
  const double* vx,
  const double* vy,
  const double* vz,
  int nchunk,
  int value_offset,
  int nvalues,
  double* ke_sum)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int chunk_id = ichunk[n];
    if (chunk_id >= 0 && chunk_id < nchunk) {
      double v2 = vx[n] * vx[n] + vy[n] * vy[n] + vz[n] * vz[n];
      double ke = 0.5 * mass[n] * v2;
      atomicAdd(&ke_sum[chunk_id * nvalues + value_offset], ke);
    }
  }
}

AveChunk::AveChunk(const char** param, int num_param, Measure& measure)
{
  property_name = "ave_chunk";

  // Initialize defaults
  nevery_ = 1;
  nrepeat_ = 1;
  nfreq_ = 1;
  irepeat_ = 0;
  ave_mode_ = ONE;
  nwindow_ = 1;
  nvalues_ = 0;
  compute_chunk_ = nullptr;
  nchunk_ = 0;
  iwindow_ = 0;

  // Initialize flags
  compute_density_number_ = false;
  compute_density_mass_ = false;
  compute_vx_ = compute_vy_ = compute_vz_ = false;
  compute_temp_ = false;
  compute_fx_ = compute_fy_ = compute_fz_ = false;
  compute_mass_ = false;

  filename_ = "ave_chunk.out";
  fid_ = nullptr;

  parse(param, num_param, measure);
}

void AveChunk::parse(const char** param, int num_param, Measure& measure)
{
  if (num_param < 5) {
    PRINT_INPUT_ERROR("ave_chunk requires at least 4 arguments: nevery nrepeat nfreq property...");
  }

  // Parse timing parameters
  if (!is_valid_int(param[1], &nevery_)) {
    PRINT_INPUT_ERROR("nevery must be a positive integer");
  }
  if (!is_valid_int(param[2], &nrepeat_)) {
    PRINT_INPUT_ERROR("nrepeat must be a positive integer");
  }
  if (!is_valid_int(param[3], &nfreq_)) {
    PRINT_INPUT_ERROR("nfreq must be a positive integer");
  }

  if (nevery_ <= 0 || nrepeat_ <= 0 || nfreq_ <= 0) {
    PRINT_INPUT_ERROR("nevery, nrepeat, nfreq must all be positive");
  }

  if (nfreq_ % nevery_ != 0) {
    PRINT_INPUT_ERROR("nfreq must be a multiple of nevery");
  }

  // Parse properties and optional keywords
  int i = 4;
  while (i < num_param) {
    if (strcmp(param[i], "mode") == 0) {
      if (i + 1 >= num_param) {
        PRINT_INPUT_ERROR("mode keyword requires an argument");
      }
      if (strcmp(param[i + 1], "ONE") == 0) {
        ave_mode_ = ONE;
      } else if (strcmp(param[i + 1], "RUNNING") == 0) {
        ave_mode_ = RUNNING;
      } else if (strcmp(param[i + 1], "WINDOW") == 0) {
        ave_mode_ = WINDOW;
        if (i + 2 >= num_param) {
          PRINT_INPUT_ERROR("WINDOW mode requires nwindow argument");
        }
        if (!is_valid_int(param[i + 2], &nwindow_)) {
          PRINT_INPUT_ERROR("nwindow must be a positive integer");
        }
        if (nwindow_ <= 0) {
          PRINT_INPUT_ERROR("nwindow must be positive");
        }
        i += 1;  // Skip nwindow
      } else {
        PRINT_INPUT_ERROR("mode must be ONE, RUNNING, or WINDOW");
      }
      i += 2;
    } else if (strcmp(param[i], "file") == 0) {
      if (i + 1 >= num_param) {
        PRINT_INPUT_ERROR("file keyword requires a filename");
      }
      filename_ = std::string(param[i + 1]);
      i += 2;
    } else {
      // This is a property to compute
      parse_property(std::string(param[i]));
      i++;
    }
  }

  if (nvalues_ == 0) {
    PRINT_INPUT_ERROR("ave_chunk requires at least one property to compute");
  }

  // Find ComputeChunk instance
  for (auto& prop : measure.properties) {
    if (prop->property_name == "compute_chunk") {
      compute_chunk_ = static_cast<ComputeChunk*>(prop.get());
      break;
    }
  }

  if (compute_chunk_ == nullptr) {
    PRINT_INPUT_ERROR("ave_chunk requires compute_chunk to be defined first");
  }

  printf("ave_chunk: initialized with nevery=%d nrepeat=%d nfreq=%d\n",
         nevery_, nrepeat_, nfreq_);
  printf("ave_chunk: computing %d properties, output to %s\n",
         nvalues_, filename_.c_str());
}

void AveChunk::parse_property(const std::string& prop)
{
  property_names_.push_back(prop);

  if (prop == "density/number") {
    compute_density_number_ = true;
    nvalues_ += 1;
  } else if (prop == "density/mass") {
    compute_density_mass_ = true;
    nvalues_ += 1;
  } else if (prop == "vx") {
    compute_vx_ = true;
    nvalues_ += 1;
  } else if (prop == "vy") {
    compute_vy_ = true;
    nvalues_ += 1;
  } else if (prop == "vz") {
    compute_vz_ = true;
    nvalues_ += 1;
  } else if (prop == "temp") {
    compute_temp_ = true;
    nvalues_ += 1;
  } else if (prop == "fx") {
    compute_fx_ = true;
    nvalues_ += 1;
  } else if (prop == "fy") {
    compute_fy_ = true;
    nvalues_ += 1;
  } else if (prop == "fz") {
    compute_fz_ = true;
    nvalues_ += 1;
  } else if (prop == "mass") {
    compute_mass_ = true;
    nvalues_ += 1;
  } else {
    printf("Warning: unknown property '%s' will be ignored\n", prop.c_str());
  }
}

void AveChunk::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  // Get nchunk from compute_chunk
  nchunk_ = compute_chunk_->get_nchunk();

  if (nchunk_ <= 0) {
    PRINT_INPUT_ERROR("ave_chunk: nchunk must be positive");
  }

  // Allocate GPU arrays
  allocate_arrays();

  // Open output file
  fid_ = my_fopen(filename_.c_str(), "a");

  // Write header
  fprintf(fid_, "# ave_chunk output: timestep nchunk total_count ");
  fprintf(fid_, "[chunk_id coord(s) count ");
  for (const auto& prop : property_names_) {
    fprintf(fid_, "%s ", prop.c_str());
  }
  fprintf(fid_, "]\n");
  fflush(fid_);

  printf("ave_chunk: preprocessed with %d chunks and %d properties\n",
         nchunk_, nvalues_);
}

void AveChunk::allocate_arrays()
{
  // GPU arrays
  count_one_.resize(nchunk_, 0);
  values_one_.resize(nchunk_ * nvalues_, 0.0);
  count_many_.resize(nchunk_, 0);
  values_many_.resize(nchunk_ * nvalues_, 0.0);

  // CPU arrays for RUNNING/WINDOW
  if (ave_mode_ == RUNNING || ave_mode_ == WINDOW) {
    count_total_.resize(nchunk_, 0);
    values_total_.resize(nchunk_ * nvalues_, 0.0);
  }

  if (ave_mode_ == WINDOW) {
    count_list_.resize(nwindow_, std::vector<int>(nchunk_, 0));
    values_list_.resize(nwindow_, std::vector<double>(nchunk_ * nvalues_, 0.0));
  }
}

void AveChunk::process(
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
  // Check if this is a sampling step
  if ((step + 1) % nevery_ != 0) {
    return;
  }

  // Initialize at start of averaging period
  if (irepeat_ == 0) {
    count_many_.fill(0);
    values_many_.fill(0.0);
  }

  // Sample current timestep
  sample(atom, box);

  // Accumulate into many arrays
  const int block_size = 128;
  const int N = atom.number_of_atoms;
  const int grid_size = (N - 1) / block_size + 1;

  // Add count_one to count_many
  const int nchunk = nchunk_;
  GPU_Vector<int> temp_count(nchunk);
  temp_count.copy_from_device(count_one_.data(), nchunk);
  std::vector<int> cpu_count_one(nchunk);
  temp_count.copy_to_host(cpu_count_one.data(), nchunk);

  std::vector<int> cpu_count_many(nchunk);
  count_many_.copy_to_host(cpu_count_many.data(), nchunk);

  for (int i = 0; i < nchunk; i++) {
    cpu_count_many[i] += cpu_count_one[i];
  }
  count_many_.copy_from_host(cpu_count_many.data(), nchunk);

  // Add values_one to values_many (done on GPU would be better, but this works)
  std::vector<double> cpu_values_one(nchunk * nvalues_);
  std::vector<double> cpu_values_many(nchunk * nvalues_);
  values_one_.copy_to_host(cpu_values_one.data(), nchunk * nvalues_);
  values_many_.copy_to_host(cpu_values_many.data(), nchunk * nvalues_);

  for (int i = 0; i < nchunk * nvalues_; i++) {
    cpu_values_many[i] += cpu_values_one[i];
  }
  values_many_.copy_from_host(cpu_values_many.data(), nchunk * nvalues_);

  irepeat_++;

  // Check if we've completed nrepeat samples
  if (irepeat_ == nrepeat_) {
    compute_final_averages();
    write_output(step + 1);
    irepeat_ = 0;
  }
}

void AveChunk::sample(const Atom& atom, const Box& box)
{
  const int N = atom.number_of_atoms;
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;

  // Zero arrays
  count_one_.fill(0);
  values_one_.fill(0.0);

  // Get ichunk from compute_chunk
  const GPU_Vector<int>& ichunk = compute_chunk_->get_ichunk();

  // Count atoms per chunk
  count_atoms_per_chunk<<<grid_size, block_size>>>(
    N, ichunk.data(), nchunk_, count_one_.data()
  );
  GPU_CHECK_KERNEL

  // Get atom data
  const double* mass = atom.mass.data();
  const double* vx = atom.velocity_per_atom.data();
  const double* vy = atom.velocity_per_atom.data() + N;
  const double* vz = atom.velocity_per_atom.data() + 2 * N;
  const double* fx = atom.force_per_atom.data();
  const double* fy = atom.force_per_atom.data() + N;
  const double* fz = atom.force_per_atom.data() + 2 * N;

  int value_idx = 0;

  // Compute each property
  for (const auto& prop : property_names_) {
    if (prop == "density/number" || prop == "density/mass") {
      // Will be computed in compute_final_averages
      if (prop == "density/mass") {
        // Sum mass per chunk
        sum_property_per_chunk<<<grid_size, block_size>>>(
          N, ichunk.data(), mass, nchunk_, value_idx, nvalues_, values_one_.data()
        );
        GPU_CHECK_KERNEL
      }
      value_idx++;
    } else if (prop == "vx" || prop == "vy" || prop == "vz") {
      const double* v = (prop == "vx") ? vx : ((prop == "vy") ? vy : vz);
      sum_property_per_chunk<<<grid_size, block_size>>>(
        N, ichunk.data(), v, nchunk_, value_idx, nvalues_, values_one_.data()
      );
      GPU_CHECK_KERNEL
      value_idx++;
    } else if (prop == "temp") {
      // Sum kinetic energy per chunk
      compute_temp_per_chunk<<<grid_size, block_size>>>(
        N, ichunk.data(), mass, vx, vy, vz, nchunk_, value_idx, nvalues_, values_one_.data()
      );
      GPU_CHECK_KERNEL
      value_idx++;
    } else if (prop == "fx" || prop == "fy" || prop == "fz") {
      const double* f = (prop == "fx") ? fx : ((prop == "fy") ? fy : fz);
      sum_property_per_chunk<<<grid_size, block_size>>>(
        N, ichunk.data(), f, nchunk_, value_idx, nvalues_, values_one_.data()
      );
      GPU_CHECK_KERNEL
      value_idx++;
    } else if (prop == "mass") {
      sum_property_per_chunk<<<grid_size, block_size>>>(
        N, ichunk.data(), mass, nchunk_, value_idx, nvalues_, values_one_.data()
      );
      GPU_CHECK_KERNEL
      value_idx++;
    }
  }
}

void AveChunk::compute_final_averages()
{
  // Copy GPU data to CPU
  std::vector<int> cpu_count_int(nchunk_);
  std::vector<double> cpu_values(nchunk_ * nvalues_);

  count_many_.copy_to_host(cpu_count_int.data(), nchunk_);
  values_many_.copy_to_host(cpu_values.data(), nchunk_ * nvalues_);

  // Get chunk volumes
  const std::vector<double>& volumes = compute_chunk_->get_chunk_volume();

  // Convert count to double and store averaged values
  std::vector<double> cpu_count_avg(nchunk_);

  // Divide by nrepeat and normalize
  for (int chunk = 0; chunk < nchunk_; chunk++) {
    // Average count over nrepeat samples
    double count = static_cast<double>(cpu_count_int[chunk]) / nrepeat_;
    cpu_count_avg[chunk] = count;

    int value_idx = 0;
    for (const auto& prop : property_names_) {
      double& val = cpu_values[chunk * nvalues_ + value_idx];
      val /= nrepeat_;  // Average over samples

      if (prop == "density/number") {
        // Number density = count / volume
        val = count / volumes[chunk];
      } else if (prop == "density/mass") {
        // Mass density = total_mass / volume
        val = val / volumes[chunk];
      } else if (prop == "vx" || prop == "vy" || prop == "vz" ||
                 prop == "fx" || prop == "fy" || prop == "fz") {
        // Average per atom
        if (count > 0) {
          val /= count;
        }
      } else if (prop == "temp") {
        // Temperature = 2*KE / (k_B * DOF)
        // DOF = 3 * N_atoms
        if (count > 0) {
          double dof = DIM * count;
          val = 2.0 * val / (K_B * dof);
        }
      } else if (prop == "mass") {
        // Total mass (already summed)
      }

      value_idx++;
    }
  }

  // Update RUNNING/WINDOW totals
  if (ave_mode_ == RUNNING) {
    for (int i = 0; i < nchunk_; i++) {
      count_total_[i] += cpu_count_int[i];
    }
    for (int i = 0; i < nchunk_ * nvalues_; i++) {
      values_total_[i] += cpu_values[i];
    }
  } else if (ave_mode_ == WINDOW) {
    // Subtract oldest window entry
    for (int i = 0; i < nchunk_; i++) {
      count_total_[i] -= count_list_[iwindow_][i];
      count_total_[i] += cpu_count_int[i];
      count_list_[iwindow_][i] = cpu_count_int[i];
    }
    for (int i = 0; i < nchunk_ * nvalues_; i++) {
      values_total_[i] -= values_list_[iwindow_][i];
      values_total_[i] += cpu_values[i];
      values_list_[iwindow_][i] = cpu_values[i];
    }
    iwindow_ = (iwindow_ + 1) % nwindow_;
  }

  // Store final values back (for ONE mode, or as current values)
  values_many_.copy_from_host(cpu_values.data(), nchunk_ * nvalues_);

  // Store averaged count for output
  count_avg_output_ = cpu_count_avg;
}

void AveChunk::write_output(int step)
{
  // Get chunk coordinates
  const std::vector<double>& coords = compute_chunk_->get_chunk_coords();
  int ncoord = compute_chunk_->get_ncoord();

  // Get count and values for output
  std::vector<double> output_count;
  std::vector<double> cpu_values(nchunk_ * nvalues_);

  if (ave_mode_ == ONE) {
    // For ONE mode, use the averaged count
    output_count = count_avg_output_;
    values_many_.copy_to_host(cpu_values.data(), nchunk_ * nvalues_);
  } else {
    // For RUNNING/WINDOW modes, compute averaged count from total
    output_count.resize(nchunk_);
    for (int i = 0; i < nchunk_; i++) {
      output_count[i] = static_cast<double>(count_total_[i]) / nrepeat_;
    }
    cpu_values = values_total_;
  }

  // Calculate total count
  double total_count = 0.0;
  for (int i = 0; i < nchunk_; i++) {
    total_count += output_count[i];
  }

  // Write header line
  fprintf(fid_, "%d %d %.6f\n", step, nchunk_, total_count);

  // Write per-chunk data
  for (int chunk = 0; chunk < nchunk_; chunk++) {
    fprintf(fid_, "%d ", chunk);

    // Write coordinates
    for (int c = 0; c < ncoord; c++) {
      fprintf(fid_, "%.6f ", coords[chunk * ncoord + c]);
    }

    // Write count (as double)
    fprintf(fid_, "%.6f ", output_count[chunk]);

    // Write property values
    for (int v = 0; v < nvalues_; v++) {
      fprintf(fid_, "%.10e ", cpu_values[chunk * nvalues_ + v]);
    }
    fprintf(fid_, "\n");
  }

  fflush(fid_);
}

void AveChunk::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (fid_) {
    fclose(fid_);
  }
  printf("ave_chunk: postprocessed\n");
}
