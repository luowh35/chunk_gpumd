# GPUMD Chunk Computation Feature - Installation Guide

## Summary of Changes

### New Files Created (4 files)

1. **src/measure/compute_chunk.cuh** - Header for spatial binning
   - Defines ComputeChunk class that assigns atoms to spatial bins
   - Supports 1D/2D/3D Cartesian binning
   - Provides interface for chunk access (ichunk array, volumes, coordinates)

2. **src/measure/compute_chunk.cu** - Implementation of spatial binning
   - GPU kernels for 1D/2D/3D bin assignment
   - Auto-calculates number of bins from box size and bin width
   - Handles PBC and different origin specifications (lower/center/upper)

3. **src/measure/ave_chunk.cuh** - Header for chunk averaging
   - Defines AveChunk class that computes time-averaged properties per chunk
   - Supports THREE averaging modes: ONE, RUNNING, WINDOW
   - Manages multiple property types (density, velocity, temperature, force, mass)

4. **src/measure/ave_chunk.cu** - Implementation of chunk averaging
   - GPU kernels for per-chunk property accumulation
   - Time averaging with nevery/nrepeat/nfreq control
   - Proper normalization for density and temperature

### Modified Files (1 file)

**src/main_gpumd/run.cu** - Command registration
- Added includes for compute_chunk.cuh and ave_chunk.cuh (after line ~30)
- Added command registration blocks for "compute_chunk" and "ave_chunk" (after line ~515)

### Total Lines of Code Added
- New files: ~1800 lines
- Modified run.cu: ~14 lines added

---

## Installation Instructions

### Quick Install (Recommended)

Run the installation script from the GPUMD root directory:

```bash
cd /path/to/GPUMD
./install_chunk_feature.sh
```

Or specify GPUMD path explicitly:

```bash
./install_chunk_feature.sh /path/to/GPUMD
```

### What the Script Does

1. **Verifies** directory structure exists
2. **Creates backup** at `.chunk_install_backup_YYYYMMDD_HHMMSS/`
3. **Checks** for existing files (prompts for overwrite)
4. **Copies** 4 new files to `src/measure/`
5. **Modifies** `src/main_gpumd/run.cu` automatically using sed
6. **Verifies** installation success
7. **Provides** rollback instructions if needed

### After Installation

1. Rebuild GPUMD:
   ```bash
   cd src
   make clean
   make gpumd
   ```

2. Verify installation:
   ```bash
   ls src/measure/compute_chunk.*
   ls src/measure/ave_chunk.*
   grep compute_chunk src/main_gpumd/run.cu
   ```

---

## Manual Installation (Alternative)

If you prefer manual installation:

### Step 1: Copy New Files

```bash
cp src/measure/compute_chunk.cuh /path/to/original_gpumd/src/measure/
cp src/measure/compute_chunk.cu /path/to/original_gpumd/src/measure/
cp src/measure/ave_chunk.cuh /path/to/original_gpumd/src/measure/
cp src/measure/ave_chunk.cu /path/to/original_gpumd/src/measure/
```

### Step 2: Modify run.cu

Edit `src/main_gpumd/run.cu`:

**Add includes** (after existing measure includes around line 30):
```cpp
#include "measure/ave_chunk.cuh"
#include "measure/compute_chunk.cuh"
```

**Add command registration** (inside parse_one_keyword function, after compute_rdf block around line 515):
```cpp
  } else if (strcmp(param[0], "compute_chunk") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new ComputeChunk(param, num_param, box));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "ave_chunk") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new AveChunk(param, num_param, measure));
    measure.properties.emplace_back(std::move(property));
```

### Step 3: Rebuild

```bash
cd src
make clean
make gpumd
```

---

## Usage Examples

### Example 1: 1D Temperature Profile

```bash
# In run.in file:
compute_chunk bin/1d z lower 1.0
ave_chunk 10 5 50 temp density/number file temp_profile.out
```

This will:
- Divide simulation box into bins along z-axis with 1.0 Angstrom width
- Number of bins is auto-calculated from box size
- Sample every 10 steps, 5 samples per output, output every 50 steps
- Compute temperature and number density per bin
- Output to temp_profile.out

### Example 2: 2D Velocity Field (Running Average)

```bash
compute_chunk bin/2d x lower 2.0 y lower 2.0
ave_chunk 20 10 200 vx vy vz mode RUNNING file velocity_field.out
```

### Example 3: 3D Density Distribution (Window Average)

```bash
compute_chunk bin/3d x lower 1.0 y lower 1.0 z lower 1.0
ave_chunk 100 10 1000 density/mass mass file density_3d.out mode WINDOW 5
```

---

## Rollback Instructions

If you need to restore the original version:

1. **From automatic backup:**
   ```bash
   cp .chunk_install_backup_*/run.cu.bak src/main_gpumd/run.cu
   ```

2. **Remove new files:**
   ```bash
   rm src/measure/compute_chunk.*
   rm src/measure/ave_chunk.*
   ```

3. **Rebuild:**
   ```bash
   cd src
   make clean
   make gpumd
   ```

---

## Testing

After installation, test with the provided example:

```bash
# Use chunk_test_example.in as a template
# Add to your run.in file:
compute_chunk bin/1d z lower 1.0
ave_chunk 10 5 50 temp density/number file chunk.out
```

Run your simulation and check for `chunk.out` file.

---

## Troubleshooting

### Installation Script Fails

- **Permission denied**: Run `chmod +x install_chunk_feature.sh`
- **Directory not found**: Ensure you're running from GPUMD root with `src/` subdirectory
- **Files already exist**: Script will prompt for overwrite confirmation

### Compilation Errors

- **K_B redefined warning**: Safe to ignore (minor warning, doesn't affect functionality)
- **Undefined identifier**: Make sure all 4 new files are copied correctly

### Runtime Errors

- **"ave_chunk requires compute_chunk"**: Define `compute_chunk` before `ave_chunk` in run.in
- **"nfreq must be a multiple of nevery"**: Check timing parameters satisfy nfreq % nevery == 0

---

## Key Improvements Over LAMMPS

1. **Auto-calculated bins**: No need to specify nlayers explicitly
2. **GPU acceleration**: All chunk assignment and property calculations on GPU
3. **Unified interface**: Follows GPUMD's Property-based architecture
4. **Double-precision counts**: Time-averaged atom counts (not just integers)

---

## Support

For issues or questions:
- Check CHUNK_IMPLEMENTATION.md for detailed documentation
- Review chunk_test_example.in for usage examples
- Verify installation using the verification steps above

---

## Version Compatibility

Tested on GPUMD version with:
- Commit: f35a51f6 (Merge pull request #1259)
- CUDA compilation support
- Existing measure framework (Property base class)

Should be compatible with recent GPUMD versions that follow the same architecture.
