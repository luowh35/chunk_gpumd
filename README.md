# GPUMD Chunk Computation Feature - Installation Package

## Overview

This package adds LAMMPS-style chunk computation and time-averaging functionality to GPUMD.

**Features:**
- ✅ Spatial binning (1D/2D/3D Cartesian bins)
- ✅ Auto-calculated bin numbers from box size
- ✅ Time-averaged chunk properties
- ✅ Multiple properties: density, velocity, temperature, force, mass
- ✅ Three averaging modes: ONE, RUNNING, WINDOW
- ✅ GPU-accelerated implementation

---

## Quick Installation


```bash
git clone https://github.com/luowh35/chunk_gpumd.git
git clone https://github.com/brucefan1983/GPUMD.git
mv chunk_gpumd/ GPUMD/
cd GPUMD/chunk_gpumd/
chmod +x install.sh
sh ./install.sh
cd ../src
make
```

---



## Usage Examples

### Example 1: 1D Temperature Profile

Add to your `run.in` file:

```bash
# Divide box into bins along z-axis with 1.0 Å width
# Number of bins auto-calculated from box size
compute_chunk bin/1d z lower 1.0

# Sample every 10 steps, 5 samples per output, output every 50 steps
# Compute temperature and number density
ave_chunk 10 5 50 temp density/number file temp_profile.out
```

### Example 2: 2D Velocity Field with Running Average

```bash
# 2D grid in xy plane with 2.0 Å bins
compute_chunk bin/2d x lower 2.0 y lower 2.0

# Compute velocity components with cumulative averaging
ave_chunk 20 10 200 vx vy vz mode RUNNING file velocity_field.out
```

### Example 3: 3D Density Distribution with Window Average

```bash
# 3D grid with 1.0 Å bins
compute_chunk bin/3d x lower 1.0 y lower 1.0 z lower 1.0

# Compute density with moving window of 5 epochs
ave_chunk 100 10 1000 density/mass mass file density_3d.out mode WINDOW 5
```

---

## Command Syntax

### compute_chunk

Assigns atoms to spatial bins:

```bash
# 1D binning
compute_chunk bin/1d <dim> <origin> <delta>

# 2D binning
compute_chunk bin/2d <dim1> <origin1> <delta1> <dim2> <origin2> <delta2>

# 3D binning
compute_chunk bin/3d <dim1> <origin1> <delta1> <dim2> <origin2> <delta2> <dim3> <origin3> <delta3>
```

**Parameters:**
- `dim`: Axis (x, y, or z)
- `origin`: Starting point
  - `lower` - Box lower edge (default, = 0.0)
  - `center` - Box center
  - `upper` - Box upper edge
  - Numeric value - Explicit coordinate
- `delta`: Bin width (Angstrom)
- **Note:** Number of bins is auto-calculated

### ave_chunk

Computes time-averaged properties per chunk:

```bash
ave_chunk <nevery> <nrepeat> <nfreq> <property1> <property2> ... [mode <MODE>] [file <filename>]
```

**Timing parameters:**
- `nevery`: Sample interval
- `nrepeat`: Samples per output
- `nfreq`: Output interval (must be multiple of nevery)

**Properties:**
- `density/number` - Number density (atoms/volume)
- `density/mass` - Mass density
- `vx`, `vy`, `vz` - Velocity components
- `temp` - Temperature
- `fx`, `fy`, `fz` - Force components
- `mass` - Total mass

**Averaging modes:**
- `ONE` - Reset after each epoch (default)
- `RUNNING` - Cumulative averaging
- `WINDOW <n>` - Moving window of last n epochs

---

## Output Format

```
# ave_chunk output: timestep nchunk total_count [chunk_id coord(s) count property1 property2 ...]
<timestep> <nchunk> <total_count>
<chunk_id> <coord1> [coord2] [coord3] <count> <value1> <value2> ...
...
```

**Example output:**
```
# ave_chunk output: timestep nchunk total_count [chunk_id coord(s) count temp ]
50 10 1000.000000
0 0.500000 100.000000 2.9854321000e+02
1 1.500000 100.000000 3.0512345000e+02
2 2.500000 100.000000 3.1278901000e+02
...
```

---

## Verification

After installation, verify:

```bash
# Check files exist
ls ../src/measure/compute_chunk.*
ls ../src/measure/ave_chunk.*

# Check registration in run.cu
grep compute_chunk ../src/main_gpumd/run.cu
grep ave_chunk ../src/main_gpumd/run.cu

# Check executable built
ls ../src/gpumd
```

---

## Rollback

If you need to uninstall:

```bash
# Restore original run.cu from backup
cp ../.chunk_install_backup_*/run.cu.bak ../src/main_gpumd/run.cu

# Remove new files
rm ../src/measure/compute_chunk.*
rm ../src/measure/ave_chunk.*

# Rebuild
cd ../src
make clean
make gpumd
```

---

## Troubleshooting

### Installation Issues

**"GPUMD src directory not found"**
- Ensure package is placed in GPUMD root directory
- Directory structure should be: `GPUMD/chunk_feature_package/`

**"Permission denied"**
- Make script executable: `chmod +x install.sh`

**"Files already exist"**
- Script will prompt for overwrite confirmation
- Previous installation detected

### Compilation Issues

**K_B redefined warning**
- Safe to ignore (minor warning)
- Doesn't affect functionality

**Undefined identifier errors**
- Ensure all 4 files copied correctly
- Check `src/measure/` directory

### Runtime Issues

**"ave_chunk requires compute_chunk"**
- Define `compute_chunk` before `ave_chunk` in run.in

**"nfreq must be a multiple of nevery"**
- Check: `nfreq % nevery == 0`

---

## Documentation

**Detailed guides in `docs/` directory:**

- **CHUNK_IMPLEMENTATION.md** - Complete technical documentation
  - Implementation details
  - Algorithm descriptions
  - GPU kernel design
  - Output format specification

- **INSTALLATION_GUIDE.md** - Comprehensive installation instructions
  - Manual installation steps
  - Testing procedures
  - Advanced usage examples

**Examples in `examples/` directory:**

- **chunk_test_example.in** - Ready-to-use example configurations

---

## Version Information

- **Created:** 2025-12-11
- **GPUMD Version:** Compatible with recent versions using Property base class
- **Tested on:** Commit f35a51f6
- **CUDA:** Required for GPU acceleration

---

## Support

For issues or questions:
1. Check `docs/INSTALLATION_GUIDE.md` for troubleshooting
2. Review `examples/chunk_test_example.in` for usage patterns
3. Verify installation using verification steps above

---

## License

Follows GPUMD licensing (GPL v3).

Copyright 2017 Zheyong Fan and GPUMD development team.
