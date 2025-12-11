# GPUMD Chunk Feature - Quick Start Guide

## Installation (3 Simple Steps)

### Method 1: Using tar.gz archive

```bash
# 1. Extract to GPUMD directory
cd /path/to/GPUMD
tar -xzf gpumd_chunk_feature.tar.gz

# 2. Run installation
cd chunk_feature_package
sh ./install.sh

# 3. Build GPUMD
cd ../src
make clean
make gpumd
```

### Method 2: Using zip archive

```bash
# 1. Extract to GPUMD directory
cd /path/to/GPUMD
unzip gpumd_chunk_feature.zip

# 2. Run installation
cd chunk_feature_package
sh ./install.sh

# 3. Build GPUMD
cd ../src
make clean
make gpumd
```

### Method 3: Copy directory directly

```bash
# 1. Copy package directory to GPUMD
cp -r chunk_feature_package /path/to/GPUMD/

# 2. Run installation
cd /path/to/GPUMD/chunk_feature_package
sh ./install.sh

# 3. Build GPUMD
cd ../src
make clean
make gpumd
```

---

## Quick Test

After installation, add these lines to your `run.in`:

```bash
# Divide box into 1D bins along z-axis
compute_chunk bin/1d z lower 1.0

# Compute temperature profile
ave_chunk 10 5 50 temp density/number file chunk.out
```

Run GPUMD and check for `chunk.out` file.

---

## What Gets Installed

- 4 new files in `src/measure/`:
  - compute_chunk.cuh, compute_chunk.cu
  - ave_chunk.cuh, ave_chunk.cu

- Modified `src/main_gpumd/run.cu`:
  - Added header includes
  - Registered two new commands

- Automatic backup created in:
  - `.chunk_install_backup_YYYYMMDD_HHMMSS/`

---

## Complete Documentation

- **README.md** - Overview and quick reference
- **docs/CHUNK_IMPLEMENTATION.md** - Technical details
- **docs/INSTALLATION_GUIDE.md** - Comprehensive guide
- **examples/chunk_test_example.in** - Usage examples

---

## Command Reference

### compute_chunk - Spatial Binning

```bash
# 1D
compute_chunk bin/1d <dim> <origin> <delta>

# 2D
compute_chunk bin/2d <dim1> <origin1> <delta1> <dim2> <origin2> <delta2>

# 3D (all three dimensions)
compute_chunk bin/3d <dim1> <origin1> <delta1> <dim2> <origin2> <delta2> <dim3> <origin3> <delta3>
```

**dim**: x, y, or z
**origin**: lower, center, upper, or numeric value
**delta**: Bin width (Angstrom)

### ave_chunk - Time-Averaged Properties

```bash
ave_chunk <nevery> <nrepeat> <nfreq> <property> ... [mode <MODE>] [file <filename>]
```

**Properties**: density/number, density/mass, vx, vy, vz, temp, fx, fy, fz, mass
**Modes**: ONE (default), RUNNING, WINDOW <n>

---

## Examples

### 1D Temperature Profile
```bash
compute_chunk bin/1d z lower 1.0
ave_chunk 10 5 50 temp density/number file temp_profile.out
```

### 2D Velocity Field
```bash
compute_chunk bin/2d x lower 2.0 y lower 2.0
ave_chunk 20 10 200 vx vy vz mode RUNNING file velocity.out
```

### 3D Density Distribution
```bash
compute_chunk bin/3d x lower 1.0 y lower 1.0 z lower 1.0
ave_chunk 100 10 1000 density/mass file density.out mode WINDOW 5
```

---

## Troubleshooting

**Installation fails:**
- Ensure package is in GPUMD root directory
- Check: `GPUMD/chunk_feature_package/install.sh` exists

**Compilation fails:**
- K_B warning is safe to ignore
- Check all 4 new files copied correctly

**Runtime error:**
- Define `compute_chunk` before `ave_chunk` in run.in
- Verify: `nfreq % nevery == 0`

**Rollback:**
```bash
cp .chunk_install_backup_*/run.cu.bak src/main_gpumd/run.cu
rm src/measure/compute_chunk.* src/measure/ave_chunk.*
cd src && make clean && make gpumd
```

---

## Package Contents

```
chunk_feature_package/
├── install.sh              ← Run this to install
├── README.md               ← Full documentation
├── QUICK_START.md          ← This file
├── src/measure/            ← New source files (4 files)
├── examples/               ← Usage examples
└── docs/                   ← Technical documentation
```

---

## Support

Read the detailed guides:
- `README.md` for overview
- `docs/INSTALLATION_GUIDE.md` for troubleshooting
- `docs/CHUNK_IMPLEMENTATION.md` for technical details
- `examples/chunk_test_example.in` for more examples
