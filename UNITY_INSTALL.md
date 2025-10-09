# Installing gulls on Unity HPC

This guide covers building gulls on OSU's Unity cluster.

## Prerequisites

You'll need access to the Numerical Recipes files (`random.cpp`, `random.h`, `zroots2.cpp`, `zroots2.h`) which are under copyright and not included in this repository. Contact your lab if you need access to these files.

## Installation Steps

### 1. Get an interactive session

Don't install packages on the login node:

```bash
sinteractive
```

### 2. Set up the conda environment

Load mamba and create the environment:

```bash
module load mamba
mamba env create -f unity.yml
conda activate gulls
```

This installs:
- CMake
- GSL (GNU Scientific Library)
- CFITSIO
- C++/Fortran compilers

### 3. Clone the repository

```bash
git clone https://github.com/gulls-microlensing/gulls.git
cd gulls
git checkout dev
```

### 4. Add required files

**ESPL.tbl** - Download from VBMicrolensing:

```bash
curl -L -o src/ESPL.tbl https://raw.githubusercontent.com/valboz/VBMicrolensing/main/VBMicrolensing/data/ESPL.tbl
```

**Numerical Recipes files** - Copy from lab shared storage:

```bash
# Copy from wherever your lab stores these licensed files
cp /path/to/random.cpp src/classes/
cp /path/to/random.h src/headers/
cp /path/to/zroots2.cpp src/classes/
cp /path/to/zroots2.h src/headers/
```

### 5. Build

Make sure you're in the gulls conda environment, then build:

```bash
cmake -S . -B build
cmake --build build
```

**Note:** If you get `libssl.so.10` error, unload the system cmake:
```bash
module unload cmake
cmake -S . -B build
```

Executables will be in `./bin/`:
- `gulls_std.x`
- `gulls_croin.x`
- `gullsFish.x`

## Running gulls

Activate the environment whenever you want to run gulls:

```bash
conda activate gulls
./bin/gulls_std.x <parameter_file> [options]
```

### In SLURM scripts

```bash
#!/bin/bash
#SBATCH --job-name=gulls
#SBATCH --time=24:00:00
#SBATCH --ntasks=1

source ~/.bashrc
module load mamba
conda activate gulls

cd $SLURM_SUBMIT_DIR
./bin/gulls_std.x my_params.prm
```

## Troubleshooting

### CMake library errors

Make sure you're in the gulls conda environment and unload conflicting modules:

```bash
module unload cmake
conda activate gulls
```

The conda environment provides all necessary dependencies.

### Missing files during build

If you get "Cannot find source file" errors, make sure you've copied all the Numerical Recipes files and `ESPL.tbl` as described in step 4.

## Updating

To update to the latest code:

```bash
cd gulls_mp
git pull
conda activate gulls
cmake --build build
```

To update the conda environment:

```bash
mamba env update -f unity.yml
```
