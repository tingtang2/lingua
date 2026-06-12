# Complete, reproducible script to build and prepare environment
export UV_CACHE_DIR="/p/lustre5/mcleish1/uv_cache"

# modify the installation path and env name if you want
INSTALLDIR=/p/vast1/$USER
ENV_NAME="tuolumne_conda_642_ting"

cd ${INSTALLDIR}

# Base the installation on previously installed miniconda.
# Note, this is a manual process currently.

source deactivate > /dev/null 2>&1 # discard potentially preloaded conda environments
echo "Conda Version:" $(conda env list | grep '*') 


# Create conda environment, and print whether it is loaded correctly
conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.11 --yes --override-channels -c defaults
conda activate ${INSTALLDIR}/$ENV_NAME
echo "Pip Version:" $(which pip)  # should be from the new environment!

# Conda packages:
conda install --override-channels -c conda-forge conda-pack libstdcxx-ng -y

# # Load module family
rocm_version=6.4.2

# Load modules
module load rocm/$rocm_version
module load gcc-native/13.2
module load gcc/13.3.1
module load rocm-compiler/4.3.0

######### COMPILE PIP PACKAGES ########################

# pytorch and core reqs
pip install uv
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
uv pip install ninja
uv pip install -r /usr/workspace/mcleish1/loss-spikes-project/lingua/requirements.txt

# amdsmi
cp -R /opt/rocm-${rocm_version}/share/amd_smi/ $INSTALLDIR/amd_smi_${rocm_version}
cd $INSTALLDIR/amd_smi_${rocm_version}
pip install .
cd ${INSTALLDIR}

rm -r $UV_CACHE_DIR