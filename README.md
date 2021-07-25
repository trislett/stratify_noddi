[![Build Status](https://travis-ci.com/trislett/stratify_noddi.svg?branch=main)](https://travis-ci.com/trislett/stratify_noddi)

# stratify_noddi

## Rough guide

### Requirements
mrtrix3 https://github.com/MRtrix3/mrtrix3

fsl https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

ANTs https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS


Additional ubuntu requirements (they're already installed)
```
apt-get -y install libblas-dev liblapack-dev gfortran
```

### Installation

```
pip install -U git+https://github.com/trislett/stratify_noddi
```

### Running the program

```
run_stratify_noddi -s sub-0001 -dwi /path/to/sub-0001-dwi-ap.* -pab0 /path/to/sub-0001-dwi-pa.* -rr -nt 8
```
