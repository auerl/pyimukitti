# pyimukitti

**Disclaimer: This is just a quick experiment and not cleaned up.**

## Installing GTSAM

To run this snippet you need to install [GTSAM](https://bitbucket.org/gtborg/gtsam) and its Cython bindings

```
git clone git@bitbucket.org:gtborg/gtsam.git
cd gtsam && mkdir build && cd build
cmake ..
make
cd cython/gtsam
python setup.py install
```

## Running the code

To run the code and plot a comparison with the IMUKitti.m (MATLAB) scripts
shipped with GTSAM 3.2.1 and GTSAM 4.0.0 type

```
python imukitti.py
```

The data is included in the [data](./data) folder.

