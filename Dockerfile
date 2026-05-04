# Base image for AWS Lambda Python 3.10
FROM public.ecr.aws/lambda/python:3.10

# 1. System dependencies
RUN yum -y install \
    gcc \
    gcc-c++ \
    make \
    hdf5-devel \
    && yum clean all

# 2. HDF5 paths set karein (h5py build fix)
ENV H5PY_SETUP_REQUIRES=0
ENV HDF5_DIR=/usr/lib64/hdf5

# 3. Pip aur build tools update karein
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 4. Requirements copy karein
COPY requirements.txt .

# 5. Install requirements 
# Hum sirf --only-binary use karenge taaki build system confuse na ho
RUN pip install --no-cache-dir \
    --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# 6. Project files copy karein
COPY lambda_function.py .
COPY my_model.keras .
COPY scaler.pkl .
COPY firebase_key.json .

# Lambda handler
CMD ["lambda_function.lambda_handler"]