# Base image
FROM public.ecr.aws/lambda/python:3.10

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install only required packages (lightweight)
RUN pip install --no-cache-dir \
    tensorflow==2.19.0 \
    numpy \
    scipy \
    scikit-learn \
    joblib \
    firebase-admin

# Copy project files
COPY lambda_function.py .
COPY model_dir ./model_dir
COPY scaler.pkl .
COPY target_scaler.pkl .
COPY firebase_key.json .

# Lambda handler
CMD ["lambda_function.handler"]