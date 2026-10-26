# Reproducible training environment (used on the author's Windows PC, where
# Smart App Control blocks TensorFlow/pandas native DLLs - see NOTES.md).
FROM tensorflow/tensorflow:2.21.0
WORKDIR /project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV TF_CPP_MIN_LOG_LEVEL=2 PYTHONHASHSEED=42
