FROM continuumio/miniconda3

# Set environment variables
ENV PATH=/opt/conda/bin:$PATH

# Clone git repo (overridden at runtime by bind mount, but needed for build context)
RUN git clone https://github.com/InternLM/MindSearch.git /app

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt /tmp/requirements.txt

# Create conda env and install LangGraph dependencies
RUN conda create --name fastapi python=3.10 -y && \
    conda run -n fastapi pip install -r /tmp/requirements.txt && \
    conda clean --all -f -y

EXPOSE 8002

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "fastapi"]
CMD ["python3", "-m", "mindsearch.app", "--host", "0.0.0.0", "--port", "8002", "--search_engine", "GoogleSearch"]
