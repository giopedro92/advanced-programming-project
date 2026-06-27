FROM quay.io/jupyter/datascience-notebook

RUN pip install numpy uproot scikit-learn matplotlib pandas seaborn tensorflow pydot

USER root

RUN apt-get update && \
    apt-get install -y graphviz && \
    rm -rf /var/lib/apt/lists/*

USER jovyan
WORKDIR /home/jovyan