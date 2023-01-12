FROM rayproject/ray:ed6c6f-py38-cu116
COPY . /app
USER root
WORKDIR /app
RUN pip install .
CMD ["t1000_v2"]
