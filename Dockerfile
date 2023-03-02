FROM nvcr.io/nvidia/pytorch:23.01-py3
#RUN pip3 install matplotlib opencv pillow scikit-learn scikit-image pyqt5
RUN pip3 install matplotlib opencv pillow scikit-learn scikit-image pyqt5 redis
ENV MPLBACKEND=qtagg
COPY . /dextr
WORKDIR /dextr/models
RUN chmod +x download_dextr_model.sh && ./download_dextr_model.sh
WORKDIR /dextr
CMD [ "python3", "worker.py" ]
