FROM nvcr.io/nvidia/pytorch:23.01-py3
#RUN pip3 install matplotlib opencv pillow scikit-learn scikit-image pyqt5
RUN pip3 install matplotlib opencv pillow scikit-learn scikit-image pyqt5 redis
ENV MPLBACKEND=qtagg
COPY models/download_dextr_model.sh /models/download_dextr_model.sh
WORKDIR /models
RUN ./download_dextr_model.sh
COPY . /dextr
RUN mv /models/* /dextr/models/.
WORKDIR /dextr
CMD [ "python3", "worker.py" ]
