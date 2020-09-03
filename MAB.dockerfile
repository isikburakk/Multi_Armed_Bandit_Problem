FROM ermaker/keras:py3

RUN conda install -y \
    jupyter

COPY requirements.txt ./
COPY animations ./
COPY modules ./

RUN pip install -r requirements.txt
RUN conda install -c conda-forge ffmpeg

ADD Answers.ipynb /
VOLUME /notebook
WORKDIR /notebook
EXPOSE 8888
CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --NotebookApp.allow_origin='*'