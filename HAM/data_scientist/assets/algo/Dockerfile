# this base image works in both CPU and GPU enabled environments
FROM substrafoundation/substra-tools:0.7.0

RUN mkdir /.cache
RUN chmod -R 777 /.cache

RUN mkdir /model
RUN chmod -R 777 /model

# install dependencies
RUN pip3 install pandas numpy sklearn pillow scipy 
#tensorflow  pickle-mixin
RUN pip3 install torch torchvision
#RUN pip3 install keras==2.2.4

# add your algorithm script to docker image
COPY resnet50-19c8e357.pth /model/
ADD algo.py .

# define how script is run
ENTRYPOINT ["python3", "algo.py"]
