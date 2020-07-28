# this base image works in both CPU and GPU enabled environments
FROM substrafoundation/substra-tools:0.6.0

# install dependencies
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install torch==1.5.1 torchvision==0.6.1 numpy==1.19.0 opencv-python==4.2.0.34 opencv-contrib-python==4.2.0.34 pandas==1.0.5 tqdm==4.48.0

# add your algorithm script to docker image
ADD algo.py .
ADD blazeface-pytorch/anchors.npy blazeface-pytorch/blazeface.pth blazeface-pytorch/blazeface.py ./blazeface-pytorch/
ADD deepfakes-inference-demo/resnext.pth deepfakes-inference-demo/resnext50_32x4d-7cdf4587.pth ./deepfakes-inference-demo/
ADD deepfakes-inference-demo/helpers/face_extract_1.py deepfakes-inference-demo/helpers/read_video_1.py ./deepfakes-inference-demo/helpers/

# define how script is run
ENTRYPOINT ["python3", "algo.py"]
