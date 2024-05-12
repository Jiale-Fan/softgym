# nvidia-docker run \
docker run \
  --privileged \
  --gpus all \
  -v /home/jiale/softgym:/home/jiale/softgym \
  -v /home/jiale/anaconda3:/home/jiale/anaconda3 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it xingyu/softgym:latest bash