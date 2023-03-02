build:
	docker build . -t cvisionai/dextr

bash:
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --network=host --env DISPLAY=127.0.0.1:0 --privileged --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 cvisionai/dextr /bin/bash

run:
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --network=host --env DISPLAY=127.0.0.1:0 --privileged --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 cvisionai/dextr 
