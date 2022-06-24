xhost +local:docker && docker run -d -it --rm --cap-add sys_ptrace -p0.0.0.0:7777:22 \
        --name tensorrt_dev --gpus all \
        --privileged \
        -e DISPLAY="$DISPLAY" \
        -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -v $XAUTHORITY:/tmp/.XAuthority -e XAUTHORITY=/tmp/.XAuthority \
        doc.smartparking.kz/tensorrt_dev:tenssort_dev