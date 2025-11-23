FROM docker/image/of/choice:version
ARG USERNAME=username
ARG UID=UID
ARG GID=2102
ENV PATH="/home/$USERNAME/.local/bin:$PATH"
RUN apt-get update && apt-get install -y sudo python3-pip x11-apps
RUN groupadd --gid 2102 dsi && \
    groupadd --gid 4960 docker && \
    groupadd --gid 6648 ug_dsi && \
    groupadd --gid 6653 ug_gannot && \
    groupadd --gid 6656 ug_dgx && \
    groupadd --gid 41052 ug_hpc && \
    groupadd --gid 668400055 ug_audience && \
    useradd --uid $UID --gid $GID --groups dsi,docker,ug_dsi,ug_gannot,ug_dgx,ug_hpc,ug_audience \
    --create-home --shell /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME
WORKDIR /home/workspace/$USERNAME
ENV DISPLAY=:0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN mkdir -p /dev/shm && chmod 1777 /dev/shm
COPY requirements.txt /home/workspace/$USERNAME/
RUN pip3 install --no-cache-dir -r requirements.txt
USER $USERNAME

CMD ["/bin/bash"]
