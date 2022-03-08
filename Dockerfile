# this is the version found on vocareum
FROM nvidia/cuda:8.0-devel-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive

# create a non-root user to use once podman gets fixed up
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
  && apt-get update \
  && apt-get install -y sudo \
  && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
  && chmod 0440 /etc/sudoers.d/${USERNAME}

# install helpful or neccessary packages (some of these may be replaced by homebrew once that's going)
RUN apt update \
    && apt install -y \
        git \
        tmux \
        vim


# This breaks when using podman non-root (i think because it uses the host's UID/GID?) but I think is needed for windows?
# Comment out on podman!
USER ${USERNAME}
WORKDIR /home/${USERNAME}
