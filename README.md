# ai_self_driving_car
# Ubuntu 18.04 install
Install latest updates first. I used the build in updater but this also should work
```
$ sudo apt-get update 
```
I had to reboot afterwards.
## Install Miniconda and create new AI Environment
Download for Linux 64bit Python 3.6: https://conda.io/miniconda.html
```
ai@ubuntu:~/Downloads$ sh ./Miniconda3-latest-Linux-x86_64.sh

$ source ~/.bashrc

$ conda create --name ai python=3.6

$ source activate ai

#if you are behind a reverse proxy, you can disable ssl_check
$ conda config --set ssl_verify false
 
# installing Sypder.. kind of tricky.. the actual version we can use conda, but we also need some basic libs which we can through using app-get:
$ sudo apt-get install spyder
#due to a bug in spyder with the latest pyqt install the older version for now https://github.com/ContinuumIO/anaconda-issues/issues/9331

$ sudo apt-get install spyder
$ sudo apt-get install spyder3

$ conda install pyqt=5.6
$ conda install spyder

# installing more packages for later
$ conda install numpy
$ conda install matplotlib

```

## Install kivy
To run kivy inside conda there is not really a good documentation I could find. The following steps are a combined effort from the [Mac](https://kivy.org/docs/installation/installation-osx.html) and [Linux](https://kivy.org/docs/installation/installation-linux.html) install instructions.

I ussually install the dev versions of kivy to get all the dev features:

```
# Install necessary system packages
$ sudo apt-get install -y \
    python-pip \
    build-essential \
    git \
    python \
    python-dev \
    ffmpeg \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev

# Install gstreamer for audio, vide (optional)
sudo apt-get install -y \
    libgstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good

#if you are behind a corporate reverse proxy you have to add --cert <cert>.pem to the pip command

$ pip install Cython==0.25.2
$pip install Cython==0.25.2 --cert <rootCert>.pem

# Install the dev dev version
$ pip install https://github.com/kivy/kivy/archive/master.zip


```

## Ubuntu torch install without GPU
The current solution only works with torch 3.1:
```
$ pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl --cert <path to cert.pem> 
```

## Install class files from SuperDataScience
The original course files can be downloaded: https://www.superdatascience.com/artificial-intelligence/

This repo used them as the a template and I used it for my person training:
```
$ mkdir aiclass
$ cd aiclass
$ git clone https://github.com/tomafischer/ai_self_driving_car.git

```
You ready to go!!!!!
## General

### Git cert issue:
```
$ git config --global http.sslVerify false
```
