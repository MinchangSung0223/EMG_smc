# EMG_smc

Ubuntu 설치방법

1.Ubuntu 18.04 iso파일을 받는다.
   http://mirror.kakao.com/ubuntu-releases/bionic/ubuntu-18.04.3-desktop-amd64.iso


2.Install Rufus(boot util)
   https://rufus.ie/
   
Rufus는 usb를 부팅 드라이브로 만들어 주는 프로그램이다. 
Rufus를 설치후 실행한다.
3.NVIDIA driver install
  https://www.nvidia.com/Download/Find.aspx?lang=en-us
 
   그래픽카드에 알맞는 NVIDIA DRIVER 설치

 - https://www.nvidia.com/Download/Find.aspx?lang=en-us

 - 접속후 click을 누른다.

 - 최신버전 선택

 -  'NVIDIA-Linux-x86_64-xxx.xxx.run' 와 같은 파일이 다운됨

 

 - ctrl+t를 눌러 터미널 창을 연다

 - terminal: sudo apt-get remove nvidia* && sudo apt autoremove

 - terminal: sudo apt-get install dkms build-essential linux-headers-generic

 - terminal: sudo gedit /etc/modprobe.d/blacklist.conf

 

   --blacklist.conf 파일 맨 밑에 추가한다. --

    blacklist nouveau

    blacklist lbm-nouveau

    options nouveau modeset=0

    alias nouveau off

    alias lbm-nouveau off

 - terminal: echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf

 - terminal: sudo update-initramfs -u

 - terminal: sudo reboot

 

 - terminal: cd /home/user/Download

 - terminal: sudo chmod 777 NVIDIA-Linux-x86_64-xxx.xxx.run

 - ctrl+alt+f1

 - login root

 - xserver terminal: sudo service lightdm stop

 - xserver terminal: cd /home/user/Download

 - xserver terminal: ./NVIDIA-Linux-x86_64-xxx.xxx.run

 - Instalation will run

 - When installation is finished

 - xserver terminal: sudo service lightdm start

 - ctrl+alt+f7

 - terminal: sudo apt-get install dkms nvidia-modprobe

 - terminal: sudo lspci - k

 - terminal: nvidia-smi


4.CUDA Download
  https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

5.Cublas Download
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal
6. install
 - ctrl+alt+f1

 - login root

 - xserver terminal: sudo service lightdm stop

 - xserver terminal: cd /home/user/Download

 - xserver terminal: ./CUDA~~~~~~~~~~.run
 - xserver terminal: ./CUblas~~~~~~~~~~.run

7. Git install

  - terminal: apt-get install git

8. Python3 install

  - terminal: apt-get update

  - terminal: apt-get install python3 python3-pip

  - terminal: pip install numpy
9. virtualenv install
   apt-get install virtualenv
