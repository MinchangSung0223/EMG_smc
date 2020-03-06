# Ubuntu 설치 및 Pytorch로 딥러닝 실행하기
## 1.Ubuntu 설치방법
### 1)Ubuntu 18.04 install
     http://mirror.kakao.com/ubuntu-releases/bionic/ubuntu-18.04.3-desktop-amd64.iso

### 2)Install Rufus(boot util)
     https://rufus.ie/
     google에 Ubuntu 18.04 부팅 USB 만들기 라고 치면 상세히 나온다.
     https://hiseon.me/linux/ubuntu/ubuntu-install-usb/ 
     위의 링크를 참고
### 3)만들어진 부팅 USB를 컴퓨터에 연결한 뒤에 재부팅한다.
     컴퓨터 마다 boot옵션으로 들어가는 키가 다르다. google에 boot옵션 들어가기를 찾아보고
     알맞은 키를 눌러서 boot옵션화면으로 들어간다.
### 4)Ubuntu가 설치된 USB를 boot drive로 설정하여 부팅한다.

### 5)Ubuntu를 설치한다. 설치 시에 영어로 설치하는 것이 좋다. 이후 한글설정 가능

### 6) 설치 후 자동으로 재부팅된다. Boot USB를 뺀 뒤 재부팅한다.

## 2. 알맞은 Graphic Driver 설치하기
### 1) 자신의 Graphic Driver를 확인한다. 
### 2) NVIDIA DRIVER 사이트에서 Graphic Driver를 찾아서 설치한다.
      https://www.nvidia.com/Download/Find.aspx?lang=en-us
