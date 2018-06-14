# ai_self_driving_car

## Ubuntu torch install without GPU
The current solution only works with torch 3.1:
```
$ pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl --cert <path to cert.pem> 
```

## General

### Git cert issue:
```
$ git config --global http.sslVerify false
```
