This is a tutorial to show how to install Apache Spark and integrate with IPython notebooks in Vagrant. With the release of 4.0>, IPython was deprecated such that Jupyter would be used as the main kernel. However, using Spark on Jupyter is still not straight forward with instructions. 

## Prerequisites 
- Java (might be difficult to install on Vagrant ubuntu_trusty64, so install Scala)
- Apache Spark
- Python 2.7
- IPython == 3.2.1
- Vagrant 

## Installation 
1) Install Vagrant on your host computer. 
``` bash 
vagrant init ubuntu/trusty64
```
