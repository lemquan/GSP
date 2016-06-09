This is a tutorial to show how to install Apache Spark and integrate with IPython notebooks in Vagrant. With the release of 4.0>, IPython was deprecated such that Jupyter would be used as the main kernel. However, using Spark on Jupyter is still not straight forward with instructions. 

## Prerequisites 
- Java (might be difficult to install on Vagrant ubuntu_trusty64, so install Scala)
- Apache Spark
- Python 2.7
- IPython == 3.2.1
- Vagrant 

## Installation 
1) Install Vagrant on your host computer. We use the trusty64 version of Ubuntu. It is best practice
to store the Vagrant file in a separate folder (e.g. user/virtual_envs/spark). Run the following commands. 
``` bash 
vagrant init ubuntu/trusty64
vagrant up
vagrant ssh 
```
The following commands will create the VagrantFile, initate the box, and log  into the Vagrant box.

2) Install pip, be sure to get the latest version of pip. Install IPython
``` bash
sudo pip install 'ipython==3.2.1'
```
This installation will create a .ipython folder in the user's directory (e.g. /home/vagrant). If that is not the case, run ipython from the terminal once. 

3) We will not ensure the IPython can work from the localhost with a given port. The VagrantFile will need to be modified to include a nonconflicting port number. Insert the following line:
``` bash
config.vm.network "forwarded_port", guest:8886, host:8886, auto_correct:true
```
Generally, we use port 8888 but it may conflict if you have other Vagrant instances. Next, we have to reload the box with this new configuration 
``` bash
vagrant reload
vagrant ssh 
```

4) After reloading the VagrantFile, the new configuration will be included in the box. We need to configure IPython to recognise the port. From the box, 
```
ipython profile create pyspark
```
This will create a the new profile inside the .ipython folder. It will be named 'profile_pyspark'. 
Create a new file as '/home/vagrant/.ipython/profile_pyspark/ipython_notebook_config.py'
Include the following:
``` python
c = get_config()
c.IPKernelApp.pylab = 'inline'
c.NotebookApp.ip = '*'
c.NotebookApp.port = 8886
c.NotebookApp.open_browser = False
```
