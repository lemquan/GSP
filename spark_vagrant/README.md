This is a tutorial to show how to install Apache Spark and integrate with IPython notebooks in Vagrant. With the release of 4.0>, IPython was deprecated such that Jupyter would be used as the main kernel. However, using Spark on Jupyter is still not straight forward with instructions. These instructions are for a single instance on Vagrant as such we will not be using Hadoop. 


## Prerequisites 
- Java (might be difficult to install on Vagrant ubuntu_trusty64, so install Scala)
- Apache Spark
- Python 2.7
- IPython == 3.2.1
- Vagrant 

## Installation of IPython notebook and connect to host
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
The follow allows for graphing inline, any IP address is application (e.g. localhost), the port to open is 8886, and to not automatically luanch a browser for an IPython notebook. 

At this point, test if this configuration works with the command 
``` bash
sudo ipython notebook --profile=pyspark
```
Go to the host browser, and type in 'localhost:8886'

## Installation of Apache Spark 
1) Vagrant will not include Java. An easy way to get Java is to install Scala which will include Java.
``` bash
wget http://downloads.lightbend.com/scala/2.11.1/scala-2.11.7.deb # can skip unless you want a specific version of Scala
sudo dpkg -i scala-2.11.7.deb                                     # installing specific version of Scala 
sudo apt-get update 
sudo apt-get install scala 
```

2) Install Spark. Be sure to select the option to include Pre-Built Hadoop 2.6. This saves us time from compiling Hadoop with sbt or Maven as the files are ready for use. 
``` bash
wget http://www.apache.org/dyn/closer.lua/spark/spark-1.6.1/spark-1.6.1-bin-hadoop2.6.tgz
tar -zxvf spark-1.6.1-bin-hadoop2.6.tgz -C /usr/local/spark 
```

Add Scala and Spark to .bashrc. Usually, the package will install the SCALA_HOME environment. If not, doing the following:
``` bash
export SCALA_HOME=/usr/local/src/scala/scala-2.11.7
export SPARK_HOME=/usr/local/spark/spark-1.6.1-bin-hadoop2.6
export PATH=$PATH:$SCALA_HOME/bin:$SPARK_HOME
```
Include the changes by exiting Vagrant or call 
```bash
source .bashrc
```

Verify that Scala and Spark works,
```bash
scala -version
cd $SPARK_HOME
./bin/spark-shell
```

## Using Spark with IPython
In this section, we will enable Spark on IPython notebooks. This process is simple, but it might not always work. 
Go to the ~/.ipython/profile_pyspark/startup. The startup folder initiates any scripts needed to run upon startup of the notebook. *Sometimes this will not work.* But we will include it anyways as it might work for you. 
``` python
import os
import sys

spark_home = os.environ.get('SPARK_HOME', None)
if not spark_home:
      raise ValueError('SPARK_HOME environment variable is not set')
sys.path.insert(0, spark_home +'/python')
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.9-src.zip')) #be sure this is the correct version in Hadoop
execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))
```
At this point, you should be able to launch IPython notebook as above. 
Go to the host browser, and type in 'localhost:8886'. In the notebook, import pyspark to test if Spark works. 

## Spark does not work in IPython
For some people, the startup scripts will not launch in the notebook. Some tips that may remedy:

1) Check if Spark launches by typing ipython in the terminal. The Spark prompt should populate. This means that the envs have been set properly. 

2) Copy the structure of profile_pyspark to that of profile_default. 

3) Lastly, manually run the script written in 00-pyspark-startup.py

A test to perform when in the notebook is to import sys. Check the paths that sys recongises, it should include
```
/usr/local/spark/spark-1.6.1-bin-hadoop2.6/python/lib/py4j-0.9-src.zip',
 '/usr/local/spark/spark-1.6.1-bin-hadoop2.6/python',
```
Sometimes the startup script does not run automatically depending on how ipython is called. If ipython is ran, sometimes Spark will run automatically but with a specific profile it will not run. 
