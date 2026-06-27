To build the **docker image**

```sudo docker build -f nn.dockerfile -t nn .```

To execute the image to create the container

```sudo docker run -p 8888:8888 -v /home/giovanni-pedrelli/NN/:/home/jovyan/work nn```

To execute the image to create the container accessing it via VScode

```sudo docker run -it -p 8888:8888 -v /home/giovanni-pedrelli/NN/:/home/jovyan/work nn jupyter notebook --ip 0.0.0.0```