# Advanced Programming project: Neural Network signal vs. background

## How to run the code

To build the **docker image**

```sudo docker build -f nn.dockerfile -t nn .```

To execute the image to create the container

```sudo docker run --rm -p 8888:8888 -v /home/giovanni-pedrelli/advanced-programming-project/:/home/jovyan/ nn```

To execute the image to create the container accessing it via VScode

```sudo docker run -it --rm -p 8888:8888 -v /home/giovanni-pedrelli/advanced-programming-project/:/home/jovyan/ nn jupyter notebook --ip 0.0.0.0```