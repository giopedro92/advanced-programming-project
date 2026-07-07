# Advanced Programming project: Neural Network signal vs. background

## How to run the code

1. Download the content of this repository
```bash
cd ~/ && git clone https://github.com/giopedro92/advanced-programming-project.git && cd ~/advanced-programming-project
```

2. Build the **docker image**

```bash
sudo docker build -f nn.dockerfile -t nn .
```

3. Execute the image to create the container

To execute the image to create the container and access it via a browser

```bash
sudo docker run --rm -p 8888:8888 -v ~/advanced-programming-project/:/home/jovyan/ nn
```

To execute the image to create the container accessing it via VScode

```bash
sudo docker run -it --rm -p 8888:8888 -v ~/advanced-programming-project/:/home/jovyan/ nn jupyter notebook --ip 0.0.0.0
```