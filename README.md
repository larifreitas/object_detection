buffes de protocolo são um mecanismo extensível para  serializar dados estruturados, sendo necessário ára a execução e  descongelamento do modelo treinado frozen graph.

<p>Fazer unzip e compilação: 
<a href="https://github.com/protocolbuffers/protobuf/releases">Buffers aqui</a>

<p>compilação em models/research: $ protoc object_detection/protos/*.proto --python_out=.

<p><p>Dependência: pip install tensorflow-object-detection-api

<p>Uso do Tensorflow 2.0 
<p>Uso do protoc 3.13
<p>Instalação do models do Tensorflow: <a href="https://github.com/tensorflow/models">models</a>

<p>Uso do dataset coco: <a href="https://cocodataset.org/">coco API</a>

Estudo principal em: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
