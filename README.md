No projeto de estudo antigo era usado o TF, que já estava se encontrando obsoleto, mudei para o modelo Yolov8, bem mais leve e mais limpo, onde não precisa sem usado mais protocolBuffers, o modelo não pecisa mais ser baixado manualmente e não precisa mais ser carregado label_map.pbtxt

Funcionamento atual:
- O modelo é carregado automaticamente
- Frame passa pelo modelo
- O modelo faz a detecção filtrando somente a classe 0 do COCO, sendo Person
- Retorna as coordenadas do bounding box
- É desenhado o bbox e a confiança
