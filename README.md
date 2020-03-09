# Face Recognition

Projeto de reconhecimento facial realizado para buscar um número mínimo de imagens por pessoa que permite atingir um reconhecimento satisfatório, comparando essa diferença para os classificadores KNN, SVM e MLP.<br>
Desenvolvido na disciplina de Resolução de Problemas II do curso de Sistemas de Informação da Escola de Artes, Ciências e Humanidades da Universidade de São Paulo.

- Para a detecção facial facial, e extração de embeddings para codificar as faces, foi utilizado o _port_ do MTCNN e FaceNet para PyTorch feito por [ESLER](https://github.com/timesler/), a partir da implementação e modelos treinados de [SANDBERG](https://github.com/davidsandberg/) usando Tensorflow.
- O modelo de treinamento do FaceNet foi realizado com VGGFace2, e é baixado pelo [facenet-pytorch]((https://github.com/timesler/facenet-pytorch)) na primeira execução.
- Para facilitar o download do Pytorch com pip, pode ser utilizado o script em _intall.py_.

## Processo de Reconhecimento

Dado uma imagem a passar por reconhecimento, o processo é divido em 3 fases:
- Detecção, que realiza o recorte da face em uma imagem, onde será usado o MTCNN;
- Extração de embeddings, que transforma a imagem recortada da face em um vetor, em que o modelo de FaceNet usado gera um vetor de 512;
- Classificação, que atribui o vetor conhecido a uma pessoa, a partir de um treinamento prévio (implementados em PEDREGOSA et al).

Antes do treinamento, entretanto, pode ser realizada uma otimização dos parâmetros dos classificadores, utilizando otimização de bayes, presente em HEAD et al.<br>
Para comparação das taxas de reconhecimento por número de imagens, é selecionado um número aleatório de imagens (de acordo com uma random_seed), de pessoas com ao menos 10 imagens, para que o número de classes seja constante. 

## Estrutura

A execução exige uma pasta de entrada, contendo pastas para cada pessoa, com suas respectivas imagens. Os dados sobre as imagens e pessoas são salvos em “.\data\output”, com embeddings.bz2 contendo os vetores de embeddings gerados para cada imagem, e people.bz2 armazenando dados sobre cada pessoa (como quantidade de imagens, etc). Estes dois arquivos podem ser apagados para forçar a reanalise do conteúdo da pasta de entrada.

O processo de treinamento e teste tem início em face_recognition_train.py, onde é criada a classe TrainTestRecognition, que percorre as pastas para carregar imagens, interagindo com as classes FaceDetector (de face_detector.py) e FaceEmbeddings (de face_embeddings.py) para efetuar a detecção de faces e gerar sua representação em vetores (embeddings), além de selecionar imagens aleatórias, para as execuções. Após selecionados os conjuntos de dados que participarão da execução, é chamada a classe FaceClassifier (de face_classifier.py), que realiza o processo de otimização de hiperparâmetros, treinamento e teste, além de salvar relatórios sobre as execuções realizadas.
Os relatórios gerados permitem comparar os hiperparâmetros que geram os melhores resultados (f1-score), além de mostrar também os scorings por número de imagens por pessoa. 


## Execução

A execução de face_recognition_train.py aceita diversos parâmetros de entrada, como:
-	-i path ou --input_dir path: define o caminho da pasta de entrada (padrão é “.\data\input”);
-	-down ou --download: baixa o banco de imagens LFW na pasta de entrada;
-	-ap ou --append: adiciona as imagens na pasta de entrada aos arquivos embeddings.bz2 e people.bz2 já gerados (utilizar para adicionar novas imagens ao sistema);
-	-clf model ou --classifier model: define o modelo do classificador desejado, ou “all” para executar todos (“svm” é o padrão);

-	-pt ou --parameter_tuning: ativa a execução da otimização de parâmetros, que exige uso do k-fold;
-	–ns sets ou --num_sets sets: número de sets utilizados no k-fold (3 é o padrão);
-	-ipp num ou --images_per_person num: número de imagens por pessoa para ser dividido entre os k sets (6 é o padrão);

-	-rs num ou --rand_seed num: define o valor utilizado na geração de números pseudoaleatórios (42 é o padrão, inserir -1 para ser utilizar um random seed);
-	-si ou --save_images: ativa o armazenamento de faces detectadas (salvas dentro de uma pasta MTCNN, dentro da pasta das imagens da pessoa);
-	-oni ou --optimize_num_images: ativa a otimização para testar número mínimo de imagens com bom desempenho;
-	-itn num ou --images_train num: número de imagens utilizadas para treinamento, sendo utilizada quando não for executada otimização de parâmetros (padrão é 4);
-	-itt num ou --images_test num: número de imagens utilizadas para treinamento sendo utilizada quando não for executada otimização de parâmetros (padrão é 2, pode ser definido para 0 para não realizar testes, apenas treinamento)
  

## Referências

1. CAO, Q.; SHEN, L.; XIE, W.; PARKHI, O. M.; ZISSERMAN, A. VGGFace2: A dataset for recognising face across pose and age. International Conference on Automatic Face and Gesture Recognition, 2018. Disponível em: http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf.
2. ESLER, T. Face Recognition Using Pytorch. Disponível em: https://github.com/timesler/facenet-pytorch.
3. HEAD, T. et al. Scikit-optimize. Scikit-optimize, v0.5.2,  2018. https://doi.org/10.5281/zenodo.1207017. Disponível em: https://github.com/scikit-optimize/scikit-optimize. 
4. PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011. Disponível em: https://github.com/scikit-learn/scikit-learn. 
5. SANDBERG, D. Face Recognition using Tensorflow Build Status. Disponível em: https://github.com/davidsandberg/facenet.
6. SCHROFF, F.; KALENICHENKO, D.; PHILBIN, J. FaceNet: A Unified Embedding for Face Recognition and Clustering. 2015. arXiv:1503.03832. Disponível em: https://arxiv.org/abs/1503.03832.
7. ZHANG, K.; ZHANG, Z.; LI, Z.; QIAO, Y. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters, 2016. Disponível em: https://arxiv.org/pdf/1604.02878.
