
# Alpaca with LoRA

This repository aim to reproduce the Stanford Alpaca using low-rank adaptation (LoRA) based on fairseq toolkit. 


## Setup
```
pip install fairseq
pip install fairscale
```

[Optinal] We can install sentencepiece from [official repo](https://github.com/google/sentencepiece) to process data or hack for your specific task.

```
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
make install
```

## Prepare Model and Data:

1. Download the llama checkpoint from official repo [llama](https://github.com/facebookresearch/llama), or [unofficial repo](https://github.com/shawwn/llama-dl)

2. Prepare the checkpoint to fairseq toolkit:
    ```
    python alpaca_lora/scripts/process_llama_ckpt.py --model-dir [your llama dir] --model-file [your llama file]
    ```
    after that, we can get new checkpoint file ``model.pt`` in [your llama dir].

3. Download Alpaca training file [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json), which contains 52K instruction-following data for fine-tuning the Alpaca model.

4. Prepare the training data for fairseq toolkit:

    Set *DATA* path in following scripts for processed data:
    ```
    bash alpaca_lora/scripts/prepare_llama_training_data.sh
    ```

## Training Step:

```
bash alpaca_lora/scripts/run_train_alpaca.sh
```

## Infernce Step:
We support the batch-level inference:

```
bash alpaca_lora/scripts/run_inf_alpaca.sh
```

## some case:

Just sampled by epoch=1 checkpoint (30min):

1. **Question**: what is the capital of china?

    **Answer**: Beijing is the capital of China.

2. **Question**: please give a pieces of python code about socket connection.

    **Answer**: 
    ``` 
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Constructing a socket connection")
    print(s)

    s.bind(('', 5000)) # Bind the socket to the '(' and 5000'

    print("Listening for connections")
    while True:
        connection, address = s.accept()
        print("Got connection from", address)
        print("Connected!")
        connection.close() # Close the connection.

    print("Exiting the program")
    ```

3. **Question**: 请讲一个武侠故事.

    **Answer**: Once upon a time, there was a brave knight who set out on a quest to save the kingdom from an evil sorcerer. The knight was accompanied by a brave squire and a noble steed. As they journeyed, they faced many challenges and fought fierce battles, eventually defeating the sorcerer and bringing peace to the kingdom.

4. **Question**: what is an alpaca? how is it different form a llama?

    **Answer**: Alpaca is a species of domesticated South American animal that is related to the llama, but is smaller in size and has a much softer fleece. Alpacas are gentle, intelligent and docile. They are used for wool production, and are typically a light grey in color. Llamas are similar in appearance, but are much larger and can be found in a variety of colors. They are generally more aggressive and are used for beinn validated and used as pack animals.

5. **Question**: write a e-mail to congratulate Lee and mention that you are exited about meeting all of them in person.

    **Answer**: Dear Lee,

    I am excited to meet all of you in person! It is such an honor to be part of such an amazing team. Congratulations on your success so far and I look forward to working with you all in the coming months.

5. **Question**: write a well-thought our abstract for a machine learning parper that proves that 42 is the optimal seed for training neural network.

    **Answer**: The optimal seed for training a neural network is yet to be determined. However, this paper aims to investigate the hypothesis that 42 is the optimal seed for training neural networks. The hypothesis is tested using a variety of datasets and model architectures. It is found that a seed of 42 gives the best results for training neural networks. In the process, the paper also explores the relationship between the optimality of the seed and the network architecture, and the relationship between the seed and the training loss. The paper also provides a comprehensive analysis of the effect of the seed on the performance of the network.


## Reference

LLaMA: Open and Efficient Foundation Language Models \[[paper](https://arxiv.org/abs/2302.13971)\]\[[github](https://github.com/facebookresearch/llama)\]

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

Stanford Alpaca: An Instruction-following LLaMA model \[[github](https://github.com/tatsu-lab/stanford_alpaca)\]

```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}