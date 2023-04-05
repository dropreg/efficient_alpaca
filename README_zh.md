<p align="center" width="100%">
<img src="efficient_alpaca_logo.PNG" alt="Efficient-Alpaca" style="width: 40%; min-width: 300px; display: block; margin: auto;">
</p>


<h2 align="center">
    <p>Efficient Alpaca</p>
</h2>

<h4 align="center">
    <p>
        <a href="https://github.com/dropreg/efficient_alpaca/blob/main/README.md">English</a> | <b>中文</b>
    <p>
</h4>

Efficient Alpaca 的目的是为了方便构建或者增强基于 LLMs 的 Chatbots，其功能包括但不限于 **减少资源使用 （GPU 显存，训练时间）**，**推理速度**，**方便开发者使用（尤其是熟悉 Fairseq 的用户）**。项目会持续更新，欢迎使用！

**************************** 更新记录 ****************************
- 4/5 我们支持 FSDP 进行 Fine-tuning，可以使用额外内存来减少 GPU 显存占用！
- 3/17 我们支持使用 Megatron-LM 来减少 GPU 显存，包括 Fine-tuning 和 Efficient-finetuning !
- 3/15 我们支持使用 LoRA 来进行 Efficient-finetuning 来复现 Stanford Alpaca !


## 可供选择的推理设备

你可以选择下面的任意设备来支持推理，即使是 12G 的 1080。

| Method   | Support       | Device     | GPU     | Inference Speed     |
| -------- | ------------- | ---------- | ------- | ------------------- |
| Original | all           | 1 24G 3090 |  14G    |                     |
| Megatron | megatron_lora | 2 12G 1080 |  8G     |                     |

## 可供选择的训练方法和设备

你可以根据下标的组合来选择可用方法：例如，我有两块3090和大量的内存，这个时候你可以有两种选择：1. 使用Megatron-LM 来进行 Efficient-Finetuning（不会使用大量内存）。 2. 使用 FSDP 来进行 Fine-tuning （会使用额外大量内存）。

| Method        | Type                  | Support       | Data Para | Model Para | Device     | GPU     | Memory Limit  | Training Speed      |
| ------------- | --------------------- | ------------- | --------- | ---------- | ---------- | ------- | ------------- | ------------------- |
| LoRA          | Efficient Fine-tuning | lora          | &check;   | &check;    | 1 40G A100 |  30G    | No            | 90 sec / 100 step   |
| Megatron-LoRA | Efficient Fine-tuning | megatron_lora | &cross;   | &check;    | 2 24G 3090 |  21G    | No            | 190 sec / 100 step  |
| FSDP          | Fine-tuning           | fsdp          | &check;   | &check;    | 1 40G A100 |  32G    | 128G +        | 1600 sec / 100 step |
|               |                       |               |           |            | 8 40G A100 |  32G    | No            | 400 sec / 100 step  |
|               |                       |               |           |            | 2 24G 3090 |  13G    | 128G +        | 900 sec / 100 step  |
|               |                       |               |           |            | 8 24G 3090 |  22G    | 128G +        | 800 sec / 100 step  |
| Megatron      | Fine-tuning           | megatron      | &cross;   | &check;    | 4 40G A100 |  25G    | No            | 130 sec / 100 step  |
|               |                       |               |           |            | 8 24G 3090 |  14G    | No            | 130 sec / 100 step  |


<details><summary>关于表格的解释</summary><p>

以上所有的实验都是使用 --max-tokens 2048 这一参数进行测试.

* Data Para: 是否支持数据并行.
* Model Para: 是否支持模型并行.
* GPU: 在训练中实际大概使用的 GPU 显存.
* Memory Limit: 内存限制，只是大概的测试，并不代表实际情况.
* Training Speed: 仅仅代表训练速度而非训练时间，比如数据并行可以加快训练时间，但是并不会加快训练速度。

</p></details>


## 进行演示

我们支持使用 [Gradio](https://gradio.app/) 来进行演示。 

```
bash  alpaca_lora/scripts/run_webapp.sh
```

<p align="center" width="100%">
<img src="webapp.PNG" alt="Examples" style="width: 60%; min-width: 300px; display: block; margin: auto;">
</p>


## 环境依赖
请确保你有正常的CUDA环境，并安装以下依赖:

```
pip install fairseq
pip install fairscale
```

如果要训练模型，请安装 sentencepiece [official repo](https://github.com/google/sentencepiece) 来处理数据。

```
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
make install
```

## 数据和模型处理

+ [Data and Model Preprocess](alpaca/scripts/utils/README.md)

## 训练模型

Efficient-Finetuning
+ [LoRA](alpaca/scripts/lora/README.md)
+ [Megatron + LoRA](alpaca/scripts/megatron_lora/README.md)

Fine-tuning
+ [Megatron](alpaca/scripts/megatron/README.md)
+ [Fully Sharded Data Parallel](alpaca/scripts/fsdp/README.md)

## 模型输出示例:

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

    **Answer**:
    The legendary warrior Guan Yu Once upon a time, there was a legendary warrior named Guan Yu. He was a highly skilled fighter and commander and was known for his great bravery and loyalty. He was the loyal servant of Emperor Liu Bei, and fought in the Three Kingdoms civil war alongside him. He was brave enough to challenge the powerful enemy warriors, and eventually helped Liu Bei's army defeat the Wei Kingdom, the strongest of the three kingdoms. Guan Yu was a great warrior and a hero in Chinese history.

4. **Question**: what is an alpaca? how is it different form a llama?

    **Answer**:
    Alpacas and llamas are both members of the Camelidae family. They are both herbivorous mammals that are domesticated for their coats, but they are distinctly different species. Llamas are generally larger than alpacas and have more pronounced spiral horns, while alpacas have smaller horns and more delicate coats. Alpacas also have smaller teeth and a narrower neck than llamas, and they are able to reproduced year-round. Llamas are also unable to reproduce during the winter months and have a more social behavior than alpacas.

5. **Question**: write a e-mail to congratulate Lee and mention that you are exited about meeting all of them in person.

    **Answer**:
    ```
    Dear Lee,

    I want to congratulate you on your success and I'm looking forward to meeting you and the other team members in person. 
    It's exciting to see the progress you've made and I'm sure the hard work you have put in has paid off. 
    I'm very proud of you and I'm looking forward to meeting you in person and seeing all of your progress.

    Sincerely,
    [Your Name]
    ```

5. **Question**: write a well-thought our abstract for a machine learning parper that proves that 42 is the optimal seed for training neural network.

    **Answer**:
    This paper investigates the effect of different seeds on the performance of neural networks. Through experiments, we demonstrate that the optimal seed is 42, which leads to the best performance on a wide range of tasks. We analyze the effects of different seeds on the model's accuracy, robustness, and other metrics. Our results show that 42 is the best seed for neural networks, and we discuss the implications of this finding.


## Reference

In addition, thanks to the open source projects and communities that this project relies on, including but not limited to the following:

Fairseq: a sequence modeling toolkit \[[github](https://github.com/facebookresearch/fairseq)\]
```
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```

FairScale: is a PyTorch extension library for high performance and large scale training. \[[github](https://github.com/facebookresearch/fairscale)\]
```
@Misc{FairScale2021,
  author =       {{FairScale authors}},
  title =        {FairScale:  A general purpose modular PyTorch library for high performance and large scale training},
  howpublished = {\url{https://github.com/facebookresearch/fairscale}},
  year =         {2021}
}
```

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