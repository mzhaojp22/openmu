This is the codebase for the project: 

*OpenMU: Your Swiss Army Knife for Music Understanding*

[Paper](https://arxiv.org/abs/2410.15573)

## Quick start

#### 1.1 Build the environment
A Dockerfile is provided and it should work out-of-the-box.


#### 1.2 Datasets and checkpoints

For OpenMU-Bench, our created benchmark for music understanding tasks, please download from [here](https://drive.google.com/drive/folders/1bkl8dWLviHGZWgrhcxf-zPZA-vlTlq7B?usp=sharing).

Please also download the checkpoints for inference.

#### 1.3 Training

OpenMU contains two training stages:

- Stage 1 training: OpenMU is trained to output captions conditioned on an input image;
- Stage 2 training: instruction following, where OpenMU follows instructions in the music domain.

To launch training, please checkout and use `stage1.sh` and `stage2.sh` respectively.

#### 1.4 Inference

Please c.f. `run_inference.sh` for running inference of the provided checkpoints. 
We use lyrics understanding as an example in the scripts.


## If you find our work helpful, please consider citing us:
```bibtex
@article{zhao2024openmu,
  title={OpenMU: Your Swiss Army Knife for Music Understanding},
  author={Zhao, Mengjie and Zhong, Zhi and Mao, Zhuoyuan and Yang, Shiqi and Liao, Wei-Hsiang and Takahashi, Shusuke and Wakaki, Hiromi and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2410.15573},
  year={2024}
}