## PEAN: A Diffusion-Based Prior-Enhanced Attention Network for Scene Text Image Super-Resolution (ACMMM 2024)

*[Zuoyan Zhao](http://palm.seu.edu.cn/homepage/zhaozuoyan/index.html), [Hui Xue](http://palm.seu.edu.cn/hxue/), [Pengfei Fang](https://fpfcjdsg.github.io/), [Shipeng Zhu](http://palm.seu.edu.cn/homepage/zhushipeng/demo/index.html)*

This repository offers the official Pytorch code for this paper. If you have any question, feel free to contact Zuoyan Zhao ([zuoyanzhao@seu.edu.cn](mailto:zuoyanzhao@seu.edu.cn)).

[[Main Paper]](https://doi.org/10.1145/3664647.3680974)    [[Full Paper (arXiv)]](https://arxiv.org/abs/2311.17955)   [[Code]](https://github.com/jdfxzzy/PEAN)    [[OpenReview]](https://openreview.net/forum?id=IxSKhO7ed6)    [[Slides]](https://github.com/jdfxzzy/PEAN/releases/download/assets/slides_MM24_PEAN.pdf)    [[Video]](https://github.com/jdfxzzy/PEAN/releases/download/assets/video_MM24_PEAN.mp4)     [[Poster]](https://github.com/jdfxzzy/PEAN/releases/download/assets/poster_MM24_PEAN.pdf)

## News

[2024.10.21] I am rated as the Outstanding Reviewer for ACMMM 2024. [[Link]](https://2024.acmmm.org/outstanding-ac-reviewer)

[2024.10.04] Poster of this paper is now available. This poster will be displayed at Poster Session 3 (Oct. 31st, 4:10pm ~ 6:10pm) at posterboard P222.

[2024.08.20] 🔥🔥🔥 Code and weights of this model is now available on Github. [[Link]](https://github.com/jdfxzzy/PEAN)

[2024.07.23] Full paper (including Supplementary Material) is now available on arXiv. [[Link]](https://arxiv.org/abs/2311.17955)

[2024.07.16] 🎉🎉🎉 This paper is accepted by ACMMM 2024. Congratulations to myself. Thank every reviewers for their appreciation to this work.

[2024.06.11] Reviews of this paper have been released. Luckily, it receives a score of "4 4 4" from three reviewers.

[2024.04.15] Preprint version of this paper is now available on arXiv. [[Link]](https://arxiv.org/abs/2311.17955)

[2024.04.14] I am nominated as a reviewer for ACMMM 2024. 

[2024.04.13] This paper has been submitted to ACMMM 2024. Wish me good luck.

## Environment

![python](https://img.shields.io/badge/Python-v3.6-green.svg?style=plastic)  ![pytorch](https://img.shields.io/badge/Pytorch-v1.10-green.svg?style=plastic)  ![cuda](https://img.shields.io/badge/Cuda-v11.3-green.svg?style=plastic)  ![numpy](https://img.shields.io/badge/Numpy-v1.19-green.svg?style=plastic)

Other possible Python packages are also needed, please refer to *requirements.txt* for more information.

## Datasets and Pre-trained Recognizers

- Download the TextZoom dataset from: https://github.com/JasonBoy1/TextZoom.
- Download the pre-trained recognizers from:
  - ASTER: https://github.com/ayumiymk/aster.pytorch.
  - CRNN: https://github.com/meijieru/crnn.pytorch.
  - MORAN: https://github.com/Canjie-Luo/MORAN_v2.
  - PARSeq: https://github.com/baudm/parseq.
- **Notes:** It is necessary for you to modify the *./config/super_resolution.yaml* file according to your path of dataset and recognizers.

## Training and Testing the Model

- According to our paper, the training phase of this model including a pre-training (optional) and fine-tuning process. If you want to start the pre-training process, you could use scripts like this: 

  ```shell
  python main.py --batch_size="32" --mask --rec="aster" --srb="1" --pre_training
  ```

  Assuming that the pre-trained weight is saved at *./ckpt/checkpoint.pth*. If you want to start the fine-tuning process with this checkpoint, you could use scripts like this:

  ```shell
  python main.py --batch_size="32" --mask --rec="aster" --srb="1" --resume="./ckpt/checkpoint.pth"
  ```

  Of course, the pre-training process is not necessary. You can also directly train the full model without a pre-trained checkpoint. Before training, you should firstly modify the value of "checkpoint" in *./config/cfg_diff_prior.json* to your directory for saving the checkpoints of the TPEM.

  The Transformer-based recognizer for the SFM loss can be downloaded at [https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt](https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt).

- If you want to test the pre-trained model under the easy subset of TextZoom (assuming that this dataset is saved at */root/dataset/TextZoom/test/easy*), you could use scripts like this: 

  ```shell
  python main.py --batch_size="32" --mask --rec="aster" --srb="1" --resume="./ckpt/checkpoint.pth" --pre_training --test --test_data_dir="/root/dataset/TextZoom/test/easy"
  ```

  Assuming that the fine-tuned weight is saved at *./ckpt/checkpoint.pth*, and the trained TPEM is saved at *./ckpt/TPEM_ckpt.pth*. You should firstly change the value of "resume_state" in *./config/cfg_diff_prior.json* to *./ckpt/TPEM_ckpt.pth*. Then you can use scripts like this to test the model under the easy subset of TextZoom (assuming that this dataset is saved at */root/dataset/TextZoom/test/easy*):
  
  ```shell
  python main.py --batch_size="32" --mask --rec="aster" --srb="1" --resume="./ckpt/checkpoint.pth" --test --test_data_dir="/root/dataset/TextZoom/test/easy"
  ```

## Weights of Our Implemented Models

- We provide the weights of the pre-trained version of the model (*PEAN_pretrained.pth*) and the full model (*PEAN_final.pth* and *TPEM_final.pth*).
- Baidu Netdisk: https://pan.baidu.com/s/1Bu2WdoZ1gIfHz8JRujVq9w, password: nr2n.
- Google Drive: https://drive.google.com/file/d/1kGhPN2wUCV12Cu4yX4WGgMer3U9sNNPu/view?usp=sharing.

## Acknowledgement

- We inherited most of the frameworks from [TATT](https://github.com/mjq11302010044/TATT), [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) and [Stripformer](https://github.com/pp00704831/Stripformer). Thank you for your contribution!

## Recommended Papers

- **[DPMN]** This is my first work on Scene Text Image Super-Resolution (STISR), which was accepted by AAAI 2023. [[Paper]](https://arxiv.org/abs/2302.10414) [[Code]](https://github.com/jdfxzzy/DPMN)
- **[GSDM]** An intresting work on Text Image Inpainting (TII), which was accepted by AAAI 2024. The idea of using a Structure Prediction Module and diffusion-based Reconstruction Module to complete this task was proposed by me. [[Paper]](https://arxiv.org/abs/[2401.14832](https://arxiv.org/abs/2401.14832)[) [[Code]](https://github.com/blackprotoss/GSDM)

## Citation

```
@inproceedings{zhao2024pean,
  title={{PEAN}: A Diffusion-Based Prior-Enhanced Attention Network for Scene Text Image Super-Resolution},
  author={Zuoyan Zhao and Hui Xue and Pengfei Fang and Shipeng Zhu},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  pages={9769--9778},
  year={2024},
}
```
