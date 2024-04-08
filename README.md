# PitSurgRT: real-time localization of critical anatomical structures in endoscopic pituitary surgery
### [Paper](https://doi.org/10.1007/s11548-024-03094-2) | [BibTex](#citation)

## PitSurgRT

Purpose: Endoscopic pituitary surgery entails navigating through the nasal cavity and sphenoid sinus to access the sella using an endoscope. This procedure is intricate due to the proximity of crucial anatomical structures (e.g. carotid arteries and optic nerves) to pituitary tumours, and any unintended damage can lead to severe complications including blindness and death. Intraoperative guidance during this surgery could support improved localization of the critical structures leading to reducing the risk of complications.

Methods: A deep learning network PitSurgRT is proposed for real-time localization of critical structures in endoscopic pituitary surgery. The network uses High-Resolution Net (HRNet) as a backbone with a multi-head for jointly localizing critical anatomical structures while segmenting larger structures simultaneously. Moreover, the trained model is optimized and accelerated by using TensorRT. Finally, the model predictions are shown to neurosurgeons, to test their guidance capabilities.

Results: Compared with the state-of-the-art method, our model significantly reduces the mean error in landmark detection of the critical structures from 138.76 to 54.40 pixels in a 1280 $\times$ 720-pixel image. Furthermore, the semantic segmentation of the most critical structure, sella, is improved by 4.39\% IoU. The inference speed of the accelerated model achieves 298 frames per second with floating-point-16 precision. In the study of 15 neurosurgeons, 88.67\% of predictions are considered accurate enough for real-time guidance.

Conclusion: The results from the quantitative evaluation, real-time acceleration, and neurosurgeon study demonstrate the proposed method is highly promising in providing real-time intraoperative guidance of the critical anatomical structures in endoscopic pituitary surgery.

## Architecture
<p align="center">
<img src="./docs/PitSurgRT.png" align="center">
</p>



<!-- ---------------------------------------------- -->
## Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@article{mao2024pitsurgrt,
  title={PitSurgRT: real-time localization of critical anatomical structures in endoscopic pituitary surgery},
  author={Mao, Zhehua and Das, Adrito and Islam, Mobarakol and Khan, Danyal Z and Williams, Simon C and Hanrahan, John G and Borg, Anouk and Dorward, Neil L and Clarkson, Matthew J and Stoyanov, Danail and others},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--8},
  year={2024},
  publisher={Springer}
}
```
