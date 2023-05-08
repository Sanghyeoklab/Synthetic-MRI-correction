# Synthetic-MRI-correction
# List
1. [Propose](#01-Propose)
2. [Method](#02-Method)
3. [Parameters](#03-Parameters)
4. [Generate mask](#04-Generate-mask)
5. [Train and Test](#05-Train-and-Test)
6. [Inference](#06-Inference)

- # 01 Propose
 ![01_Synthetic MRI artifact](https://user-images.githubusercontent.com/52817707/236194247-c426ba57-76b0-48a1-8380-e5a9eac0ce6c.png)
 Synthetic MRI obtains multi-model MRI data by a single shot. 
 Although it takes much less time than the conventional method, several artifacts are in the T2 FLAIR image. 
 In <b>Figure 1</b>, there are bright artifacts in synthetic T2 FLAIR's red box, while conventional methods do not. 
 These artifacts can cause misdiagnosis and must be corrected.<br>

- # 02 Method
 ![02_SyntheticMRI](https://user-images.githubusercontent.com/52817707/236370698-27464362-e816-425b-9bb9-69739062b780.png)
 Deep learning has been solving more complex problems than previous methods. 
 Among many deep learning architectures, U-net is widely used in medical image processing. 
 It uses an image pyramid to use local features and global features. 
 In this project, input images are synthetic MRI images; T1, T2, PD, T1 FLAIR, T2 FLAIR, and STIR in <b>Figure 2</b>. 
 The target image is a conventional T2 FLAIR image. 
 In my case, conventional T2 FLAIR's contrast was not stable, so I matched synthetic T2 FLAIR with NYUL method. (It is int <b>Util.py</b>)<br>

- # 03 Parameters
 ![03_Parameter](https://user-images.githubusercontent.com/52817707/236402874-6baacc7d-9c70-44a2-84c9-e997c76bbf93.png)
 There is <b>yaml</b> file in <b>Config</b> folder for parameters. 
 They save in class name <b>"args"</b>, you can change parameters by easily.<br>

- # 04 Generate mask
 ![04_Mask](https://user-images.githubusercontent.com/52817707/236408448-1ff6ccf8-6860-4468-9290-9bb1829ab759.png)
 ROI is the brain, not the skull or background. You can make a mask in <b>Mask_generate</b> folder. 
 There are 3 files, one for functions(Analysis_Util.py) another for generating mask(GetMask.py), and the other for revising mask(FloodFill.py).
 If you generate <b>GetMask.py</b>, left-click the coordinate for a mask. If you want to undo it, right-click. If you finish clicking every coordinate, press ctrl with the left click.
 If coordinates are less than 3, a mask does not generate. Sometimes masks generate weird. You can revise by <b>Floodfill.py</b>.<br>
 
- # 05 Train and Test 
 ![05_TrainandTest](https://user-images.githubusercontent.com/52817707/236405989-75a6391f-d1a3-40ba-ae85-d4948a92efb7.png)
 There are <b>Train.py</b> and <b>Test.py</b> in <b>bin</b> folder. you can generate both files by <b>Main.py</b>.
 <b>Main.py</b> file make <b>args</b> by yaml file that include every parameters. I make a tensorboard, so you can check the train loss and validation loss function in real time. <br>
 

- # 06 Inference
 ![06_Inference](https://user-images.githubusercontent.com/52817707/236796153-5e465492-221d-4d44-9bbd-5aa9904bee79.png)
 You can make artifact correction dicom data by <b>inference.py</b>. It also needs <b>yaml file</b>(--config) to get parameters, <b>load folder</b>(--load_folder) for dicom path, and <b>save folder</b>(--Save_folder) for saving outputs.