# Synthetic-MRI-correction
# List
1. [Propose](#01-Propose)
2. [Method](#02-Method)
- # 01 Propose
 ![01_Synthetic MRI artifact](https://user-images.githubusercontent.com/52817707/236194247-c426ba57-76b0-48a1-8380-e5a9eac0ce6c.png)
 Synthetic MRI obtains multi model mri data by single shot.<br>
 It takes much less time then conventional method but there are several artifacts in T2 FLAIR image.<br>
 In <b>Figure 1</b>, there are bright artifacts in synthetic T2 FLAIR's red box while conventional method do not.<br>
 These artifacts can cause misdiagnosis and must be corrected.<br>

- # 02 Method
 ![02_SyntheticMRI](https://user-images.githubusercontent.com/52817707/236203472-730f24f0-9ea6-487c-9d3f-62f9149ef538.png)
 Synthetic MRI generate T1, T2, PD, T1 FLAIR, T2 FLAIR and STIR. So in this case, 