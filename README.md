# KekeHu_DTLDP

We conducted experiments to asses the performance of DTL-DP and to compare it with existing deep learning-based defect prediction approaches, for both within-project, WPDP, and cross-project, CPDP, defect prediction. We ran experiments on a Linux server with 3 Titan XP GPUs. Unless otherwise stated, each experiment was run 10 times and the average results are reported.

In order to directly compare our work with prior research, we used publicly available data from the PROMISE3 data repository, which has been widely used in defect prediction work [1,2,3,4]. We selected all open source Java projects from this repository, and collected their version number, class names, and the buggy label for each file. Based on the version number and class name, we obtained the source code for each file from Github and fed it to our end-to-end framework. In total, data for 10 Java projects were collected. It should be noted that the average number of files over all projects ranges between 150 and 1046, and the defect rates of the projects have a minimum value of 13.4% and a maximum value of 49.7%. 

[1] Song Wang, Taiyue Liu, and Lin Tan. Automatically learning semantic features for defect prediction. In Software Engineering (ICSE), 2016 IEEE/ACM 38th International Conference on, pages 297–308. IEEE, 2016.  
[2] Jian Li, Pinjia He, Jieming Zhu, and Michael R Lyu. Software defect prediction via convolutional neural network. In Software Quality, Reliability and Security (QRS), 2017 IEEE International Conference on, pages 318–328. IEEE, 2017.  
[3] Hoa Khanh Dam, Trang Pham, Shien Wee Ng, Truyen Tran, John Grundy, Aditya Ghose, Taeksu Kim, and Chul-Joo Kim. A deep tree-based model for software defect prediction. arXiv preprint arXiv:1802.00921, 2018.  
[4] Song Wang, Taiyue Liu, Jaechang Nam, and Lin Tan. Deep semantic feature learning for software defect prediction. IEEE Transactions on Software Engineering, 2018.


Contact person: Ajie Utama, utama@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

Please reach out to us for further questions or if you encounter any issue. You can cite this work by the following:

@InProceedings{UtamaDebias2020,
  author    = {Utama, P. Ajie and Moosavi, Nafise Sadat and Gurevych, Iryna},
  title     = {Mind the Trade-off: Debiasing NLU Models without Degrading the In-distribution Performance},
  booktitle = {Proceedings of the 58th Conference of the Association for Computational Linguistics},
  month     = jul,
  year      = {2020},
  publisher = {Association for Computational Linguistics}
}
