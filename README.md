# KekeHu_DTLDP

> **Abstract:**
Software defect prediction aims to automatically locate defective
code modules to better focus testing resources and human effort.
Typically, software defect prediction pipelines are comprised of two
parts: the first extracts program features, like abstract syntax trees,
by using external tools, and the second applies machine learningbased classification models to those features in order to predict
defective modules. Since such approaches depend on specific feature
extraction tools, machine learning classifiers have to be customtailored to effectively build most accurate models.  
To bridge the gap between deep learning and defect prediction,
we propose an end-to-end framework which can directly get prediction results for programs without utilizing feature-extraction tools.
To that end, we first visualize programs as images, apply the selfattention mechanism to extract image features, use transfer learning
to reduce the difference in sample distribution between projects, and
finally feed the image files into a pre-trained deep learning model for
defect prediction. Experiments with 10 open source projects from the
PROMISE dataset show that our method can improve cross-project
and within-project defect prediction.

## Experiments
We conducted experiments to asses the performance of DTL-DP and to compare it with existing deep learning-based defect prediction approaches, for both within-project, WPDP, and cross-project, CPDP, defect prediction. We ran experiments on a Linux server with 3 Titan XP GPUs. Unless otherwise stated, each experiment was run 10 times and the average results are reported.

In order to directly compare our work with prior research, we used publicly available data from the PROMISE3 data repository, which has been widely used in defect prediction work [1,2,3,4]. We selected all open source Java projects from this repository, and collected their version number, class names, and the buggy label for each file. Based on the version number and class name, we obtained the source code for each file from Github and fed it to our end-to-end framework. In total, data for 10 Java projects were collected. It should be noted that the average number of files over all projects ranges between 150 and 1046, and the defect rates of the projects have a minimum value of 13.4% and a maximum value of 49.7%. 

### Dataset Description
|Project |Description |Versions |#Files |Avg files |Avg size(kb) |% Defective|
|-----|-----|-----|-----|-----|-----|-----|
|ant |Java based build tool |1.5,1.6,1.7 |1,465 |488 |6.2 |13.4|
|camel |Enterprise integration framework |1.2,1.4,1.6 |3,140 |1,046 |2.9 |18.7|
|jEdit |Text editor designed for programmers |3.2,4.0,4.1 |1,935 |645 |8.7 |19.2|
|log4j |Logging library for Java |1.0,1.1 |300 |150 |3.4 |49.7|
|lucene |Text search engine library |2.0,2.2,2.4 |607 |402 |3.8 |35.8|
|xalan |A library for transforming XML files |2.4,2.5 1,|984 |992 |4.6 |29.6|
|xerces |XML parser |1.2,1.3 |1,647 |549 |2.9 |15.7|
|ivy |Dependency management library |1.4,2.0 |622 |311 |4.1 |20.0|
|synapse |Data transport adapters |1.0,1.1,1.2 |661 |220 |3.8 |22.7|
|poi |Java library to access Microsoft format files |1.5,2.5,3.0 |1,248 |416 |3.6 |40.7|



[1] Song Wang, Taiyue Liu, and Lin Tan. Automatically learning semantic features for defect prediction. In Software Engineering (ICSE), 2016 IEEE/ACM 38th International Conference on, pages 297–308. IEEE, 2016.  
[2] Jian Li, Pinjia He, Jieming Zhu, and Michael R Lyu. Software defect prediction via convolutional neural network. In Software Quality, Reliability and Security (QRS), 2017 IEEE International Conference on, pages 318–328. IEEE, 2017.  
[3] Hoa Khanh Dam, Trang Pham, Shien Wee Ng, Truyen Tran, John Grundy, Aditya Ghose, Taeksu Kim, and Chul-Joo Kim. A deep tree-based model for software defect prediction. arXiv preprint arXiv:1802.00921, 2018.  
[4] Song Wang, Taiyue Liu, Jaechang Nam, and Lin Tan. Deep semantic feature learning for software defect prediction. IEEE Transactions on Software Engineering, 2018.


## Contact
Contact person: 
Keke Hu (hukeke15@163.com, 2111703352@zjut.edu.cn)

Please reach out to us for further questions or if you encounter any issue.
You can cite this work by the following:
```
@InProceedings{ChenSoftware2020,
  author    = {Jinyin, Chen and Keke, Hu and Yue, Yu and Zhuangzhi, Chen and Qi, Xuan and Yi, Liu and Vladimir, Filkov},
  title     = {Software Visualization and Deep Transfer Learning for Effective Software Defect Prediction},
  booktitle = {Proceedings of the 42th International Conference on Software Engineering},
  month     = May,
  year      = {2020},
  publisher = { Association for Computing Machinery}
}
```
