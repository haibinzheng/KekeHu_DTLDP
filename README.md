# KekeHu_DTLDP



We conducted experiments to asses the performance of DTL-DP and to compare it with existing deep learning-based defect prediction approaches, for both within-project, WPDP, and cross-project, CPDP, defect prediction. We ran experiments on a Linux server with 3 Titan XP GPUs. Unless otherwise stated, each experiment was run 10 times and the average results are reported.



In order to directly compare our work with prior research, we used publicly available data from the PROMISE3 data repository, which has been widely used in defect prediction work [1,2,3,4]. We selected all open source Java projects from this repository, and collected their version number, class names, and the buggy label for each file. Based on the version number and class name, we obtained the source code for each file from Github and fed it to our end-to-end framework. In total, data for 10 Java projects were collected. It should be noted that the average number of files over all projects ranges between 150 and 1046, and the defect rates of the projects have a minimum value of 13.4% and a maximum value of 49.7%. 
