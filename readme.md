The Recognition of Kernel Research Team
===============
Abstract
-----------
Scientific projects are usually created by teams rather than individuals since the realizations of the projects need complex instruments and multidisciplinary cooperations. Although there is a myriad of reports on the assembly mechanisms of research teams, most are restricted to the empirical analysis of some special teams, and they failed to analyze the research team from big co-authorship networks. Inspired by L. G. Adams's ``basic elements'' of the successful research team, this paper proposed a method for identifying the kernel research teams from the co-author networks. We create a database containing all articles published in the journals recommended by the China Computer Federation (CCF), based on which the networks of ten subfields in computer science are constructed. In the empirical analysis, a handful of scholars are found to contribute a large portion of literature and gather numerous citations; this proves the presence of the Pareto principle in academic networks. Furthermore, the information of $34$ famous research teams is collected and analyzed; our study shows most leaders and members who belong to these $34$ teams can be recovered from the network of kernel research teams even when more than $70\%$ of authors are removed from the original co-authorship network. Finally, in order to take full advantage of the authors' research interests, we improve the original label propagation method to guarantee good performance in our dataset.

Requirements
-----------
networkx \
nltk \
pymogno \
gensim \
numpy \
spacy \
pickle \

Usage
----------
1 Unpack the files in /data \
2 Install mongodb, mongodb compass \
3 import the data into mongodb by mongodb compass \
4 run ./kernel_author_detected/detect_core_net.py \
5 run ./team_recognized/LDA_model.py \
6 run ./team_recognized/LPALS.py \
