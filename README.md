## Large-scale Personalized Video Game Recommendation via Social-aware Contextualized Graph Neural Network

> Authors: Liangwei Yang, Zhiwei Liu, Yu Wang, Chen Wang, Ziwei Fan, Philip S. Yu
> Affiliation: University of Illinois at Chicago

![](./assets/draftrec_model.png)


> **Abstract:** 
Because of the large number of online games available nowadays, online game recommender systems are necessary for users and online game platforms. The former can discover more potential online games of their interests, and the latter can attract users to dwell longer in the platform. This paper investigates the characteristics of user behaviors with respect to the online games on the Steam platform. Based on the observations, we argue that a satisfying recommender system for online games is able to characterize: personalization, game contextualization and social connection. However, simultaneously solving all is rather challenging for game recommendation. Firstly, personalization for game recommendation requires the incorporation of the dwelling time of engaged games, which are ignored in existing methods.
Secondly, game contextualization should reflect the complex and high-order properties of those relations. Last but not least, it is problematic to use social connections directly for game recommendations due to the massive noise within social connections. To this end, we propose a Social-aware Contextualized Graph Neural Recommender System~(SCGRec), which harnesses three perspectives to improve game recommendation. We conduct a comprehensive analysis of users' online game behaviors, which motivates the necessity of handling those three characteristics in the online game recommendation.

## Dataset

[Google drive link](https://drive.google.com/file/d/1F9kr_YWimBtexJEH-zkDzCOwl1q7GmFp/view)

![](./assets/dataset.png)

## How to run
python main.py

## Cite


```
@inproceedings{SCGRec,
  author    = {Liangwei Yang and
               Zhiwei Liu and
               Yu Wang and
               Chen Wang and
               Ziwei Fan and
               Philip S. Yu},
  title     = {Large-scale Personalized Video Game Recommendation via Social-aware
               Contextualized Graph Neural Network},
  booktitle = {{WWW} '22: The {ACM} Web Conference 2022, Virtual Event, Lyon, France,
               April 25 - 29, 2022},
  pages     = {3376--3386},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3485447.3512273},
  doi       = {10.1145/3485447.3512273},
  timestamp = {Tue, 26 Apr 2022 16:02:09 +0200},
  biburl    = {https://dblp.org/rec/conf/www/YangLWWFY22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


